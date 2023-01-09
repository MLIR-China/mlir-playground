import {
  delMany as idb_delMany,
  getMany as idb_getMany,
  setMany as idb_setMany,
} from "idb-keyval";

import WasmFetcher from "../WasmFetcher";

import {
  LLVM_VERSION,
  SYSTEM_LIB_NAMES,
  LLVM_LIB_FILES,
  LLVM_PACKAGE_CHECKSUM,
} from "./wasm/constants.js";
import ClangModule from "./wasm/clang.mjs";
import LldModule from "./wasm/wasm-ld.mjs";
import TemplateModule from "./template.js";

import { RunStatus, RunStatusListener } from "../Utils/RunStatus";

const LLVM_PACKAGE_CHECKSUM_METADATA_KEY = "llvm_package_checksum";
const LLVM_VERSION_METADATA_KEY = "llvm_version";
const CLANG_DATA_FILE = "onlyincludes.data";
const CLANG_WASM_FILE = "clang.wasm";
const LLD_DATA_FILE = "onlylibs.data";
const LLD_WASM_FILE = "lld.wasm";

class LoggingTimer {
  private startTime: Date;

  constructor() {
    this.startTime = new Date();
  }

  log(prefix: string) {
    const duration = new Date().getTime() - this.startTime.getTime();
    console.log("%s. Elapsed time: %d ms", prefix, duration);
  }
}

const STATUS_FETCHING_DATA = new RunStatus(
  "Fetching clang module and dependencies.",
  10
);
const STATUS_PREPARING_CLANG_MODULE = new RunStatus(
  "Instantiating clang module.",
  20
);
const STATUS_COMPILING_SOURCE_CODE = new RunStatus(
  "Compiling user source code",
  30
);
const STATUS_PREPARING_LLD_MODULE = new RunStatus(
  "Instantiating lld module.",
  50
);
const STATUS_LINKING_BINARIES = new RunStatus("Linking binaries.", 80);
const STATUS_RUNNING_COMPILED_MODULE = new RunStatus(
  "Running compiled user binaries.",
  90
);

class WasmCompiler {
  private static readonly wasmFetcher = WasmFetcher.getSingleton();

  private static readonly SYSTEM_LIB_LINKER_ARGS = SYSTEM_LIB_NAMES.map(
    (name) => "-l" + name
  );

  private _compileSourceToWasm(
    allSources: Record<string, string>,
    printer: (log: string) => void,
    statusListener: RunStatusListener
  ) {
    return WasmCompiler.wasmFetcher
      .getModuleParams(CLANG_WASM_FILE, CLANG_DATA_FILE, printer)
      .then((params) => {
        statusListener(STATUS_PREPARING_CLANG_MODULE);
        return ClangModule(params);
      })
      .then((loadedClangModule) => {
        console.log("Loaded Clang Module!");

        let cppSources: Array<string> = [];
        for (const sourceName in allSources) {
          if (sourceName.endsWith(".cpp")) {
            cppSources.push(sourceName);
          }
          loadedClangModule.FS.writeFile(sourceName, allSources[sourceName]);
        }
        console.log("Saved source code to file!");

        let commonArgs = [
          "-D",
          "EMSCRIPTEN",
          "-isysroot",
          "/",
          "-internal-isystem",
          "/include/c++/v1",
          "-internal-isystem",
          "/clang-15",
          "-internal-isystem",
          "/include",
          "-triple",
          "wasm32-unknown-emscripten",
          "-ferror-limit",
          "19",
          "-iwithsysroot/include/SDL",
          "-iwithsysroot/include/compat",
          "-fgnuc-version=4.2.1",
          "-fvisibility",
          "default",
          "-fno-rtti",
        ];
        statusListener(STATUS_COMPILING_SOURCE_CODE);
        const compileTimer = new LoggingTimer();
        try {
          let ret = loadedClangModule.callMain([
            "-cc1",
            "-emit-obj",
            ...commonArgs,
            "-o",
            "hello.o",
            "-x",
            "c++",
            ...cppSources,
          ]);
          if (ret) {
            return Promise.reject(
              "Failed to compile. Clang exited with: " + ret.toString()
            );
          }
        } catch (e: any) {
          return Promise.reject("Failed to compile. Error: " + e.toString());
        }
        compileTimer.log("Compiled .o object file");

        statusListener(STATUS_PREPARING_LLD_MODULE);
        return WasmCompiler.wasmFetcher
          .getModuleParams(LLD_WASM_FILE, LLD_DATA_FILE, printer)
          .then((params: any) => {
            return LldModule({ ...params, thisProgram: "wasm-ld" });
          })
          .then((loadedLldMod) => {
            console.log("Loaded Lld Module!");

            loadedLldMod.FS.mkdir("/clangmod");
            loadedLldMod.FS.mount(
              loadedLldMod.PROXYFS,
              {
                root: "/",
                fs: loadedClangModule.FS,
              },
              "/clangmod"
            );
            console.log("Mounted proxy fs!");

            const linkOptions = [
              "--strip-debug",
              "--no-entry",
              "--export=main",
              "--import-undefined",
              "--export-table",
              "--export-if-defined=__start_em_asm",
              "--export-if-defined=__stop_em_asm",
              "--export-if-defined=__stdio_exit",
              "--export=emscripten_stack_get_end",
              "--export=emscripten_stack_get_free",
              "--export=emscripten_stack_get_base",
              "--export=emscripten_stack_init",
              "--export=stackSave",
              "--export=stackRestore",
              "--export=stackAlloc",
              "--export=__wasm_call_ctors",
              "--export=__errno_location",
              "-z",
              "stack-size=5242880",
              "--initial-memory=16777216",
              "--max-memory=16777216",
              "--global-base=1024",
            ];
            statusListener(STATUS_LINKING_BINARIES);
            const linkTimer = new LoggingTimer();
            try {
              let ret = loadedLldMod.callMain([
                "/clangmod/hello.o",
                "-o",
                "hello.wasm",
                "-L/lib/wasm32-emscripten",
                ...WasmCompiler.SYSTEM_LIB_LINKER_ARGS,
                ...linkOptions,
                ...LLVM_LIB_FILES,
              ]);
              if (ret) {
                return Promise.reject(
                  "Failed to link. Lld exited with: " + ret.toString()
                );
              }
            } catch (e: any) {
              return Promise.reject("Failed to link. Error: " + e.toString());
            }
            linkTimer.log("Linked .wasm file");
            // compiled executable
            let wasm = loadedLldMod.FS.readFile("hello.wasm");
            console.log(wasm);

            return WebAssembly.compile(wasm);
          });
      });
  }

  private _cachingCompiler = (() => {
    let prev:
      | { sources: Record<string, string>; result: Promise<WebAssembly.Module> }
      | undefined = undefined;

    return (
      allSources: Record<string, string>,
      printer: (log: string) => void,
      statusListener: RunStatusListener
    ) => {
      if (prev && prev.sources === allSources) {
        return prev.result;
      }

      prev = {
        sources: allSources,
        result: this._compileSourceToWasm(allSources, printer, statusListener),
      };
      return prev.result;
    };
  })();

  compileAndRun(
    allSources: Record<string, string>,
    inputMlir: string,
    mlirOptArgs: Array<string>,
    printer: (log: string) => void,
    statusListener: RunStatusListener
  ) {
    statusListener(STATUS_FETCHING_DATA);
    return this._cachingCompiler(allSources, printer, statusListener).then(
      (inst) => {
        console.log(inst);
        return TemplateModule({
          noInitialRun: true,
          thisProgram: "mlir-opt",
          instantiateWasm: (
            imports: WebAssembly.Imports,
            callback: (inst: WebAssembly.Instance) => void
          ) => {
            WebAssembly.instantiate(inst, imports).then((compiledInst) => {
              callback(compiledInst);
            });
          },
          print: printer,
          printErr: printer,
        }).then((compiledMod: any) => {
          statusListener(STATUS_RUNNING_COMPILED_MODULE);
          compiledMod.FS.writeFile("input.mlir", inputMlir);
          console.log("Running mlir-opt...");
          const runTimer = new LoggingTimer();
          try {
            let ret = compiledMod.callMain([
              ...mlirOptArgs,
              "input.mlir",
              "-o",
              "output.mlir",
            ]);
            if (ret) {
              return Promise.reject(
                "Failed to run. mlir-opt exited with: " + ret.toString()
              );
            }
          } catch (e: any) {
            return Promise.reject("Failed to run. Error: " + e.toString());
          }
          runTimer.log("Run successful");
          return compiledMod.FS.readFile("output.mlir", { encoding: "utf8" });
        });
      },
      (fail_msg) => {
        console.log("compileAndRun failed during compile phase.");
        return Promise.reject(fail_msg);
      }
    );
  }

  // Returns whether initialization was successful.
  static initialize(): Promise<boolean> {
    return WasmCompiler.wasmFetcher.invalidateAll().then(() => {
      const fetches = [
        WasmCompiler.wasmFetcher.fetchData(CLANG_DATA_FILE),
        WasmCompiler.wasmFetcher.fetchData(LLD_DATA_FILE),
        WasmCompiler.wasmFetcher.fetchWasm(CLANG_WASM_FILE),
        WasmCompiler.wasmFetcher.fetchWasm(LLD_WASM_FILE),
      ];
      return Promise.all(fetches).then(
        (results) => {
          // set metadata version
          idb_setMany([
            [LLVM_VERSION_METADATA_KEY, LLVM_VERSION],
            [LLVM_PACKAGE_CHECKSUM_METADATA_KEY, LLVM_PACKAGE_CHECKSUM],
          ]);
          return true;
        },
        (err) => {
          idb_delMany([
            LLVM_VERSION_METADATA_KEY,
            LLVM_PACKAGE_CHECKSUM_METADATA_KEY,
          ]);
          return false;
        }
      );
    });
  }

  // If data files are cached, returns the llvm version & whether the cached llvm package matches with what the wasm code expects.
  // If not cached, returns llvm version as an empty string.
  static dataFilesCachedVersion(): Promise<[string, boolean]> {
    return WasmCompiler.wasmFetcher
      .idbCachesExists([CLANG_DATA_FILE, LLD_DATA_FILE])
      .then((cached) => {
        if (cached) {
          return idb_getMany([
            LLVM_VERSION_METADATA_KEY,
            LLVM_PACKAGE_CHECKSUM_METADATA_KEY,
          ]).then(([version, checksum]) => {
            return [version || "", checksum === LLVM_PACKAGE_CHECKSUM];
          });
        }
        return ["", false];
      });
  }
}

export default WasmCompiler;
