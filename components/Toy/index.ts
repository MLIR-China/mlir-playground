import WasmFetcher from "../WasmFetcher";

// Due to emscripten generating incompatible syscall APIs, each chapter needs its own wrapper.
import Module1 from "./wasm/toyc-ch1.js";
import Module2 from "./wasm/toyc-ch2.js";
import Module3 from "./wasm/toyc-ch3.js";
import Module4 from "./wasm/toyc-ch4.js";
import Module5 from "./wasm/toyc-ch5.js";

const ToyWasm = (() => {
  const wasmFetcher = WasmFetcher.getSingleton();

  const moduleMap: any = {
    1: Module1,
    2: Module2,
    3: Module3,
    4: Module4,
    5: Module5,
  };

  let runChapter = (
    chapterIndex: number,
    input: string,
    args: Array<String>,
    printer: (log: string) => void
  ) => {
    if (chapterIndex < 1 || chapterIndex > 5) {
      console.log(
        "Internal Error: Looking for non-existent Toy chapter: " +
          chapterIndex.toString()
      );
      return Promise.reject(
        "Non-existent Toy chapter: " + chapterIndex.toString()
      );
    }

    const wasmName = "toyc-ch" + chapterIndex.toString() + ".wasm";
    const module = moduleMap[chapterIndex];

    return wasmFetcher
      .getModuleParams(wasmName, "", printer)
      .then((params) => {
        return module(params);
      })
      .then((compiledMod) => {
        compiledMod.FS.writeFile("input.toy", input, { encoding: "utf8" });
        console.log("Running toy...");
        try {
          let ret = compiledMod.callMain(["input.toy", ...args]);
          if (ret) {
            return Promise.reject(
              "Failed to run. toy exited with: " + ret.toString()
            );
          }
        } catch (e: any) {
          return Promise.reject("Failed to run. Error: " + e.toString());
        }
        // Caveat: No output file. All outputs are emitted via stderr.
        return "(See Logs Window)";
      });
  };

  return {
    runChapter: runChapter,
  };
})();

export default ToyWasm;
