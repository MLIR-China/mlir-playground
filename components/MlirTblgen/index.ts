import WasmFetcher from "../WasmFetcher";

import MlirTblgenModule from "./wasm/mlir-tblgen.mjs";

const DATA_FILE = "onlyincludes.data";
const WASM_FILE = "mlir-tblgen.wasm";

class MlirTblgen {
  private static readonly wasmFetcher = WasmFetcher.getSingleton();

  private _run(sourceCode: string, args: Array<string>, printer: (log: string) => void): Promise<string> {
    let output = "";
    const outputPrinter = (text: string) => {
      output += text + "\n";
    };

    // mlir-tblgen outputs to stdout. Use own printer for stdout, and redirect stderr to user's printer.
    return MlirTblgen.wasmFetcher
      .getModuleParams(WASM_FILE, DATA_FILE, outputPrinter, printer)
      .then((params) => {
        return MlirTblgenModule(params);
      })
      .then((loadedModule) => {
        console.log("Loaded mlir-tblgen Module!");

        loadedModule.FS.writeFile("input.td", sourceCode);
        console.log("Saved source code into file!");

        try {
          let ret = loadedModule.callMain([...args, "-I", "/include", "input.td"]);
          if (ret) {
            return Promise.reject("Failed to run. mlir-tblgen exited with: " + ret.toString());
          }
        } catch (e: any) {
          return Promise.reject("Failed to run. Error: " + e.toString());
        }

        return Promise.resolve(output);
      });
  }

  runDRR(sourceCode: string, printer: (log: string) => void): Promise<string> {
    return this._run(sourceCode, ["--gen-rewriters"], printer);
  }
}

export default MlirTblgen;
