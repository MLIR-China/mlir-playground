import WasmFetcher from "../WasmFetcher";

import Module from './wasm/toyc.js'

const ToyWasm = (() => {
  const wasmFetcher = WasmFetcher();

  let runChapter = (chapterIndex, input, args, printer) => {
    if (chapterIndex < 1 || chapterIndex > 7) {
      console.log("Internal Error: Looking for non-existent Toy chapter: " + chapterIndex.toString());
      return Promise.reject("Non-existent Toy chapter: " + chapterIndex.toString());
    }

    const wasmName = "toyc-ch" + chapterIndex.toString() + ".wasm";
    const moduleParams = wasmFetcher.getModuleParams(wasmName, null);

    return Module({
      ...moduleParams,
      print: printer,
      printErr: printer
    }).then((compiledMod) => {
      compiledMod.FS.writeFile("input.toy", input, { encoding: "utf8" });
      console.log("Running toy...");
      try {
        let ret = compiledMod.callMain(["input.toy", ...args]);
        if (ret) {
            return Promise.reject("Failed to run. toy exited with: " + ret.toString());
        }
      } catch(e) {
        return Promise.reject("Failed to run. Error: " + e.toString());
      }
      // Caveat: No output file. All outputs are emitted via stderr.
      return "(See Logs Window)";
    });
  };

  return {
    runChapter: runChapter
  };
})();

export default ToyWasm;
