import WasmFetcher from "../WasmFetcher";

// Due to emscripten generating incompatible syscall APIs, each chapter needs its own wrapper.
import Module1 from "./wasm/toyc-ch1.js";
import Module2 from "./wasm/toyc-ch2.js";
import Module3 from "./wasm/toyc-ch3.js";
import Module4 from "./wasm/toyc-ch4.js";
import Module5 from "./wasm/toyc-ch5.js";
import Module6 from "./wasm/toyc-ch6.js";
import Module7 from "./wasm/toyc-ch7.js";

const ToyWasm = (() => {
  const wasmFetcher = new WasmFetcher();

  const moduleMap = {
    1: Module1,
    2: Module2,
    3: Module3,
    4: Module4,
    5: Module5,
    6: Module6,
    7: Module7,
  };

  let runChapter = (chapterIndex, input, args, printer) => {
    if (chapterIndex < 1 || chapterIndex > 7) {
      console.log(
        "Internal Error: Looking for non-existent Toy chapter: " +
          chapterIndex.toString()
      );
      return Promise.reject(
        "Non-existent Toy chapter: " + chapterIndex.toString()
      );
    }

    const wasmName = "toyc-ch" + chapterIndex.toString() + ".wasm";
    const moduleParams = wasmFetcher.getModuleParams(wasmName, null);
    const module = moduleMap[chapterIndex];

    return module({
      ...moduleParams,
      print: printer,
      printErr: printer,
    }).then((compiledMod) => {
      compiledMod.FS.writeFile("input.toy", input, { encoding: "utf8" });
      console.log("Running toy...");
      try {
        let ret = compiledMod.callMain(["input.toy", ...args]);
        if (ret) {
          return Promise.reject(
            "Failed to run. toy exited with: " + ret.toString()
          );
        }
      } catch (e) {
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
