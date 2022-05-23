import WasmFetcher from "../WasmFetcher";

import Module from './wasm/toyc.js'

const ToyWasm = (() => {
  const wasmFetcher = WasmFetcher();

  let toyChapters = new Array(7);

  let getToyChapter = (chapterIndex) => {
    if (toyChapters[chapterIndex]) {
      return toyChapters[chapterIndex];
    }

    if (chapterIndex < 1 || chapterIndex > 7) {
      console.log("Internal Error: Looking for non-existent Toy chapter: " + chapterIndex.toString());
      return null;
    }

    const wasmName = "toyc-ch" + chapterIndex.toString() + ".wasm";
    const moduleParams = wasmFetcher.getModuleParams(wasmName, null);

    let wasmInstance = {
      ready: new Promise(resolve => {
        Module({
          ...moduleParams,
          onRuntimeInitialized() {
            wasmInstance = Object.assign(this, {
              ready: Promise.resolve(this),
              runToy: this.cwrap("main", "number", [])
            });
            resolve(this);
          }
        })
      })
    };

    wasmInstance = Object.assign(wasmInstance, {
      runToy: (input_text) => {
        return (
          new Promise(resolve => {
            wasmInstance.ready.then(readyInstance => {
              readyInstance.FS.writeFile("input.toy", input_text, { encoding: "utf8" });
              readyInstance.runToy();
              let output_text = readyInstance.FS.readFile("output.mlir", { encoding: "utf8" });
              resolve(output_text);
            });
          })
        );
      },
      getSource: new Promise(resolve => {
        wasmInstance.ready.then(readyInstance => {
          let input_text = readyInstance.FS.readFile("input.toy", { encoding: "utf8" });
          resolve(input_text);
        });
      })
    });

    toyChapters[chapterIndex] = wasmInstance;
    return wasmSingleton;
  };

  return getToyChapter;
})();

export default ToyWasm;
