import Module from './toyc-ch7.js'

const ToyWasm = (() => {
  let wasmSingleton = undefined;

  let getWasmInstance = () => {
    if (wasmSingleton) {
      return wasmSingleton;
    }

    let wasmInstance = {
      ready: new Promise(resolve => {
        Module({
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

    wasmSingleton = wasmInstance;
    return wasmSingleton;
  };

  return getWasmInstance;
})();

export default ToyWasm;
