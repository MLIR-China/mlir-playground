import WasmCompiler from ".";

let wasmCompiler: WasmCompiler | undefined = undefined;
function getWasmCompiler() {
  if (!wasmCompiler) {
    wasmCompiler = new WasmCompiler();
  }
  return wasmCompiler;
}

onmessage = function (event) {
  const data = event.data;
  const printer = (text: string) => {
    postMessage({ log: text });
  };
  getWasmCompiler()
    .compileAndRun(data.code, data.input, data.arg.split(/\s+/), printer)
    .then(
      (output) => {
        postMessage({ output: output });
      },
      (error) => {
        postMessage({ error: error });
      }
    );
};
