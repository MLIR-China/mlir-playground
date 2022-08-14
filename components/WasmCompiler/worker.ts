import WasmCompiler from ".";

import { RunStatus } from "../Utils/RunStatus";

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
  const statusListener = (status: RunStatus) => {
    postMessage({ label: status.label, percentage: status.percentage });
  };
  getWasmCompiler()
    .compileAndRun(
      data.code,
      data.input,
      data.arg.split(/\s+/),
      printer,
      statusListener
    )
    .then(
      (output) => {
        postMessage({ output: output });
      },
      (error) => {
        postMessage({ error: error });
      }
    );
};
