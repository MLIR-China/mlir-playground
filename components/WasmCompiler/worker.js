import WasmCompiler from './index.js';

let wasmCompiler = null;
function getWasmCompiler() {
    if (!wasmCompiler) {
        wasmCompiler = WasmCompiler();
    }
    return wasmCompiler;
}

onmessage = function(event) {
    const data = event.data;
    const printer = (text) => {
        postMessage({log: text});
    }
    getWasmCompiler()
        .compileAndRun(data.code, data.input, data.arg.split(/\s+/), printer)
        .then((output) => { postMessage({output: output}); }, (error) => { postMessage({error: error}); });
}
