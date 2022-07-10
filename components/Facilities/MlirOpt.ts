import { PlaygroundFacility } from './PlaygroundFacility';

import WasmCompiler from '../WasmCompiler/index.js';

const defaultCode =
`#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Custom optimizer driver\\n", registry));
}
`;

const defaultMLIRInput =
`module  {
  func @main() -> i32 {
    %0 = constant 42 : i32
    return %0 : i32
  }
}
`;

export class MlirOpt extends PlaygroundFacility {
    wasmWorker: Worker;
    running: boolean;
    constructor() {
        super();
        this.wasmWorker = new Worker(new URL("../WasmCompiler/worker.js", import.meta.url));
        this.running = false;
    }

    isCodeEditorEnabled(): boolean { return true; }
    getInputFileName(): string { return "input.mlir"; }
    getOutputFileName(): string { return "output.mlir"; }
    getDefaultCodeFile(): string { return defaultCode; }
    getDefaultInputFile(): string { return defaultMLIRInput; }
    getDefaultAdditionalRunArgs(): string { return "--convert-std-to-llvm"; }
    getRunArgsLeftAddon(): string { return "mlir-opt"; }
    getRunArgsRightAddon(): string { return "input.mlir -o output.mlir"; }

    run(code: string, input: string, arg: string, printer: (text: string) => void): Promise<string> {
        if (this.running) {
            return Promise.reject("Previous instance is still running. Cannot launch another.");
        }

        return new Promise((resolve, reject) => {
            this.wasmWorker.onmessage = (event) => {
                if (event.data.log) {
                    printer(event.data.log);
                } else if (event.data.error) {
                    reject(event.data.error);
                    this.running = false;
                } else if (event.data.output) {
                    resolve(event.data.output);
                    this.running = false;
                }
            };

            this.wasmWorker.postMessage({
                code: code,
                input: input,
                arg: arg,
            });
        });
    }
}
