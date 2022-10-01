import {
  PlaygroundPreset,
  PlaygroundPresetPane,
  PlaygroundPresetAction,
} from "./PlaygroundPreset";

import { RunStatus, RunStatusListener } from "../Utils/RunStatus";

const defaultCode = `#include "mlir/IR/Dialect.h"
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

const defaultMLIRInput = `module  {
  func @main() -> i32 {
    %0 = constant 42 : i32
    return %0 : i32
  }
}
`;

const presetPanes: Array<PlaygroundPresetPane> = [
  {
    shortName: "C++",
    defaultEditorContent: defaultCode,
  },
];

export class MlirOpt extends PlaygroundPreset {
  wasmWorker: Worker | undefined;
  running: boolean;
  constructor() {
    super();
    this.wasmWorker = undefined;
    this.running = false;
  }

  getPanes(): Array<PlaygroundPresetPane> {
    return presetPanes;
  }
  getActions(): Record<string, PlaygroundPresetAction> {
    return {};
  }
  isMultiStageCompatible(): boolean {
    return true;
  }
  getInputFileExtension(): string {
    return "mlir";
  }
  getOutputFileExtension(): string {
    return "mlir";
  }
  getDefaultInputFile(): string {
    return defaultMLIRInput;
  }
  getDefaultAdditionalRunArgs(): string {
    return "--convert-std-to-llvm";
  }
  getRunArgsLeftAddon(inputFileName: string, outputFileName: string): string {
    return "mlir-opt";
  }
  getRunArgsRightAddon(inputFileName: string, outputFileName: string): string {
    return inputFileName + " -o " + outputFileName;
  }

  run(
    code: Array<string>,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string> {
    let allSources: Record<string, string> = {};
    allSources["input.cpp"] = code[0];
    return this.invokeCompilerAndRun(
      allSources,
      input,
      arg,
      printer,
      statusListener
    );
  }

  invokeCompilerAndRun(
    allSources: Record<string, string>,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string> {
    if (this.running) {
      return Promise.reject(
        "Previous instance is still running. Cannot launch another."
      );
    }

    if (!this.wasmWorker) {
      this.wasmWorker = new Worker(
        new URL("../WasmCompiler/worker.ts", import.meta.url)
      );
    }

    return new Promise((resolve, reject) => {
      this.wasmWorker!.onmessage = (event) => {
        if (event.data.log) {
          printer(event.data.log);
        } else if (event.data.error) {
          reject(event.data.error);
          this.running = false;
        } else if (event.data.output) {
          resolve(event.data.output);
          this.running = false;
        } else if (event.data.percentage) {
          statusListener(
            new RunStatus(event.data.label, event.data.percentage)
          );
        }
      };

      this.wasmWorker!.postMessage({
        allSources: allSources,
        input: input,
        arg: arg,
      });
    });
  }
}
