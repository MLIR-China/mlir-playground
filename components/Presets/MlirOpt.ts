import {
  PlaygroundPreset,
  PlaygroundPresetPane,
  PlaygroundPresetAction,
} from "./PlaygroundPreset";

import { RunStatusListener } from "../Utils/RunStatus";

import { WasmCompilerWorkerManager } from "../WasmCompiler/workerManager";

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
  wasmWorkerManager: WasmCompilerWorkerManager;
  constructor() {
    super();
    this.wasmWorkerManager = new WasmCompilerWorkerManager();
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
    return this.wasmWorkerManager.invokeCompilerAndRun(
      allSources,
      input,
      arg,
      printer,
      statusListener
    );
  }
}
