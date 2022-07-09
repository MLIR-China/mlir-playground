import { PlaygroundFacility } from './PlaygroundFacility';

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
    isCodeEditorEnabled(): boolean { return true; }
    getInputFileName(): string { return "input.mlir"; }
    getOutputFileName(): string { return "output.mlir"; }
    getDefaultCodeFile(): string { return defaultCode; }
    getDefaultInputFile(): string { return defaultMLIRInput; }
    getDefaultAdditionalRunArgs(): string { return "--convert-std-to-llvm"; }
    getRunArgsLeftAddon(): string { return "mlir-opt"; }
    getRunArgsRightAddon(): string { return "input.mlir -o output.mlir"; }
}
