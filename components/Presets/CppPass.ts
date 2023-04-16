import { MlirOpt } from "./MlirOpt";
import { PlaygroundPresetPane } from "./PlaygroundPreset";

const defaultCode = `#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {

class MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
    StringRef getArgument() const final {
        return "my-pass";
    }

    StringRef getDescription() const final {
        return "My custom pass for playing with MLIR.";
    }

    void runOnOperation() {
        // Implement custom pass logic below.
        // The example pass prints to stderr the name of the function being operated on.
        func::FuncOp fn = getOperation();
        llvm::errs() << "Found function: " << fn.getName() << "\\n";
    }
};

}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::MyPass>();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Custom optimizer driver\\n", registry));
}
`;

export class CppPass extends MlirOpt {
  getPanes(): Array<PlaygroundPresetPane> {
    let panes = super.getPanes();
    panes[0].defaultEditorContent = defaultCode;
    return panes;
  }
  getDefaultAdditionalRunArgs(): string {
    return "--my-pass";
  }
}
