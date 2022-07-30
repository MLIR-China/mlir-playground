import { MlirOpt } from "./MlirOpt";

const defaultCode = `#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MlirOptMain.h"

namespace mlir {

class MyPass : public PassWrapper<MyPass, OperationPass<FuncOp>> {
    StringRef getArgument() const final {
        return "my-pass";
    }

    StringRef getDescription() const final {
        return "My custom pass for playing with MLIR.";
    }

    void runOnOperation() {
        // Implement custom pass logic below
        auto fn = getOperation();
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
  getDefaultCodeFile(): string {
    return defaultCode;
  }
  getDefaultAdditionalRunArgs(): string {
    return "--my-pass";
  }
}
