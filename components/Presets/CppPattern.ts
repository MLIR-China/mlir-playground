import { MlirOpt } from "./MlirOpt";
import { PlaygroundPresetPane } from "./PlaygroundPreset";

const defaultCode = `#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

class MyPattern : public OpRewritePattern<ConstantIntOp> {
  public:
    using OpRewritePattern<ConstantIntOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ConstantIntOp constant, PatternRewriter &rewriter) const override {
        // Implement custom pattern below.
        // The example pattern sets every integer constant to zero.
        if (constant.getValue() == 0) {
            return failure();
        }

        rewriter.setInsertionPoint(constant);
        ConstantIntOp newConstant =
            rewriter.create<ConstantIntOp>(constant->getLoc(), 0, constant.getResult().getType());
        rewriter.replaceOp(constant, {newConstant.getResult()});
        return success();
    }
};

class MyRewritePass : public PassWrapper<MyRewritePass, OperationPass<FuncOp>> {
    StringRef getArgument() const final {
        return "my-rewrite-pass";
    }

    StringRef getDescription() const final {
        return "Applies my custom C++ rewrite patterns on FuncOps.";
    }

    void runOnOperation() {
        RewritePatternSet patterns(&getContext());
        patterns.add<MyPattern>(patterns.getContext());

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::MyRewritePass>();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Custom optimizer driver\\n", registry));
}
`;

export class CppPattern extends MlirOpt {
  getPanes(): Array<PlaygroundPresetPane> {
    let panes = super.getPanes();
    panes[0].defaultEditorContent = defaultCode;
    return panes;
  }
  getDefaultAdditionalRunArgs(): string {
    return "--my-rewrite-pass";
  }
}
