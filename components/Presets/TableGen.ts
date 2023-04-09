import { CppPattern } from "./CppPattern";
import {
  PlaygroundPresetPane,
  PlaygroundPresetAction,
} from "./PlaygroundPreset";

import { RunStatusListener } from "../Utils/RunStatus";

import MlirTblgen from "../MlirTblgen";

const defaultDrrCode = `include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def NotZero: Constraint<CPred<"$_self.cast<IntegerAttr>().getInt() != 0">, "not zero">;

def MakeConstantsZero : Pat<(Arith_ConstantOp I32Attr:$value),
                            (Arith_ConstantOp ConstantAttr<I32Attr, "0">),
                            [(NotZero:$value)]>;
`;

const defaultTableGenInput = `module  {
  func.func @main() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}`;

const defaultCppCode = `#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

// This references the generated file. Do not remove.
#include "generated.h"

class MyRewritePass : public PassWrapper<MyRewritePass, OperationPass<ModuleOp>> {
    StringRef getArgument() const final {
        return "my-rewrite-pass";
    }

    StringRef getDescription() const final {
        return "Applies my DDR rewrite patterns.";
    }

    void runOnOperation() {
        RewritePatternSet patterns(&getContext());
        populateWithGenerated(patterns);

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

const presetPanes: Array<PlaygroundPresetPane> = [
  {
    shortName: "DRR",
    defaultEditorContent: defaultDrrCode,
  },
  {
    shortName: "Generated",
    defaultEditorContent: "",
  },
  {
    shortName: "Driver",
    defaultEditorContent: defaultCppCode,
  },
];

export class TableGen extends CppPattern {
  getPanes(): Array<PlaygroundPresetPane> {
    return presetPanes;
  }
  getActions(): Record<string, PlaygroundPresetAction> {
    return { "Generate Rewriters": this.generate };
  }
  getDefaultInputFile(): string {
    return defaultTableGenInput;
  }

  run(
    code: Array<string>,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string> {
    // Report error if user provided an empty "generated.h" file.
    // This is likely not what they had intended.
    if (code[1].trim().length == 0) {
      return Promise.reject(
        "'Generated' file is empty. Please run 'Generate Rewriters' first."
      );
    }

    let allSources: Record<string, string> = {};
    allSources["generated.h"] = code[1];
    allSources["driver.cpp"] = code[2];
    return this.invokeCompilerAndRun(
      allSources,
      input,
      arg,
      printer,
      statusListener
    );
  }

  generate(
    sources: Array<string>,
    printer: (text: string) => void
  ): Promise<Array<string>> {
    const tblgen = new MlirTblgen();
    return tblgen
      .runDRR(sources[0], printer)
      .then((output) => [sources[0], output, sources[2]]);
  }
}
