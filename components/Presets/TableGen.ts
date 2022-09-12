import { MlirOpt } from "./MlirOpt";
import {
  PlaygroundPresetPane,
  PlaygroundPresetAction,
} from "./PlaygroundPreset";

import { RunStatusListener } from "../Utils/RunStatus";

import MlirTblgen from "../MlirTblgen";

const defaultCode = `include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/IR/OpBase.td"

def MakeConstantsZero : Pat<(ConstantOp I32Attr:$_), (ConstantOp ConstantAttr<I32Attr, "0">)>;
`;

const defaultTableGenInput = `module  {
  func @main() -> i32 {
    %0 = constant 42 : i32
    return %0 : i32
  }
}
`;

const presetPanes: Array<PlaygroundPresetPane> = [
  {
    shortName: "DRR",
    defaultEditorContent: defaultCode,
  },
  {
    shortName: "Generated",
    defaultEditorContent: "",
  },
  {
    shortName: "Driver",
    defaultEditorContent: "",
  },
];

export class TableGen extends MlirOpt {
  getPanes(): Array<PlaygroundPresetPane> {
    return presetPanes;
  }
  getActions(): Record<string, PlaygroundPresetAction> {
    return { Generate: this.generate };
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
    return super.run([code[2]], input, arg, printer, statusListener);
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
