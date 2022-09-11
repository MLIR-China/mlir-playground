import { PlaygroundPreset, PlaygroundPresetPane } from "./PlaygroundPreset";

import { RunStatus, RunStatusListener } from "../Utils/RunStatus";

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
  }
];

export class TableGen extends PlaygroundPreset {
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
    return defaultTableGenInput;
  }
  // /opt/llvm/llvm-native-build/bin/mlir-tblgen --gen-rewriters -I /emsdk/upstream/emscripten/cache/sysroot/include tes.td
  getDefaultAdditionalRunArgs(): string {
    return "--gen-rewriters";
  }
  getRunArgsLeftAddon(inputFileName: string, outputFileName: string): string {
    return "drr.td";
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
        code: code,
        input: input,
        arg: arg,
      });
    });
  }
}
