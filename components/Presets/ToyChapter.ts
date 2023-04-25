import {
  PlaygroundPreset,
  PlaygroundPresetPane,
  PlaygroundPresetAction,
} from "./PlaygroundPreset";

import { RunStatusListener } from "../Utils/RunStatus";

import Toy from "../Toy/index";

const simpleToyInput = `def main() {
  print([[1, 2], [3, 4]]);
}
`;

const optimizableToyInput = `def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}`;

const multiFuncToyInput = `def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}`;

export class ToyChapter extends PlaygroundPreset {
  chapterNumber: number;
  constructor(chapterNumber: number) {
    super();
    this.chapterNumber = chapterNumber;
  }
  getPanes(): Array<PlaygroundPresetPane> {
    return [];
  }
  getActions(): Record<string, PlaygroundPresetAction> {
    return {};
  }
  isMultiStageCompatible(): boolean {
    return false;
  }
  getInputFileExtension(): string {
    return "toy";
  }
  getOutputFileExtension(): string {
    return "mlir";
  }
  getDefaultInputFile(): string {
    switch (this.chapterNumber) {
      case 1:
      case 2:
        return simpleToyInput;
      case 3:
        return optimizableToyInput;
      case 4:
      case 5:
        return multiFuncToyInput;
      default:
        return ""; // Unreachable.
    }
  }
  getDefaultAdditionalRunArgs(): string {
    switch (this.chapterNumber) {
      case 1:
        return "-emit=ast";
      case 2:
        return "-emit=mlir";
      case 3:
      case 4:
        return "-emit=mlir -opt";
      case 5:
        return "-emit=mlir-affine -opt";
      default:
        return ""; // Unreachable.
    }
  }
  getRunArgsLeftAddon(inputFileName: string, outputFileName: string): string {
    return "toy " + inputFileName;
  }
  getRunArgsRightAddon(inputFileName: string, outputFileName: string): string {
    return "";
  }

  run(
    _: Array<string>,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string> {
    return Toy.runChapter(this.chapterNumber, input, arg.split(/\s+/), printer);
  }
}
