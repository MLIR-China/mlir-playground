import { PlaygroundPreset } from "./PlaygroundPreset";

import { RunStatusListener } from "../Utils/RunStatus";

import Toy from "../Toy/index.js";

const defaultToyInput = `def main() {
  print([[1, 2], [3, 4]]);
}
`;

export class ToyChapter extends PlaygroundPreset {
  chapterNumber: number;
  constructor(chapterNumber: number) {
    super();
    this.chapterNumber = chapterNumber;
  }
  isCodeEditorEnabled(): boolean {
    return false;
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
  getDefaultCodeFile(): string {
    return "// Code editor not applicable to Toy chapter modes.";
  }
  getDefaultInputFile(): string {
    return defaultToyInput;
  }
  getDefaultAdditionalRunArgs(): string {
    return "--emit=mlir";
  }
  getRunArgsLeftAddon(inputFileName: string, outputFileName: string): string {
    return "toy " + inputFileName;
  }
  getRunArgsRightAddon(inputFileName: string, outputFileName: string): string {
    return "";
  }

  run(
    _: string,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string> {
    return Toy.runChapter(this.chapterNumber, input, arg.split(/\s+/), printer);
  }
}
