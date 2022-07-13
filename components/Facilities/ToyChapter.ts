import { PlaygroundFacility } from "./PlaygroundFacility";

import Toy from "../Toy/index.js";

const defaultToyInput = `def main() {
  print([[1, 2], [3, 4]]);
}
`;

export class ToyChapter extends PlaygroundFacility {
  chapterNumber: number;
  constructor(chapterNumber: number) {
    super();
    this.chapterNumber = chapterNumber;
  }
  isCodeEditorEnabled(): boolean {
    return false;
  }
  getInputFileName(): string {
    return "input.toy";
  }
  getOutputFileName(): string {
    return "output.mlir";
  }
  getDefaultCodeFile(): string {
    return "";
  }
  getDefaultInputFile(): string {
    return defaultToyInput;
  }
  getDefaultAdditionalRunArgs(): string {
    return "--emit=mlir";
  }
  getRunArgsLeftAddon(): string {
    return "toy input.toy";
  }
  getRunArgsRightAddon(): string {
    return "";
  }

  run(
    _: string,
    input: string,
    arg: string,
    printer: (text: string) => void
  ): Promise<string> {
    return Toy.runChapter(this.chapterNumber, input, arg.split(/\s+/), printer);
  }
}
