import { PlaygroundFacility } from './PlaygroundFacility';

const defaultToyInput =
`def main() {
  print([[1, 2], [3, 4]]);
}
`;

export class ToyChapter extends PlaygroundFacility {
    chapterNumber: number;
    constructor(chapterNumber: number) {
        super();
        this.chapterNumber = chapterNumber;
    }
    getInputFileName(): string { return "input.toy"; }
    getOutputFileName(): string { return "output.mlir"; }
    getDefaultCodeFile(): string { return ""; }
    getDefaultInputFile(): string { return defaultToyInput; }
    getDefaultAdditionalRunArgs(): string { return "--emit=mlir"; }
    getRunArgsLeftAddon(): string { return "toy input.toy"; }
    getRunArgsRightAddon(): string { return ""; }
}
