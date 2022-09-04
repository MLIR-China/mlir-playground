import { RunStatusListener } from "../Utils/RunStatus";

export abstract class PlaygroundPreset {
  abstract isCodeEditorEnabled(): boolean;
  abstract getInputFileExtension(): string;
  abstract getOutputFileExtension(): string;
  abstract getDefaultCodeFile(): string;
  abstract getDefaultInputFile(): string;
  abstract getDefaultAdditionalRunArgs(): string;
  abstract getRunArgsLeftAddon(
    inputFileName: string,
    outputFileName: string
  ): string;
  abstract getRunArgsRightAddon(
    inputFileName: string,
    outputFileName: string
  ): string;

  abstract run(
    code: string,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string>;
}
