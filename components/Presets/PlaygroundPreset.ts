import { RunStatusListener } from "../Utils/RunStatus";

export type PlaygroundPresetPane = {
  // The name of this pane to display in the tabs.
  shortName: string;
  // The default editor content when user first initializes this pane.
  defaultEditorContent: string;
};

// An action accepts all the editor contents as input, and outputs the new set of editor contents.
// An action is a non-run action.
export type PlaygroundPresetAction = (
  sources: Array<string>,
  printer: (text: string) => void
) => Promise<Array<string>>;

export abstract class PlaygroundPreset {
  abstract getPanes(): Array<PlaygroundPresetPane>;
  abstract getActions(): Record<string, PlaygroundPresetAction>;

  abstract isMultiStageCompatible(): boolean;
  abstract getInputFileExtension(): string;
  abstract getOutputFileExtension(): string;
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
    code: Array<string>,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string>;
}
