export abstract class PlaygroundFacility {
    abstract isCodeEditorEnabled(): boolean;
    abstract getInputFileName(): string;
    abstract getOutputFileName(): string;
    abstract getDefaultCodeFile(): string;
    abstract getDefaultInputFile(): string;
    abstract getDefaultAdditionalRunArgs(): string;
    abstract getRunArgsLeftAddon(): string;
    abstract getRunArgsRightAddon(): string;

    abstract run(code: string, input: string, arg: string, printer: (text: string) => void): Promise<string>;
}