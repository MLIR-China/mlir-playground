export abstract class PlaygroundFacility {
    abstract getInputFileName(): string;
    abstract getOutputFileName(): string;
    abstract getDefaultCodeFile(): string;
    abstract getDefaultInputFile(): string;
    abstract getDefaultAdditionalRunArgs(): string;
    abstract getRunArgsLeftAddon(): string;
    abstract getRunArgsRightAddon(): string;
}
