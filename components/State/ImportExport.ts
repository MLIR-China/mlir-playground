import Schema_0_0_1_Type from "../../schema/types/0.0.1";

import { newStageStateFromPreset, StageState } from "./StageState";
import {
  getPreset,
  getPresetNames,
  presetOption,
} from "../Presets/PresetFactory";
import { Result, Ok, Err } from "../Utils/Result";

export type SchemaObjectType = Schema_0_0_1_Type.MLIRPlaygroundState;

export type InternalState = {
  input: string;
  stages: Array<StageState>;
};

export function exportToSchema(
  internalState: InternalState,
  environment: string
): SchemaObjectType {
  return {
    version: "0.0.1",
    environment: environment,
    input: internalState.input,
    stages: internalState.stages.map((stage) => {
      return {
        preset: stage.preset as string,
        arguments: stage.additionalRunArgs,
        editors: stage.editorContents.map((contents, index) => {
          return {
            name: getPreset(stage.preset).getPanes()[index].shortName,
            contents: contents,
          };
        }),
        output: stage.output,
      };
    }) as SchemaObjectType["stages"],
  };
}

// Returns either the parsed InternalState, or an error message.
export function importFromSchema(
  source: SchemaObjectType
): Result<InternalState> {
  // Maps each stage into either a parsed StageState, or an error message.
  const stageResults: Array<Result<StageState>> = source.stages.map(
    (stageSource): Result<StageState> => {
      const presetName = stageSource.preset;
      if (!getPresetNames().includes(presetName)) {
        return Err(`Unknown Preset: ${presetName}`);
      }

      let stage = newStageStateFromPreset(presetName as presetOption);
      const presetProps = getPreset(stage.preset);

      stage.additionalRunArgs = stageSource.arguments;
      stage.output = stageSource.output || "";

      let editorNamesSeen = new Map(
        presetProps.getPanes().map((pane) => [pane.shortName, false])
      );
      let editorNameErrors: Array<string> = [];
      stageSource.editors.forEach((editor) => {
        if (!editorNamesSeen.has(editor.name)) {
          editorNameErrors.push(`Unexpected editor pane: ${editor.name}.`);
        }
        if (editorNamesSeen.get(editor.name)!) {
          editorNameErrors.push(`Duplicate editor pane: ${editor.name}.`);
        }
        editorNamesSeen.set(editor.name, true);
      });
      editorNamesSeen.forEach((val, key) => {
        if (!val) {
          editorNameErrors.push(`Missing editor pane: ${key}.`);
        }
      });
      if (editorNameErrors.length > 0) {
        return Err(editorNameErrors.join("\n"));
      }

      stage.editorContents = presetProps.getPanes().map((pane) => {
        for (const sourceEditor of stageSource.editors) {
          if (sourceEditor.name == pane.shortName) {
            return sourceEditor.contents;
          }
        }
        return ""; // Unreachable.
      });

      return Ok(stage);
    }
  );

  const [validResults, errorMsg]: [Array<StageState>, string] =
    stageResults.reduce(
      ([validResults, errorMsg]: [Array<StageState>, string], stageResult) => {
        if (stageResult.ok) {
          return [[...validResults, stageResult.value], errorMsg];
        }
        return [validResults, errorMsg + " \n" + stageResult.error];
      },
      [[], ""]
    );
  if (errorMsg) {
    return Err(errorMsg);
  }

  return Ok({
    input: source.input || "",
    stages: validResults,
  });
}
