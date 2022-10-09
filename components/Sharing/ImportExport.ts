import { FromSchema } from "json-schema-to-ts";

import Schema_0_0_1 from "./schema/0.0.1";

import { newStageStateFromPreset, StageState } from "../State/StageState";
import {
  getPreset,
  getPresetNames,
  presetOption,
} from "../Presets/PresetFactory";

type Schema_0_0_1_Type = FromSchema<typeof Schema_0_0_1>;

export type InternalState = {
  input: string;
  stages: Array<StageState>;
};

export function exportToSchema(
  internalState: InternalState,
  environment: string
): Schema_0_0_1_Type {
  return {
    version: "0.0.1",
    environment: environment,
    input: internalState.input,
    stages: internalState.stages.map((stage) => {
      return {
        preset: stage.preset,
        arguments: stage.additionalRunArgs,
        editors: stage.editorContents.map((contents, index) => {
          return {
            name: getPreset(stage.preset).getPanes()[index].shortName,
            contents: contents,
          };
        }),
        output: stage.output,
      };
    }),
  };
}

// Returns either the parsed InternalState, or an error message.
export function importFromSchema(
  source: Schema_0_0_1_Type
): InternalState | string {
  // Maps each stage into either a parsed StageState, or an error message.
  const stageResults: Array<StageState | string> = source.stages.map(
    (stageSource) => {
      const presetName = stageSource.preset;
      if (!(presetName in getPresetNames())) {
        return `Unknown Preset: ${presetName}`;
      }

      let stage = newStageStateFromPreset(presetName as presetOption);
      const presetProps = getPreset(stage.preset);

      stage.additionalRunArgs = stageSource.arguments;
      stage.output = stageSource.output || "";

      const expectedEditorNames = new Set(
        presetProps.getPanes().map((pane) => pane.shortName)
      );
      const actualEditorNames = new Set(
        stageSource.editors.map((editor) => editor.name)
      );
      if (actualEditorNames != expectedEditorNames) {
        return `Unexpected editor panes. Expected: ${Array.from(
          expectedEditorNames
        ).join(",")}. Got: ${Array.from(actualEditorNames).join(",")}.`;
      }
      stage.editorContents = presetProps.getPanes().map((pane) => {
        for (const sourceEditor of stageSource.editors) {
          if (sourceEditor.name == pane.shortName) {
            return sourceEditor.contents;
          }
        }
        return ""; // Unreachable.
      });

      return stage;
    }
  );

  let errorMsgs = stageResults.filter(
    (stageResult) => typeof stageResult === "string"
  );
  if (errorMsgs.length > 0) {
    return errorMsgs.join(" \n");
  }

  return {
    input: source.input || "",
    stages: stageResults as Array<StageState>,
  };
}
