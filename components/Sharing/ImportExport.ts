import Ajv from "ajv";
import { FromSchema } from "json-schema-to-ts";

import Schema_0_0_1 from "./schema/0.0.1";

import { newStageStateFromPreset, StageState } from "../State/StageState";
import {
  getPreset,
  getPresetNames,
  presetOption,
} from "../Presets/PresetFactory";

export const SchemaLatest = Schema_0_0_1;

export type SchemaObjectType = FromSchema<typeof SchemaLatest>;

export type InternalState = {
  input: string;
  stages: Array<StageState>;
};

const ajv = new Ajv();
const validate = ajv.compile(SchemaLatest);

// Returns an error message. If empty, means validation passed.
export function validateAgainstSchema(source: any): string {
  if (validate(source)) {
    return "";
  }
  return ajv.errorsText(validate.errors);
}

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
  source: SchemaObjectType
): InternalState | string {
  // Maps each stage into either a parsed StageState, or an error message.
  const stageResults: Array<StageState | string> = source.stages.map(
    (stageSource) => {
      const presetName = stageSource.preset;
      if (!getPresetNames().includes(presetName)) {
        return `Unknown Preset: ${presetName}`;
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
        return editorNameErrors.join("\n");
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
