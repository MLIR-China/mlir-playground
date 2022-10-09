import React from "react";

import {
  defaultPreset,
  getPreset,
  presetOption,
} from "../Presets/PresetFactory";
import { PlaygroundPresetPane } from "../Presets/PlaygroundPreset";

// Stores the configuration of a particular stage.
export type StageState = {
  preset: presetOption;
  additionalRunArgs: string;
  editorContents: Array<string>;
  currentPaneIdx: number | null;
  logs: Array<string>;
  output: string;

  outputEditor: React.MutableRefObject<any>;
  outputEditorWindow: React.MutableRefObject<any>;
};

export function newStageStateFromPreset(
  preset: string = defaultPreset
): StageState {
  const presetProps = getPreset(preset);
  const panes = presetProps.getPanes();
  return {
    preset: preset,
    editorContents: panes.map((pane: PlaygroundPresetPane) => {
      return pane.defaultEditorContent;
    }),
    currentPaneIdx: panes.length > 0 ? 0 : null,
    additionalRunArgs: presetProps.getDefaultAdditionalRunArgs(),
    logs: [],
    output: "",
    outputEditor: React.createRef(),
    outputEditorWindow: React.createRef(),
  };
}

export function stageStateIsDirty(state: StageState): boolean {
  const presetProps = getPreset(state.preset);
  return presetProps.getPanes().some((pane, paneIndex) => {
    const editorContent = state.editorContents[paneIndex];
    return (
      editorContent.trim().length > 0 &&
      editorContent != pane.defaultEditorContent
    );
  });
}
