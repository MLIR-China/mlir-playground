import { CppPass } from "./CppPass";
import { CppPattern } from "./CppPattern";
import { MlirOpt } from "./MlirOpt";
import { PlaygroundPreset } from "./PlaygroundPreset";
import { TableGen } from "./TableGen";
import { ToyChapter } from "./ToyChapter";

const PresetFactory = {
  "Custom mlir-opt": () => {
    return new MlirOpt();
  },
  "C++ Pass": () => {
    return new CppPass();
  },
  "C++ Pattern": () => {
    return new CppPattern();
  },
  "TableGen DRR": () => {
    return new TableGen();
  },
  "Toy Chapter 1": () => {
    return new ToyChapter(1);
  },
  "Toy Chapter 2": () => {
    return new ToyChapter(2);
  },
  "Toy Chapter 3": () => {
    return new ToyChapter(3);
  },
  "Toy Chapter 4": () => {
    return new ToyChapter(4);
  },
  "Toy Chapter 5": () => {
    return new ToyChapter(5);
  },
} as const;

let PresetStorage: Record<string, PlaygroundPreset> = {};

export type presetOption = keyof typeof PresetFactory;

export const defaultPreset: presetOption = "TableGen DRR";

export function getPresetNames() {
  return Object.keys(PresetFactory);
}

export function getPreset(name: presetOption) {
  if (!(name in PresetStorage)) {
    PresetStorage[name] = PresetFactory[name]();
  }
  return PresetStorage[name];
}
