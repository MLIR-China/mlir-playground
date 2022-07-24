import { MlirOpt } from "./MlirOpt";
import { PlaygroundFacility } from "./PlaygroundFacility";
import { ToyChapter } from "./ToyChapter";

const FacilityFactory: Record<string, () => PlaygroundFacility> = {
  "Custom mlir-opt": () => {
    return new MlirOpt();
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
  "Toy Chapter 6": () => {
    return new ToyChapter(6);
  },
  "Toy Chapter 7": () => {
    return new ToyChapter(7);
  },
};

let FacilityStorage: Record<string, PlaygroundFacility> = {};

type facilityOption = keyof typeof FacilityFactory;

export const defaultFacility: facilityOption = "Custom mlir-opt";

export function getFacilityNames() {
  return Object.keys(FacilityFactory);
}

export function getFacility(name: string) {
  if (!(name in FacilityStorage)) {
    FacilityStorage[name] = FacilityFactory[name]();
  }
  return FacilityStorage[name];
}
