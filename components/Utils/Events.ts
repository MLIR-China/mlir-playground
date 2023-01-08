type EnvDownloadStart = {
  kind: "EnvDownloadStart";
  props: { isUpdate: boolean };
};
type EnvDownloadDone = { kind: "EnvDownloadDone"; props: { success: boolean } };
type RunStart = { kind: "RunStart"; props: { preset: string } };
type RunEnd = { kind: "RunEnd"; props: never };

type PlaygroundEvent = EnvDownloadStart | EnvDownloadDone | RunStart | RunEnd;

export type AllPlaygroundEvents = {
  [EventKind in PlaygroundEvent as EventKind["kind"]]: EventKind["props"];
};
