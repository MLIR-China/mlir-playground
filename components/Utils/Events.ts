// Local compiler environment download initiated.
// - isUpdate: whether this is an update download or a from-scratch download.
type EnvDownloadStart = {
  kind: "EnvDownloadStart";
  props: { isUpdate: boolean };
};
// Local compiler environment download finished.
// - success: whether the download succeeded or not.
type EnvDownloadDone = { kind: "EnvDownloadDone"; props: { success: boolean } };

// User started running the compiler.
// - preset: the code preset mode the user used when running the compiler.
type RunStart = { kind: "RunStart"; props: { preset: string } };
// Compiler finished running.
type RunEnd = { kind: "RunEnd"; props: never };

// User triggered an export of the playground.
// - isLocal: whether the export is a local file download, or creating a shared link.
// - success: whether the export succeeded or not.
type Export = { kind: "Export"; props: { isLocal: boolean; success: boolean } };
// User triggered an import of a playground.
// - isLocal: whether the import is a local file upload, or from a shared link.
// - success: whether the import succeeded or not.
type Import = { kind: "Import"; props: { isLocal: boolean; success: boolean } };

type PlaygroundEvent =
  | EnvDownloadStart
  | EnvDownloadDone
  | RunStart
  | RunEnd
  | Export
  | Import;

export type AllPlaygroundEvents = {
  [EventKind in PlaygroundEvent as EventKind["kind"]]: EventKind["props"];
};
