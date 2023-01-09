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

// User created / imported a share link.
// - success: whether the action succeeded or not.
type CreateShareLink = { kind: "CreateShareLink"; props: { success: boolean } };
type ImportShareLink = { kind: "ImportShareLink"; props: { success: boolean } };

// User triggered a local file export / import
// - success: whether the action succeeded or not.
type FileExport = { kind: "FileExport"; props: never };
type FileImport = { kind: "FileImport"; props: { success: boolean } };

type PlaygroundEvent =
  | EnvDownloadStart
  | EnvDownloadDone
  | RunStart
  | RunEnd
  | CreateShareLink
  | ImportShareLink
  | FileExport
  | FileImport;

export type AllPlaygroundEvents = {
  [EventKind in PlaygroundEvent as EventKind["kind"]]: EventKind["props"];
};
