export type Result<T> =
  | { ok: true; value: T; warning: string | undefined }
  | { ok: false; error: string };

export const Ok = <T>(value: T, warning?: string): Result<T> => {
  return { ok: true, value: value, warning: warning };
};

export const Err = <T>(error: string): Result<T> => {
  return { ok: false, error: error };
};
