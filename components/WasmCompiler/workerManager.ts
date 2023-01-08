import { RunStatus, RunStatusListener } from "../Utils/RunStatus";

export class WasmCompilerWorkerManager {
  wasmWorker: Worker | undefined;
  running: boolean;

  constructor() {
    this.wasmWorker = undefined;
    this.running = false;
  }

  invokeCompilerAndRun(
    allSources: Record<string, string>,
    input: string,
    arg: string,
    printer: (text: string) => void,
    statusListener: RunStatusListener
  ): Promise<string> {
    if (this.running) {
      return Promise.reject(
        "Previous instance is still running. Cannot launch another."
      );
    }

    if (!this.wasmWorker) {
      this.wasmWorker = new Worker(
        new URL("../WasmCompiler/worker.ts", import.meta.url)
      );
    }

    return new Promise((resolve, reject) => {
      this.wasmWorker!.onmessage = (event) => {
        if (event.data.log) {
          printer(event.data.log);
        } else if (event.data.error) {
          reject(event.data.error);
          this.running = false;
        } else if (event.data.output) {
          resolve(event.data.output);
          this.running = false;
        } else if (event.data.percentage) {
          statusListener(
            new RunStatus(event.data.label, event.data.percentage)
          );
        }
      };

      this.wasmWorker!.postMessage({
        allSources: allSources,
        input: input,
        arg: arg,
      });
    });
  }
}
