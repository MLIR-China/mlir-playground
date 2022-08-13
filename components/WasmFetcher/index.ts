import {
  get as idb_get,
  getMany as idb_getMany,
  set as idb_set,
} from "idb-keyval";

class WasmFetcher {
  private static singleton: WasmFetcher;
  private constructor() {}
  static getSingleton(): WasmFetcher {
    if (!WasmFetcher.singleton) {
      WasmFetcher.singleton = new WasmFetcher();
    }
    return WasmFetcher.singleton;
  }

  private fetchCommon<ResultType>(
    file_name: string,
    post_process: (resp: Response) => Promise<ResultType>
  ): Promise<ResultType> {
    return idb_get(file_name).then((data) => {
      if (data) {
        return data;
      }

      const full_path = new URL(
        self.location.origin + "/" + process.env.staticFilePrefix + file_name
      );
      return fetch(full_path, {
        credentials: "same-origin",
      }).then((response) => {
        const data_promise = post_process(response);
        data_promise.then((processed_data) => {
          idb_set(file_name, processed_data).then(
            () => {
              console.log("Successfully cached data in idb: " + file_name);
            },
            (err) => {
              console.log("Failed to cache data in idb: " + file_name, err);
            }
          );
        });
        return data_promise;
      });
    });
  }

  fetchData(package_name: string): Promise<ArrayBuffer> {
    return this.fetchCommon(package_name, (response: Response) => {
      return response.arrayBuffer();
    });
  }

  // Important: While idb is shared between main and workers,
  // this local wasmCache is unique to each memory space.
  private wasmCache: Record<string, WebAssembly.Module> = {};
  fetchWasm(wasm_name: string): Promise<WebAssembly.Module> {
    if (wasm_name in this.wasmCache) {
      return Promise.resolve(this.wasmCache[wasm_name]);
    }

    return this.fetchData(wasm_name).then((buf: ArrayBuffer) => {
      return WebAssembly.compile(buf).then((compiledModule) => {
        this.wasmCache[wasm_name] = compiledModule;
        return compiledModule;
      });
    });
  }

  getModuleParams(
    wasmFile: string,
    dataFile: string,
    printer: (log: string) => void
  ) {
    const commonFields = {
      noInitialRun: true,
      locateFile: (path: string, scriptDir: string) => {
        return wasmFile;
      },
      instantiateWasm: (
        imports: WebAssembly.Imports,
        callback: (inst: WebAssembly.Instance) => void
      ) => {
        return this.fetchWasm(wasmFile)
          .then((module) => {
            return WebAssembly.instantiate(module, imports);
          })
          .then((instance) => {
            callback(instance);
            return instance.exports;
          });
      },
      print: printer,
      printErr: printer,
    };

    if (!dataFile) return Promise.resolve(commonFields);

    return this.fetchData(dataFile).then((dataBuffer) => {
      return Promise.resolve({
        ...commonFields,
        getPreloadedPackage: (package_name: string, package_size: number) => {
          return dataBuffer;
        },
      });
    });
  }

  // Returns true if all the keys currently exist in idb cache.
  idbCachesExists(keys: Array<string>): Promise<boolean> {
    return idb_getMany(keys).then(
      (vals) => vals.every((val) => !!val),
      () => false
    );
  }
}

export default WasmFetcher;
