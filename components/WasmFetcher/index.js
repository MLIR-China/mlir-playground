const WasmFetcher = () => {
  const cachedFetcher = {
    _cache: {},
    _fetch_common: function (file_name, post_process) {
      if (!this._cache.hasOwnProperty(file_name)) {
        const full_path = new URL(
          self.location.origin + "/" + process.env.staticFilePrefix + file_name
        );
        this._cache[file_name] = fetch(full_path, {
          credentials: "same-origin",
        }).then((response) => {
          return post_process(response);
        });
      }
      return this._cache[file_name];
    },
    fetch_data: function (package_name) {
      return this._fetch_common(package_name, (response) => {
        return response.arrayBuffer();
      });
    },
    fetch_wasm: function (wasm_name) {
      return this._fetch_common(wasm_name, (response) => {
        return WebAssembly.compileStreaming(response);
      });
    },
  };

  const getModuleParams = (wasmFile, dataFile, printer) => {
    const commonFields = {
      noInitialRun: true,
      locateFile: (path, scriptDir) => {
        return wasmFile;
      },
      instantiateWasm: (imports, callback) => {
        return cachedFetcher
          .fetch_wasm(wasmFile, imports)
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

    if (!dataFile) return commonFields;

    return cachedFetcher.fetch_data(dataFile).then((dataBuffer) => {
      return Promise.resolve({
        ...commonFields,
        getPreloadedPackage: (package_name, _) => {
          return dataBuffer;
        },
      });
    });
  };

  return {
    getModuleParams: getModuleParams,
    fetchData: (package_name) => {
      cachedFetcher.fetch_data(package_name);
    },
    fetchWasm: (wasm_name) => {
      cachedFetcher.fetch_wasm(wasm_name);
    },
  };
};

export default WasmFetcher;
