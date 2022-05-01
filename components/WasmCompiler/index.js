import ClangModule from "./wasm/clang.mjs";
import LldModule from "./wasm/wasm-ld.mjs";
import TemplateModule from "./template.js";

const WasmCompiler = (() => {
    const project_libraries = [
        "/lib/lib/libMLIRAffine.a",
        "/lib/lib/libMLIRAffineTransforms.a",
        "/lib/lib/libMLIRAffineUtils.a",
        "/lib/lib/libMLIRArmNeon.a",
        "/lib/lib/libMLIRArmSVE.a",
        "/lib/lib/libMLIRArmSVETransforms.a",
        "/lib/lib/libMLIRAsync.a",
        "/lib/lib/libMLIRAsyncTransforms.a",
        "/lib/lib/libMLIRAMX.a",
        "/lib/lib/libMLIRAMXTransforms.a",
        "/lib/lib/libMLIRComplex.a",
        "/lib/lib/libMLIRDLTI.a",
        "/lib/lib/libMLIREmitC.a",
        "/lib/lib/libMLIRGPUOps.a",
        "/lib/lib/libMLIRGPUTransforms.a",
        "/lib/lib/libMLIRLinalgAnalysis.a",
        "/lib/lib/libMLIRLinalg.a",
        "/lib/lib/libMLIRLinalgTransforms.a",
        "/lib/lib/libMLIRLinalgUtils.a",
        "/lib/lib/libMLIRLLVMIRTransforms.a",
        "/lib/lib/libMLIRLLVMIR.a",
        "/lib/lib/libMLIRNVVMIR.a",
        "/lib/lib/libMLIRROCDLIR.a",
        "/lib/lib/libMLIRMath.a",
        "/lib/lib/libMLIRMathTransforms.a",
        "/lib/lib/libMLIRMemRef.a",
        "/lib/lib/libMLIRMemRefTransforms.a",
        "/lib/lib/libMLIRMemRefUtils.a",
        "/lib/lib/libMLIROpenACC.a",
        "/lib/lib/libMLIROpenMP.a",
        "/lib/lib/libMLIRPDL.a",
        "/lib/lib/libMLIRPDLInterp.a",
        "/lib/lib/libMLIRQuant.a",
        "/lib/lib/libMLIRSCF.a",
        "/lib/lib/libMLIRSCFTransforms.a",
        "/lib/lib/libMLIRShape.a",
        "/lib/lib/libMLIRShapeOpsTransforms.a",
        "/lib/lib/libMLIRSparseTensor.a",
        "/lib/lib/libMLIRSparseTensorTransforms.a",
        "/lib/lib/libMLIRSparseTensorUtils.a",
        "/lib/lib/libMLIRSPIRV.a",
        "/lib/lib/libMLIRSPIRVModuleCombiner.a",
        "/lib/lib/libMLIRSPIRVConversion.a",
        "/lib/lib/libMLIRSPIRVTransforms.a",
        "/lib/lib/libMLIRSPIRVUtils.a",
        "/lib/lib/libMLIRStandard.a",
        "/lib/lib/libMLIRStandardOpsTransforms.a",
        "/lib/lib/libMLIRTensor.a",
        "/lib/lib/libMLIRTensorTransforms.a",
        "/lib/lib/libMLIRTosa.a",
        "/lib/lib/libMLIRTosaTransforms.a",
        "/lib/lib/libMLIRVector.a",
        "/lib/lib/libMLIRX86Vector.a",
        "/lib/lib/libMLIRX86VectorTransforms.a",
        "/lib/lib/libMLIRAffineToStandard.a",
        "/lib/lib/libMLIRArmNeon2dToIntr.a",
        "/lib/lib/libMLIRAsyncToLLVM.a",
        "/lib/lib/libMLIRComplexToLLVM.a",
        "/lib/lib/libMLIRComplexToStandard.a",
        "/lib/lib/libMLIRGPUToGPURuntimeTransforms.a",
        "/lib/lib/libMLIRGPUToNVVMTransforms.a",
        "/lib/lib/libMLIRGPUToROCDLTransforms.a",
        "/lib/lib/libMLIRGPUToSPIRV.a",
        "/lib/lib/libMLIRGPUToVulkanTransforms.a",
        "/lib/lib/libMLIRLinalgToLLVM.a",
        "/lib/lib/libMLIRLinalgToSPIRV.a",
        "/lib/lib/libMLIRLinalgToStandard.a",
        "/lib/lib/libMLIRLLVMCommonConversion.a",
        "/lib/lib/libMLIRMathToLibm.a",
        "/lib/lib/libMLIRMathToLLVM.a",
        "/lib/lib/libMLIRMemRefToLLVM.a",
        "/lib/lib/libMLIROpenACCToLLVM.a",
        "/lib/lib/libMLIROpenACCToSCF.a",
        "/lib/lib/libMLIROpenMPToLLVM.a",
        "/lib/lib/libMLIRPDLToPDLInterp.a",
        "/lib/lib/libMLIRSCFToGPU.a",
        "/lib/lib/libMLIRSCFToOpenMP.a",
        "/lib/lib/libMLIRSCFToSPIRV.a",
        "/lib/lib/libMLIRSCFToStandard.a",
        "/lib/lib/libMLIRShapeToStandard.a",
        "/lib/lib/libMLIRSPIRVToLLVM.a",
        "/lib/lib/libMLIRStandardToLLVM.a",
        "/lib/lib/libMLIRStandardToSPIRV.a",
        "/lib/lib/libMLIRTosaToLinalg.a",
        "/lib/lib/libMLIRTosaToSCF.a",
        "/lib/lib/libMLIRTosaToStandard.a",
        "/lib/lib/libMLIRVectorToROCDL.a",
        "/lib/lib/libMLIRVectorToLLVM.a",
        "/lib/lib/libMLIRVectorToGPU.a",
        "/lib/lib/libMLIRVectorToSCF.a",
        "/lib/lib/libMLIRVectorToSPIRV.a",
        "/lib/lib/libMLIRAnalysis.a",
        "/lib/lib/libMLIRCallInterfaces.a",
        "/lib/lib/libMLIRExecutionEngine.a",
        "/lib/lib/libMLIRIR.a",
        "/lib/lib/libMLIRLLVMIR.a",
        "/lib/lib/libMLIRParser.a",
        "/lib/lib/libMLIRPass.a",
        "/lib/lib/libMLIRSideEffectInterfaces.a",
        "/lib/lib/libMLIRSupport.a",
        "/lib/lib/libMLIRTargetLLVMIRExport.a",
        "/lib/lib/libMLIRTransforms.a",
        "/lib/lib/libMLIROptLib.a",
        "/lib/lib/libMLIRLinalgTransforms.a",
        "/lib/lib/libMLIRLinalgAnalysis.a",
        "/lib/lib/libMLIRSCFTransforms.a",
        "/lib/lib/libMLIRSparseTensor.a",
        "/lib/lib/libMLIRNVVMIR.a",
        "/lib/lib/libMLIRGPUToGPURuntimeTransforms.a",
        "/lib/lib/libMLIRAsyncToLLVM.a",
        "/lib/lib/libMLIRROCDLIR.a",
        "/lib/lib/libMLIRSPIRVSerialization.a",
        "/lib/lib/libMLIRSPIRVBinaryUtils.a",
        "/lib/lib/libMLIRVectorToLLVM.a",
        "/lib/lib/libMLIRArmNeon.a",
        "/lib/lib/libMLIRArmSVETransforms.a",
        "/lib/lib/libMLIRArmSVE.a",
        "/lib/lib/libMLIRAMXTransforms.a",
        "/lib/lib/libMLIRAMX.a",
        "/lib/lib/libMLIRX86VectorTransforms.a",
        "/lib/lib/libMLIRX86Vector.a",
        "/lib/lib/libMLIRVectorToSCF.a",
        "/lib/lib/libMLIRStandardOpsTransforms.a",
        "/lib/lib/libMLIRComplex.a",
        "/lib/lib/libMLIRGPUTransforms.a",
        "/lib/lib/libMLIRAsync.a",
        "/lib/lib/libLLVMMIRParser.a",
        "/lib/lib/libMLIRAffineToStandard.a",
        "/lib/lib/libMLIRShape.a",
        "/lib/lib/libMLIRSPIRVUtils.a",
        "/lib/lib/libMLIRMemRefToLLVM.a",
        "/lib/lib/libMLIRStandardToLLVM.a",
        "/lib/lib/libMLIRLLVMCommonConversion.a",
        "/lib/lib/libMLIRLinalgUtils.a",
        "/lib/lib/libMLIRTosaTransforms.a",
        "/lib/lib/libMLIRTosa.a",
        "/lib/lib/libMLIRQuant.a",
        "/lib/lib/libMLIRGPUOps.a",
        "/lib/lib/libMLIRDLTI.a",
        "/lib/lib/libMLIRSPIRVConversion.a",
        "/lib/lib/libMLIRSPIRV.a",
        "/lib/lib/libMLIRTransforms.a",
        "/lib/lib/libMLIRVector.a",
        "/lib/lib/libMLIRAffineUtils.a",
        "/lib/lib/libMLIRTransformUtils.a",
        "/lib/lib/libMLIRLoopAnalysis.a",
        "/lib/lib/libMLIRPresburger.a",
        "/lib/lib/libMLIRRewrite.a",
        "/lib/lib/libMLIRPDLToPDLInterp.a",
        "/lib/lib/libMLIRPDLInterp.a",
        "/lib/lib/libMLIRPDL.a",
        "/lib/lib/libMLIRCopyOpInterface.a",
        "/lib/lib/libMLIRLLVMToLLVMIRTranslation.a",
        "/lib/lib/libMLIRTargetLLVMIRExport.a",
        "/lib/lib/libMLIRLLVMIRTransforms.a",
        "/lib/lib/libMLIROpenACC.a",
        "/lib/lib/libMLIROpenMP.a",
        "/lib/lib/libMLIRLLVMIR.a",
        "/lib/lib/libMLIRTranslation.a",
        "/lib/lib/libLLVMOrcJIT.a",
        "/lib/lib/libLLVMExecutionEngine.a",
        "/lib/lib/libLLVMPasses.a",
        "/lib/lib/libLLVMCoroutines.a",
        "/lib/lib/libLLVMObjCARCOpts.a",
        "/lib/lib/libLLVMRuntimeDyld.a",
        "/lib/lib/libLLVMJITLink.a",
        "/lib/lib/libLLVMOrcTargetProcess.a",
        "/lib/lib/libLLVMOrcShared.a",
        "/lib/lib/libLLVMAsmPrinter.a",
        "/lib/lib/libLLVMDebugInfoDWARF.a",
        "/lib/lib/libLLVMDebugInfoMSF.a",
        "/lib/lib/libLLVMGlobalISel.a",
        "/lib/lib/libLLVMSelectionDAG.a",
        "/lib/lib/libLLVMCodeGen.a",
        "/lib/lib/libLLVMTarget.a",
        "/lib/lib/libLLVMCFGuard.a",
        "/lib/lib/libLLVMMCDisassembler.a",
        "/lib/lib/libLLVMipo.a",
        "/lib/lib/libLLVMBitWriter.a",
        "/lib/lib/libLLVMScalarOpts.a",
        "/lib/lib/libLLVMAggressiveInstCombine.a",
        "/lib/lib/libLLVMInstCombine.a",
        "/lib/lib/libLLVMVectorize.a",
        "/lib/lib/libLLVMFrontendOpenMP.a",
        "/lib/lib/libLLVMIRReader.a",
        "/lib/lib/libLLVMAsmParser.a",
        "/lib/lib/libLLVMLinker.a",
        "/lib/lib/libLLVMInstrumentation.a",
        "/lib/lib/libLLVMTransformUtils.a",
        "/lib/lib/libLLVMAnalysis.a",
        "/lib/lib/libLLVMObject.a",
        "/lib/lib/libLLVMMCParser.a",
        "/lib/lib/libLLVMMC.a",
        "/lib/lib/libLLVMDebugInfoCodeView.a",
        "/lib/lib/libLLVMBitReader.a",
        "/lib/lib/libLLVMTextAPI.a",
        "/lib/lib/libLLVMProfileData.a",
        "/lib/lib/libMLIRPass.a",
        "/lib/lib/libMLIRAnalysis.a",
        "/lib/lib/libMLIRLinalg.a",
        "/lib/lib/libMLIRAffine.a",
        "/lib/lib/libMLIRMath.a",
        "/lib/lib/libMLIRParser.a",
        "/lib/lib/libMLIRSCF.a",
        "/lib/lib/libMLIRMemRef.a",
        "/lib/lib/libMLIRMemRefUtils.a",
        "/lib/lib/libMLIRTensor.a",
        "/lib/lib/libMLIRDialect.a",
        "/lib/lib/libLLVMCore.a",
        "/lib/lib/libLLVMBinaryFormat.a",
        "/lib/lib/libLLVMRemarks.a",
        "/lib/lib/libLLVMBitstreamReader.a",
        "/lib/lib/libMLIRDialectUtils.a",
        "/lib/lib/libMLIRViewLikeInterface.a",
        "/lib/lib/libMLIRStandard.a",
        "/lib/lib/libMLIRCastInterfaces.a",
        "/lib/lib/libMLIRVectorInterfaces.a",
        "/lib/lib/libMLIRSideEffectInterfaces.a",
        "/lib/lib/libMLIRLoopLikeInterface.a",
        "/lib/lib/libMLIRCallInterfaces.a",
        "/lib/lib/libMLIRDataLayoutInterfaces.a",
        "/lib/lib/libMLIRInferTypeOpInterface.a",
        "/lib/lib/libMLIRControlFlowInterfaces.a",
        "/lib/lib/libMLIRIR.a",
        "/lib/lib/libMLIRSupport.a",
        "/lib/lib/libLLVMSupport.a",
        "/lib/lib/libLLVMDemangle.a"
    ];

    const cachedFetcher = {
        _cache: {},
        _fetch_common: function(file_name, post_process) {
            if (!this._cache.hasOwnProperty(file_name)) {
                this._cache[file_name] = fetch(file_name, {
                    credentials: "same-origin"
                }).then((response) => {
                    return post_process(response);
                });
            }
            return this._cache[file_name];
        },
        fetch_data: function(package_name) {
            return this._fetch_common(package_name, (response) => {
                return response.arrayBuffer();
            });
        }
    };

    const getModuleParams = (dataFile) => {
        return cachedFetcher.fetch_data(dataFile).then((dataBuffer) => {
            return Promise.resolve({
                noInitialRun: true,
                getPreloadedPackage: (package_name, _) => {
                    console.log(dataBuffer);
                    return dataBuffer;
                }
            });
        });
    };

    const _compileSourceToWasm = (sourceCode) => {
        return getModuleParams("onlyincludes.data").then((params) => {
            return ClangModule(params);
        }).then(function(loadedClangModule) {
            console.log("Loaded Clang Module!");

            loadedClangModule.FS.writeFile("hello.cpp", sourceCode);
            console.log("Saved source code to file!");
            // console.log(loadedClangModule.FS.readFile("hello.cpp", {encoding: "utf8"}));

            let commonArgs = [
                "-D", "EMSCRIPTEN",
                "-isysroot", "/include",
                "-internal-isystem", "/include/include/wasm32-emscripten/c++/v1",
                "-internal-isystem", "/include/include/c++/v1",
                "-internal-isystem", "/clang-15",
                "-internal-isystem", "/include/include/wasm32-emscripten",
                "-internal-isystem", "/include/include",
                "-triple", "wasm32-unknown-emscripten",
                "-ferror-limit", "19",
                "-iwithsysroot/include/SDL",
                "-iwithsysroot/include/compat",
                "-fgnuc-version=4.2.1",
                "-fvisibility", "default",
                "-fno-rtti"
            ];
            try {
                let ret = loadedClangModule.callMain(["-cc1", "-emit-obj", ...commonArgs, "-o", "hello.o", "-x", "c++", "hello.cpp"]);
                if (ret) {
                    return Promise.reject("Failed to compile. Clang exited with: " + ret.toString());
                }
            } catch(e) {
                return Promise.reject("Failed to compile. Error: " + e.toString());
            }
            console.log("compiled .o object file.");

            return getModuleParams("onlylibs.data").then((params) => {
                return LldModule({...params, thisProgram: "wasm-ld"});
            }).then(function(loadedLldMod) {
                console.log("Loaded Lld Module!");

                loadedLldMod.FS.mkdir("/clangmod");
                loadedLldMod.FS.mount(loadedLldMod.PROXYFS, {
                    root: "/",
                    fs: loadedClangModule.FS
                }, "/clangmod");
                console.log("Mounted proxy fs!");

                const system_libraries = [
                    "-lGL",
                    "-lal",
                    "-lhtml5",
                    "-lstubs-debug",
                    "-lnoexit",
                    "-lc-debug",
                    "-ldlmalloc",
                    "-lcompiler_rt",
                    "-lc++-noexcept",
                    "-lc++abi-noexcept",
                    "-lsockets"
                ];

                const linkOptions = [
                    "--strip-debug",
                    "--no-entry",
                    "--export=main",
                    "--import-undefined",
                    "--export-table",
                    "--export-if-defined=__start_em_asm",
                    "--export-if-defined=__stop_em_asm",
                    "--export-if-defined=__stdio_exit",
                    "--export=emscripten_stack_get_end",
                    "--export=emscripten_stack_get_free",
                    "--export=emscripten_stack_get_base",
                    "--export=emscripten_stack_init",
                    "--export=stackSave",
                    "--export=stackRestore",
                    "--export=stackAlloc",
                    "--export=__wasm_call_ctors",
                    "--export=__errno_location",
                    "-z", "stack-size=5242880",
                    "--initial-memory=16777216",
                    "--max-memory=16777216",
                    "--global-base=1024"
                ];
                try {
                    let ret = loadedLldMod.callMain(["/clangmod/hello.o", "-o", "hello.wasm", "-L/lib/lib/wasm32-emscripten", ...system_libraries, ...linkOptions, ...project_libraries]);
                    if (ret) {
                        return Promise.reject("Failed to link. Lld exited with: " + ret.toString());
                    }
                } catch(e) {
                    return Promise.reject("Failed to link. Error: " + e.toString());
                }
                console.log("Linked .wasm file!");
                // compiled executable
                let wasm = loadedLldMod.FS.readFile("hello.wasm");
                console.log(wasm);

                return WebAssembly.compile(wasm);
            });
        });
    };

    const _cachingCompiler = (() => {
        let prev = {source: "", result: Promise.resolve(null)};
        
        return (sourceCode) => {
            if (prev.source === sourceCode) {
                return prev.result;
            }

            prev = {source: sourceCode, result: _compileSourceToWasm(sourceCode)};
            return prev.result;
        };
    })();

    const compileAndRun = (sourceCode, inputMlir, mlirOptArgs, printer) => {
        return _cachingCompiler(sourceCode).then((inst) => {
            console.log(inst);
            return TemplateModule({
                noInitialRun: true,
                thisProgram: "mlir-opt",
                instantiateWasm: (imports, callback) => {
                    WebAssembly.instantiate(inst, imports).then((compiledInst) => {
                        callback(compiledInst);
                    });
                },
                print: printer,
                printErr: printer
            }).then((compiledMod) => {
                compiledMod.FS.writeFile("input.mlir", inputMlir);
                console.log("Running mlir-opt...");
                try {
                    let ret = compiledMod.callMain([...mlirOptArgs, "input.mlir", "-o", "output.mlir"]);
                    if (ret) {
                        return Promise.reject("Failed to run. mlir-opt exited with: " + ret.toString());
                    }
                } catch(e) {
                    return Promise.reject("Failed to run. Error: " + e.toString());
                }
                return compiledMod.FS.readFile("output.mlir", {encoding: "utf8"});
            });
        }, (fail_msg) => {
            console.log("compileAndRun failed during compile phase.");
            console.log(fail_msg);
        });
    };

    const initialize = () => {
        // prefetch data files
        cachedFetcher.fetch_data("onlyincludes.data");
        cachedFetcher.fetch_data("onlylibs.data");
    }

    return {
        compileAndRun: compileAndRun,
        initialize: initialize
    };
})

export default WasmCompiler;
