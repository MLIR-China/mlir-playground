import WasmFetcher from "../WasmFetcher";

import ClangModule from "./wasm/clang.mjs";
import LldModule from "./wasm/wasm-ld.mjs";
import TemplateModule from "./template.js";

const WasmCompiler = () => {
  const wasmFetcher = WasmFetcher.getSingleton();

  const project_libraries = [
    "/lib/libMLIRAffine.a",
    "/lib/libMLIRAffineTransforms.a",
    "/lib/libMLIRAffineUtils.a",
    "/lib/libMLIRArmNeon.a",
    "/lib/libMLIRArmSVE.a",
    "/lib/libMLIRArmSVETransforms.a",
    "/lib/libMLIRAsync.a",
    "/lib/libMLIRAsyncTransforms.a",
    "/lib/libMLIRAMX.a",
    "/lib/libMLIRAMXTransforms.a",
    "/lib/libMLIRComplex.a",
    "/lib/libMLIRDLTI.a",
    "/lib/libMLIREmitC.a",
    "/lib/libMLIRGPUOps.a",
    "/lib/libMLIRGPUTransforms.a",
    "/lib/libMLIRLinalgAnalysis.a",
    "/lib/libMLIRLinalg.a",
    "/lib/libMLIRLinalgTransforms.a",
    "/lib/libMLIRLinalgUtils.a",
    "/lib/libMLIRLLVMIRTransforms.a",
    "/lib/libMLIRLLVMIR.a",
    "/lib/libMLIRNVVMIR.a",
    "/lib/libMLIRROCDLIR.a",
    "/lib/libMLIRMath.a",
    "/lib/libMLIRMathTransforms.a",
    "/lib/libMLIRMemRef.a",
    "/lib/libMLIRMemRefTransforms.a",
    "/lib/libMLIRMemRefUtils.a",
    "/lib/libMLIROpenACC.a",
    "/lib/libMLIROpenMP.a",
    "/lib/libMLIRPDL.a",
    "/lib/libMLIRPDLInterp.a",
    "/lib/libMLIRQuant.a",
    "/lib/libMLIRSCF.a",
    "/lib/libMLIRSCFTransforms.a",
    "/lib/libMLIRShape.a",
    "/lib/libMLIRShapeOpsTransforms.a",
    "/lib/libMLIRSparseTensor.a",
    "/lib/libMLIRSparseTensorTransforms.a",
    "/lib/libMLIRSparseTensorUtils.a",
    "/lib/libMLIRSPIRV.a",
    "/lib/libMLIRSPIRVModuleCombiner.a",
    "/lib/libMLIRSPIRVConversion.a",
    "/lib/libMLIRSPIRVTransforms.a",
    "/lib/libMLIRSPIRVUtils.a",
    "/lib/libMLIRStandard.a",
    "/lib/libMLIRStandardOpsTransforms.a",
    "/lib/libMLIRTensor.a",
    "/lib/libMLIRTensorTransforms.a",
    "/lib/libMLIRTosa.a",
    "/lib/libMLIRTosaTransforms.a",
    "/lib/libMLIRVector.a",
    "/lib/libMLIRX86Vector.a",
    "/lib/libMLIRX86VectorTransforms.a",
    "/lib/libMLIRAffineToStandard.a",
    "/lib/libMLIRArmNeon2dToIntr.a",
    "/lib/libMLIRAsyncToLLVM.a",
    "/lib/libMLIRComplexToLLVM.a",
    "/lib/libMLIRComplexToStandard.a",
    "/lib/libMLIRGPUToGPURuntimeTransforms.a",
    "/lib/libMLIRGPUToNVVMTransforms.a",
    "/lib/libMLIRGPUToROCDLTransforms.a",
    "/lib/libMLIRGPUToSPIRV.a",
    "/lib/libMLIRGPUToVulkanTransforms.a",
    "/lib/libMLIRLinalgToLLVM.a",
    "/lib/libMLIRLinalgToSPIRV.a",
    "/lib/libMLIRLinalgToStandard.a",
    "/lib/libMLIRLLVMCommonConversion.a",
    "/lib/libMLIRMathToLibm.a",
    "/lib/libMLIRMathToLLVM.a",
    "/lib/libMLIRMemRefToLLVM.a",
    "/lib/libMLIROpenACCToLLVM.a",
    "/lib/libMLIROpenACCToSCF.a",
    "/lib/libMLIROpenMPToLLVM.a",
    "/lib/libMLIRPDLToPDLInterp.a",
    "/lib/libMLIRSCFToGPU.a",
    "/lib/libMLIRSCFToOpenMP.a",
    "/lib/libMLIRSCFToSPIRV.a",
    "/lib/libMLIRSCFToStandard.a",
    "/lib/libMLIRShapeToStandard.a",
    "/lib/libMLIRSPIRVToLLVM.a",
    "/lib/libMLIRStandardToLLVM.a",
    "/lib/libMLIRStandardToSPIRV.a",
    "/lib/libMLIRTosaToLinalg.a",
    "/lib/libMLIRTosaToSCF.a",
    "/lib/libMLIRTosaToStandard.a",
    "/lib/libMLIRVectorToROCDL.a",
    "/lib/libMLIRVectorToLLVM.a",
    "/lib/libMLIRVectorToGPU.a",
    "/lib/libMLIRVectorToSCF.a",
    "/lib/libMLIRVectorToSPIRV.a",
    "/lib/libMLIRAnalysis.a",
    "/lib/libMLIRCallInterfaces.a",
    "/lib/libMLIRExecutionEngine.a",
    "/lib/libMLIRIR.a",
    "/lib/libMLIRLLVMIR.a",
    "/lib/libMLIRParser.a",
    "/lib/libMLIRPass.a",
    "/lib/libMLIRSideEffectInterfaces.a",
    "/lib/libMLIRSupport.a",
    "/lib/libMLIRTargetLLVMIRExport.a",
    "/lib/libMLIRTransforms.a",
    "/lib/libMLIROptLib.a",
    "/lib/libMLIRLinalgTransforms.a",
    "/lib/libMLIRLinalgAnalysis.a",
    "/lib/libMLIRSCFTransforms.a",
    "/lib/libMLIRSparseTensor.a",
    "/lib/libMLIRNVVMIR.a",
    "/lib/libMLIRGPUToGPURuntimeTransforms.a",
    "/lib/libMLIRAsyncToLLVM.a",
    "/lib/libMLIRROCDLIR.a",
    "/lib/libMLIRSPIRVSerialization.a",
    "/lib/libMLIRSPIRVBinaryUtils.a",
    "/lib/libMLIRVectorToLLVM.a",
    "/lib/libMLIRArmNeon.a",
    "/lib/libMLIRArmSVETransforms.a",
    "/lib/libMLIRArmSVE.a",
    "/lib/libMLIRAMXTransforms.a",
    "/lib/libMLIRAMX.a",
    "/lib/libMLIRX86VectorTransforms.a",
    "/lib/libMLIRX86Vector.a",
    "/lib/libMLIRVectorToSCF.a",
    "/lib/libMLIRStandardOpsTransforms.a",
    "/lib/libMLIRComplex.a",
    "/lib/libMLIRGPUTransforms.a",
    "/lib/libMLIRAsync.a",
    "/lib/libLLVMMIRParser.a",
    "/lib/libMLIRAffineToStandard.a",
    "/lib/libMLIRShape.a",
    "/lib/libMLIRSPIRVUtils.a",
    "/lib/libMLIRMemRefToLLVM.a",
    "/lib/libMLIRStandardToLLVM.a",
    "/lib/libMLIRLLVMCommonConversion.a",
    "/lib/libMLIRLinalgUtils.a",
    "/lib/libMLIRTosaTransforms.a",
    "/lib/libMLIRTosa.a",
    "/lib/libMLIRQuant.a",
    "/lib/libMLIRGPUOps.a",
    "/lib/libMLIRDLTI.a",
    "/lib/libMLIRSPIRVConversion.a",
    "/lib/libMLIRSPIRV.a",
    "/lib/libMLIRTransforms.a",
    "/lib/libMLIRVector.a",
    "/lib/libMLIRAffineUtils.a",
    "/lib/libMLIRTransformUtils.a",
    "/lib/libMLIRLoopAnalysis.a",
    "/lib/libMLIRPresburger.a",
    "/lib/libMLIRRewrite.a",
    "/lib/libMLIRPDLToPDLInterp.a",
    "/lib/libMLIRPDLInterp.a",
    "/lib/libMLIRPDL.a",
    "/lib/libMLIRCopyOpInterface.a",
    "/lib/libMLIRLLVMToLLVMIRTranslation.a",
    "/lib/libMLIRTargetLLVMIRExport.a",
    "/lib/libMLIRLLVMIRTransforms.a",
    "/lib/libMLIROpenACC.a",
    "/lib/libMLIROpenMP.a",
    "/lib/libMLIRLLVMIR.a",
    "/lib/libMLIRTranslation.a",
    "/lib/libLLVMOrcJIT.a",
    "/lib/libLLVMExecutionEngine.a",
    "/lib/libLLVMPasses.a",
    "/lib/libLLVMCoroutines.a",
    "/lib/libLLVMObjCARCOpts.a",
    "/lib/libLLVMRuntimeDyld.a",
    "/lib/libLLVMJITLink.a",
    "/lib/libLLVMOrcTargetProcess.a",
    "/lib/libLLVMOrcShared.a",
    "/lib/libLLVMAsmPrinter.a",
    "/lib/libLLVMDebugInfoDWARF.a",
    "/lib/libLLVMDebugInfoMSF.a",
    "/lib/libLLVMGlobalISel.a",
    "/lib/libLLVMSelectionDAG.a",
    "/lib/libLLVMCodeGen.a",
    "/lib/libLLVMTarget.a",
    "/lib/libLLVMCFGuard.a",
    "/lib/libLLVMMCDisassembler.a",
    "/lib/libLLVMipo.a",
    "/lib/libLLVMBitWriter.a",
    "/lib/libLLVMScalarOpts.a",
    "/lib/libLLVMAggressiveInstCombine.a",
    "/lib/libLLVMInstCombine.a",
    "/lib/libLLVMVectorize.a",
    "/lib/libLLVMFrontendOpenMP.a",
    "/lib/libLLVMIRReader.a",
    "/lib/libLLVMAsmParser.a",
    "/lib/libLLVMLinker.a",
    "/lib/libLLVMInstrumentation.a",
    "/lib/libLLVMTransformUtils.a",
    "/lib/libLLVMAnalysis.a",
    "/lib/libLLVMObject.a",
    "/lib/libLLVMMCParser.a",
    "/lib/libLLVMMC.a",
    "/lib/libLLVMDebugInfoCodeView.a",
    "/lib/libLLVMBitReader.a",
    "/lib/libLLVMTextAPI.a",
    "/lib/libLLVMProfileData.a",
    "/lib/libMLIRPass.a",
    "/lib/libMLIRAnalysis.a",
    "/lib/libMLIRLinalg.a",
    "/lib/libMLIRAffine.a",
    "/lib/libMLIRMath.a",
    "/lib/libMLIRParser.a",
    "/lib/libMLIRSCF.a",
    "/lib/libMLIRMemRef.a",
    "/lib/libMLIRMemRefUtils.a",
    "/lib/libMLIRTensor.a",
    "/lib/libMLIRDialect.a",
    "/lib/libLLVMCore.a",
    "/lib/libLLVMBinaryFormat.a",
    "/lib/libLLVMRemarks.a",
    "/lib/libLLVMBitstreamReader.a",
    "/lib/libMLIRDialectUtils.a",
    "/lib/libMLIRViewLikeInterface.a",
    "/lib/libMLIRStandard.a",
    "/lib/libMLIRCastInterfaces.a",
    "/lib/libMLIRVectorInterfaces.a",
    "/lib/libMLIRSideEffectInterfaces.a",
    "/lib/libMLIRLoopLikeInterface.a",
    "/lib/libMLIRCallInterfaces.a",
    "/lib/libMLIRDataLayoutInterfaces.a",
    "/lib/libMLIRInferTypeOpInterface.a",
    "/lib/libMLIRControlFlowInterfaces.a",
    "/lib/libMLIRIR.a",
    "/lib/libMLIRSupport.a",
    "/lib/libLLVMSupport.a",
    "/lib/libLLVMDemangle.a",
  ];

  const _compileSourceToWasm = (sourceCode, printer) => {
    return wasmFetcher
      .getModuleParams("clang.wasm", "onlyincludes.data", printer)
      .then((params) => {
        return ClangModule(params);
      })
      .then(function (loadedClangModule) {
        console.log("Loaded Clang Module!");

        loadedClangModule.FS.writeFile("hello.cpp", sourceCode);
        console.log("Saved source code to file!");

        let commonArgs = [
          "-D",
          "EMSCRIPTEN",
          "-isysroot",
          "/",
          "-internal-isystem",
          "/include/c++/v1",
          "-internal-isystem",
          "/clang-15",
          "-internal-isystem",
          "/include",
          "-triple",
          "wasm32-unknown-emscripten",
          "-ferror-limit",
          "19",
          "-iwithsysroot/include/SDL",
          "-iwithsysroot/include/compat",
          "-fgnuc-version=4.2.1",
          "-fvisibility",
          "default",
          "-fno-rtti",
        ];
        const compileStartTime = new Date();
        try {
          let ret = loadedClangModule.callMain([
            "-cc1",
            "-emit-obj",
            ...commonArgs,
            "-o",
            "hello.o",
            "-x",
            "c++",
            "hello.cpp",
          ]);
          if (ret) {
            return Promise.reject(
              "Failed to compile. Clang exited with: " + ret.toString()
            );
          }
        } catch (e) {
          return Promise.reject("Failed to compile. Error: " + e.toString());
        }
        const compileEndTime = new Date();
        console.log(
          "Compiled .o object file. Elapsed time: %d ms",
          compileEndTime - compileStartTime
        );

        return wasmFetcher
          .getModuleParams("lld.wasm", "onlylibs.data", printer)
          .then((params) => {
            return LldModule({ ...params, thisProgram: "wasm-ld" });
          })
          .then(function (loadedLldMod) {
            console.log("Loaded Lld Module!");

            loadedLldMod.FS.mkdir("/clangmod");
            loadedLldMod.FS.mount(
              loadedLldMod.PROXYFS,
              {
                root: "/",
                fs: loadedClangModule.FS,
              },
              "/clangmod"
            );
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
              "-lsockets",
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
              "-z",
              "stack-size=5242880",
              "--initial-memory=16777216",
              "--max-memory=16777216",
              "--global-base=1024",
            ];
            const linkStartTime = new Date();
            try {
              let ret = loadedLldMod.callMain([
                "/clangmod/hello.o",
                "-o",
                "hello.wasm",
                "-L/lib/wasm32-emscripten",
                ...system_libraries,
                ...linkOptions,
                ...project_libraries,
              ]);
              if (ret) {
                return Promise.reject(
                  "Failed to link. Lld exited with: " + ret.toString()
                );
              }
            } catch (e) {
              return Promise.reject("Failed to link. Error: " + e.toString());
            }
            const linkEndTime = new Date();
            console.log(
              "Linked .wasm file. Elapsed time: %d ms",
              linkEndTime - linkStartTime
            );
            // compiled executable
            let wasm = loadedLldMod.FS.readFile("hello.wasm");
            console.log(wasm);

            return WebAssembly.compile(wasm);
          });
      });
  };

  const _cachingCompiler = (() => {
    let prev = { source: "", result: Promise.resolve(null) };

    return (sourceCode, printer) => {
      if (prev.source === sourceCode) {
        return prev.result;
      }

      prev = {
        source: sourceCode,
        result: _compileSourceToWasm(sourceCode, printer),
      };
      return prev.result;
    };
  })();

  const compileAndRun = (sourceCode, inputMlir, mlirOptArgs, printer) => {
    return _cachingCompiler(sourceCode, printer).then(
      (inst) => {
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
          printErr: printer,
        }).then((compiledMod) => {
          compiledMod.FS.writeFile("input.mlir", inputMlir);
          console.log("Running mlir-opt...");
          const runStartTime = new Date();
          try {
            let ret = compiledMod.callMain([
              ...mlirOptArgs,
              "input.mlir",
              "-o",
              "output.mlir",
            ]);
            if (ret) {
              return Promise.reject(
                "Failed to run. mlir-opt exited with: " + ret.toString()
              );
            }
          } catch (e) {
            return Promise.reject("Failed to run. Error: " + e.toString());
          }
          const runEndTime = new Date();
          console.log(
            "Run successful. Elapsed time: %d ms",
            runEndTime - runStartTime
          );
          return compiledMod.FS.readFile("output.mlir", { encoding: "utf8" });
        });
      },
      (fail_msg) => {
        console.log("compileAndRun failed during compile phase.");
        return Promise.reject(fail_msg);
      }
    );
  };

  const initialize = () => {
    // prefetch data files
    wasmFetcher.fetchData("onlyincludes.data");
    wasmFetcher.fetchData("onlylibs.data");
  };

  return {
    compileAndRun: compileAndRun,
    initialize: initialize,
  };
};

export default WasmCompiler;
