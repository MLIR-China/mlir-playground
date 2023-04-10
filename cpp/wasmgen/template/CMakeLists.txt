cmake_minimum_required(VERSION 3.22)
project(MlirOptTemplate)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(MLIRConfig)
include(LLVMConfig)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

get_property(ALL_MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)

add_executable(MlirOptTemplate template.cpp)
target_link_libraries(MlirOptTemplate PUBLIC
    ${ALL_MLIR_LIBS}
    ${LLVM_AVAILABLE_LIBS}
)

set_target_properties(MlirOptTemplate PROPERTIES LINK_FLAGS "-s ENVIRONMENT='web' -s EXIT_RUNTIME=1 -s EXPORT_ES6=1 -s MODULARIZE=1 -s LLD_REPORT_UNDEFINED=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=_main,_free,_malloc -s EXPORTED_RUNTIME_METHODS=ccall,cwrap,FS,PROXYFS,allocateUTF8,FS_createPath,FS_createDataFile,FS_createPreloadedFile,addRunDependency,removeRunDependency,callMain -s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=\$Browser -s LZ4=1 -s STACK_SIZE=5MB -s DEFAULT_PTHREAD_STACK_SIZE=2MB -s MAXIMUM_MEMORY=4GB -s WASM_BIGINT=1 -lproxyfs.js")
