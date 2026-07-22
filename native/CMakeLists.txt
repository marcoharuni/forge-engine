cmake_minimum_required(VERSION 3.25)

project(ForgeEngineNative VERSION 0.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerOptions)
include(CudaArchitectures)
include(Dependencies)
include(Sanitizers)

option(FORGE_ENABLE_CUDA "Enable optional CUDA language support" OFF)
option(FORGE_ENABLE_CUTLASS "Enable external CUTLASS experiments" OFF)
option(FORGE_ENABLE_CUTE_CPP "Enable external CuTe C++ experiments" OFF)
option(FORGE_BUILD_TESTS "Build native placeholder test targets" OFF)
option(FORGE_BUILD_BENCHMARKS "Build native placeholder benchmark targets" OFF)
option(FORGE_ENABLE_SANITIZERS "Enable supported host sanitizers" OFF)

set(FORGE_CUTLASS_ROOT "" CACHE PATH "Path to an external CUTLASS checkout")

add_library(forge_engine_native STATIC src/bindings.cpp)
add_library(ForgeEngine::Native ALIAS forge_engine_native)
target_include_directories(
    forge_engine_native
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)
target_compile_features(forge_engine_native PUBLIC cxx_std_17)
forge_set_compiler_options(forge_engine_native)
forge_enable_sanitizers(forge_engine_native)

if(FORGE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "FORGE_ENABLE_CUDA=ON, but no CUDA compiler was detected")
    endif()

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    forge_configure_cuda_architectures()

    # TODO: Add CUDA translation units only after kernels have implementations.
endif()

if(FORGE_ENABLE_CUTLASS)
    forge_require_cutlass()
    # TODO: Add CUTLASS experiment targets after implementations exist.
endif()

if(FORGE_ENABLE_CUTE_CPP)
    forge_require_cutlass()
    # TODO: Add architecture-specific CuTe C++ experiment targets.
endif()

if(FORGE_BUILD_TESTS)
    enable_testing()
    add_executable(forge_native_placeholder_test tests/test_placeholder.cpp)
    target_compile_features(forge_native_placeholder_test PRIVATE cxx_std_17)
    add_test(NAME forge_native_placeholder_test COMMAND forge_native_placeholder_test)
    set_tests_properties(forge_native_placeholder_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(FORGE_BUILD_BENCHMARKS)
    add_executable(forge_native_placeholder_benchmark benchmarks/benchmark_placeholder.cpp)
    target_compile_features(forge_native_placeholder_benchmark PRIVATE cxx_std_17)
endif()
