function(forge_require_cutlass)
    if(NOT FORGE_ENABLE_CUDA)
        message(FATAL_ERROR "CUTLASS and CuTe C++ paths require FORGE_ENABLE_CUDA=ON")
    endif()
    if(NOT FORGE_CUTLASS_ROOT)
        message(FATAL_ERROR "Set FORGE_CUTLASS_ROOT to an external CUTLASS checkout")
    endif()
    if(NOT EXISTS "${FORGE_CUTLASS_ROOT}/include/cutlass/cutlass.h")
        message(FATAL_ERROR "FORGE_CUTLASS_ROOT does not contain CUTLASS headers")
    endif()
    target_include_directories(forge_engine_native SYSTEM PRIVATE
        "${FORGE_CUTLASS_ROOT}/include"
        "${FORGE_CUTLASS_ROOT}/tools/util/include"
    )
endfunction()

