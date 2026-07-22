function(forge_configure_cuda_architectures)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
    endif()
    message(STATUS "ForgeEngine CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endfunction()

