
message(STATUS "FindCUDAToolkit for xgpu")

# cudart
if (FAISS_WITH_MACA)
    set(CUDAToolkit_INCLUDE_DIRS
        ${MACA_PATH}/tools/cu-bridge/include
        ${MACA_PATH}/include
        ${MACA_PATH}/include/mcc
        ${MACA_PATH}/include/mcr
        ${MACA_PATH}/include/mcblas
        ${MACA_PATH}/include/common
        )
    add_library(MACA::mcruntime SHARED IMPORTED)
    find_library(MACA_MCRUNTIME NAMES mcruntime PATHS ${MACA_PATH}/lib REQUIRED)
    set_property(TARGET MACA::mcruntime PROPERTY IMPORTED_LOCATION ${MACA_MCRUNTIME})
    target_include_directories(MACA::mcruntime INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

    add_library(MACA::runtime_cu SHARED IMPORTED)
    find_library(MACA_RUNTIME_CU NAMES runtime_cu PATHS ${MACA_PATH}/lib REQUIRED)
    set_property(TARGET MACA::runtime_cu PROPERTY IMPORTED_LOCATION ${MACA_RUNTIME_CU})
    target_include_directories(MACA::runtime_cu INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

    add_library(CUDA::cudart INTERFACE IMPORTED)
    set_property(TARGET CUDA::cudart PROPERTY INTERFACE_LINK_LIBRARIES MACA::mcruntime MACA::runtime_cu)
    message(STATUS "Found wrapper for CUDA::cudart ${MACA_MCRUNTIME} ${MACA_RUNTIME_CU}")

    add_library(CUDA::cublas SHARED IMPORTED)
    find_library(MACA_MCBLAS NAMES mcblas PATHS ${MACA_PATH}/lib REQUIRED)
    set_property(TARGET CUDA::cublas PROPERTY IMPORTED_LOCATION ${MACA_MCBLAS})
    target_include_directories(CUDA::cublas INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "Found wrapper for CUDA::cublas ${MACA_MCBLAS}")
endif (FAISS_WITH_MACA)
