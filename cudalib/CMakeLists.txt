find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;-use_fast_math;-lineinfo;-DRANDOMHASH_CUDA)
set(CUDA_VERBOSE_BUILD ON)

#if (NOT MSVC)
#	list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
#endif()

list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
#list(APPEND CUDA_NVCC_FLAGS_DEBUG -G)
list(APPEND CUDA_NVCC_FLAGS_DEBUG -O3)


if(RH_CUDA_ARCH STREQUAL Kepler)
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37")
elseif(RH_CUDA_ARCH STREQUAL Maxwell)
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53")
elseif(RH_CUDA_ARCH STREQUAL Pascal)
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62")
elseif(RH_CUDA_ARCH STREQUAL Volta)
	list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70")
else()
	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}
		"-gencode arch=compute_30,code=sm_30"
		"-gencode arch=compute_35,code=sm_35"
		"-gencode arch=compute_50,code=sm_50"
		"-gencode arch=compute_52,code=sm_52"
		"-gencode arch=compute_53,code=sm_53"
		"-gencode arch=compute_60,code=sm_60"
		"-gencode arch=compute_61,code=sm_61"
		"-gencode arch=compute_62,code=sm_62"
		"-gencode arch=compute_70,code=sm_70"
	)
endif()


file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.h" "*.cuh")

cuda_add_library(cudalib STATIC ${sources} ${headers})
target_link_libraries(cudalib)
target_include_directories(cudalib BEFORE PRIVATE ./)
target_include_directories(cudalib BEFORE PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

