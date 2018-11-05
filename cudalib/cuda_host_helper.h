#pragma once
#include "cuda_common.h"

#define CUDA_SET_DEVICE() CUDA_SAFE_CALL(cudaSetDevice(m_deviceID));

#define CUDA_SAFE_CALL(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		const char * errorString = cudaGetErrorString(err);	\
		PrintOutCritical("CUDA call error in %s : %s.\n", #call, errorString);			\
        RHMINER_EXIT_APP(""); \
	}														\
} while (0)

