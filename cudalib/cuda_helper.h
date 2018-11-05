// From ccminer
#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_common.h"

#include <stdint.h>

/*********************************************************************/
// Macros to catch CUDA errors in CUDA runtime calls
extern void PrintOutCritical(const char *szFormat, ...);
#if defined(_DEBUG)
#define CUDA_DEVICE_EXIT_MESSAGE(call, err) 		PrintOutCritical("%s(%d): Cuda error '%s' calling %s.\n", __FILE__, __LINE__, cudaGetErrorString(err),#call );  
#else
#define CUDA_DEVICE_EXIT_MESSAGE(call, err) 		PrintOutCritical("Cuda error '%s' calling %s.\n", cudaGetErrorString(err),#call );  
#endif

typedef union 
{
    uint4       vec;
    uint32_t    u[4];
    uint8_t     b[16];
} Uint128;

#ifdef _DEBUG
#define CUDA_SAFE_CALL(call)                                          \
    do {                                                                  \
	    cudaError_t err = call;                                           \
	    if (cudaSuccess != err)                                           \
        {                                                                 \
            CUDA_DEVICE_EXIT_MESSAGE(call, err);                          \
	    }                                                                 \
    } while (0)
#else
    #define CUDA_SAFE_CALL(call)                                          \
    do {                                                                  \
	    cudaError_t err = call;                                           \
	    if (cudaSuccess != err)                                           \
        {                                                                 \
            CUDA_DEVICE_EXIT_MESSAGE(call, err);                          \
		    exit(0xC0DAC0DA);                                             \
	    }                                                                 \
    } while (0)

#endif


#define CUDA_CALL_OR_RET(call) do {                                   \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, "");         \
		return;                                                       \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET_X(call, ret) do {                            \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, "");         \
		return ret;                                                   \
	}                                                                 \
} while (0)

/*********************************************************************/



#if __CUDA_ARCH__ < 320
#ifdef __CUDA_ARCH__
    // Kepler (Compute 3.0)
    #define ROTL32(x, n) ((x) << (n)) | ((x) >> (32 - (n)))
#endif
#else
// Kepler (Compute 3.5, 5.0)
__device__ __forceinline__ uint32_t ROTL32(const uint32_t x, const uint32_t n)
{
	return(__funnelshift_l((x), (x), (n)));
}
#endif
#if __CUDA_ARCH__ < 320
#ifdef __CUDA_ARCH__
    // Kepler (Compute 3.0)
    #define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#endif
#else
__device__ __forceinline__ uint32_t ROTR32(const uint32_t x, const uint32_t n)
{
	return(__funnelshift_r((x), (x), (n)));
}
#endif


#ifdef __CUDA_ARCH__

__device__ __forceinline__  uint8_t cuROTL8(uint8_t  x, uint8_t n) 
{
    return (x << n) | (x >> (8 - n));
}

__device__ __forceinline__  uint8_t cuROTR8(uint8_t  x, uint8_t n) 
{
    return cuROTL8(x, (uint8_t)(8-n));
}

#endif


#ifndef USE_ROT_ASM_OPT
#define USE_ROT_ASM_OPT 1
#endif

// 64-bit ROTATE RIGHT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
/* complicated sm >= 3.5 one (with Funnel Shifter beschleunigt), to bench */
__device__ __forceinline__
uint64_t ROTR64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTR64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shr.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shl.b64 %0, %1, roff;\n\t"
		"add.u64 %0, %0, lhs;\n\t"
	"}\n"
	: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#elif defined(__CUDA_ARCH__)
    #define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
__device__ __forceinline__
uint64_t ROTL64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shl.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shr.b64 %0, %1, roff;\n\t"
		"add.u64 %0, lhs, %0;\n\t"
	"}\n"
	: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#elif __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 3
__device__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"setp.lt.u32 p, %2, 32;\n\t"
		"@!p mov.b64 %0, {vl,vh};\n\t"
		"@p  mov.b64 %0, {vh,vl};\n\t"
	"}"
		: "=l"(res) : "l"(x) , "r"(offset)
	);
	return res;
#elif defined(__CUDA_ARCH__)
    #define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif


// Endian Drehung für 32 Bit Typen
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint32_t cuda_swab32(const uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}
#else
	#define cuda_swab32(x) \
	((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | \
		(((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

    
// Input:       77665544 33221100
// Output:      00112233 44556677
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint64_t cuda_swab64(const uint64_t x)
{
	uint64_t result;
	uint2 t;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(t.x), "=r"(t.y) : "l"(x));
	t.x=__byte_perm(t.x, 0, 0x0123);
	t.y=__byte_perm(t.y, 0, 0x0123);
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(t.y), "r"(t.x));
	return result;
}
#else
	/* host */
	#define cuda_swab64(x) \
		((uint64_t)((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
			(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
			(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
			(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
			(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
			(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
			(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
			(((uint64_t)(x) & 0x00000000000000ffULL) << 56)))
#endif

#ifdef __CUDA_ARCH__
__device__ __forceinline__ void devectorize4(uint4 inn, uint64_t &x, uint64_t &y)
{
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(x) : "r"(inn.x), "r"(inn.y));
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(y) : "r"(inn.z), "r"(inn.w));
}
#else
__forceinline__ void devectorize4(uint4 inn, uint64_t &x, uint64_t &y)
{
    x = ((U64)inn.x) << 32 | inn.y;
    y = ((U64)inn.z) << 32 || inn.w;
}
#endif

#endif // #ifndef CUDA_HELPER_H


