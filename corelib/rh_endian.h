/**
 * RandomHash source code implementation
 *
 * Copyright 2018 Polyminer1 <https://github.com/polyminer1>
 *
 * To the extent possible under law, the author(s) have dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along with
 * this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */
///
/// @file
/// @copyright Polyminer1

#pragma once

#include <stdint.h>
#include "corelib/rh_compiler.h"


#if defined(__MINGW32__) || defined(_WIN32) || defined(__CUDA_ARCH__) || defined(__GPU__)
  #ifndef LITTLE_ENDIAN
  # define LITTLE_ENDIAN 1234
  # define BYTE_ORDER   LITTLE_ENDIAN
  #endif
  # define NATIVE_LITTLE_ENDIAN
#elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__)
  # include <sys/endian.h>
#elif defined(__OpenBSD__) || defined(__SVR4)
  # include <sys/types.h>
#elif defined(MACOS_X) || (defined(__APPLE__) & defined(__MACH__))
  # include <machine/endian.h>
#elif defined( BSD ) && (BSD >= 199103)
  # include <machine/endian.h>
#elif defined( __QNXNTO__ ) && defined( __LITTLEENDIAN__ )
  # define LITTLE_ENDIAN 1234
  # define BYTE_ORDER    LITTLE_ENDIAN
#elif defined( __QNXNTO__ ) && defined( __BIGENDIAN__ )
  # define BIG_ENDIAN 1234
  # define BYTE_ORDER    BIG_ENDIAN
#else
# include <endian.h>
#endif

#if defined(_WIN32) && !defined(RHMINER_PLATFORM_GPU)
#include <stdlib.h>
#define RH_swap_u32(input_) _byteswap_ulong(input_)
#define RH_swap_u64(input_) _byteswap_uint64(input_)
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define RH_swap_u32(input_) OSSwapInt32(input_)
#define RH_swap_u64(input_) OSSwapInt64(input_)
#elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__)
#define RH_swap_u32(input_) bswap32(input_)
#define RH_swap_u64(input_) bswap64(input_)
#elif defined(__CUDA_ARCH__)

	extern __device__ unsigned int __byte_perm(unsigned int, unsigned	 int, unsigned int);

    // Input:       77665544 33221100
    // Output:      00112233 44556677
    __forceinline__ __device__ uint64_t RH_swap_u64(const uint64_t x)
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
    
    __forceinline__ __device__ uint32_t RH_swap_u32(const uint32_t x)
    {
	    /* device */
	    return __byte_perm(x, x, 0x0123);
    }
#else
    inline uint32_t RH_swap_u32(uint32_t val) 
    {
		val = ((val << 8) & 0xFF00FF00u) | ((val >> 8) & 0xFF00FFu); 
		return (val << 16) | ((val >> 16) & 0xFFFFu);
	}
    
    inline uint64_t RH_swap_u64(uint64_t a)
    {
    return  ((a & 0x00000000000000FFULL) << 56) | 
            ((a & 0x000000000000FF00ULL) << 40) | 
            ((a & 0x0000000000FF0000ULL) << 24) | 
            ((a & 0x00000000FF000000ULL) <<  8) | 
            ((a & 0x000000FF00000000ULL) >>  8) | 
            ((a & 0x0000FF0000000000ULL) >> 24) | 
            ((a & 0x00FF000000000000ULL) >> 40) | 
            ((a & 0xFF00000000000000ULL) >> 56);
    }
#endif


#if LITTLE_ENDIAN == BYTE_ORDER

#define fix_endian32(dst_ ,src_) dst_ = src_
#define fix_endian32_same(val_)
#define fix_endian64(dst_, src_) dst_ = src_
#define fix_endian64_same(val_)
#define fix_endian_arr32(arr_, size_)
#define fix_endian_arr64(arr_, size_)
#define RHMINER_TO_LITTLE_ENDIAN32(val) val
#define RHMINER_TO_BIG_ENDIAN32(val) RH_swap_u32(val)
#define RHMINER_TO_LITTLE_ENDIAN64(val) val
#define RHMINER_TO_BIG_ENDIAN64(val) RH_swap_u64(val)

#elif BIG_ENDIAN == BYTE_ORDER

#define RHMINER_TO_LITTLE_ENDIAN32(val) RH_swap_u32(val)
#define RHMINER_TO_BIG_ENDIAN32(val) val
#define RHMINER_TO_LITTLE_ENDIAN64(val) RH_swap_u64(val)
#define RHMINER_TO_BIG_ENDIAN64(val) val


#define fix_endian32(dst_, src_) dst_ = RH_swap_u32(src_)
#define fix_endian32_same(val_) val_ = RH_swap_u32(val_)
#define fix_endian64(dst_, src_) dst_ = RH_swap_u64(src_
#define fix_endian64_same(val_) val_ = RH_swap_u64(val_)
#define fix_endian_arr32(arr_, size_)			\
	do {										\
	for (unsigned i_ = 0; i_ < (size_), ++i_) { \
		arr_[i_] = RH_swap_u32(arr_[i_]);	\
	}											\
	while (0)
#define fix_endian_arr64(arr_, size_)			\
	do {										\
	for (unsigned i_ = 0; i_ < (size_), ++i_) { \
		arr_[i_] = RH_swap_u64(arr_[i_]);	\
	}											\
	while (0)									\

#else
# error "endian not supported"
#endif // BYTE_ORDER

#define RHMINER_SWAP_VAL(A,B, T) {T t = A; A = B; B = t;}

static inline void flip32(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 8; i++)
        dest[i] = RH_swap_u32(src[i]);
}

static inline void flip64(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 16; i++)
        dest[i] = RH_swap_u32(src[i]);
}

static inline void flip80(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 20; i++)
        dest[i] = RH_swap_u32(src[i]);
}

static inline void flip196(void *dest_p, const void *src_p)
{
	uint32_t *dest = (uint32_t *)dest_p;
	const uint32_t *src = (uint32_t *)src_p;
	int i;

	for (i = 0; i < 49; i++)
		dest[i] = RH_swap_u32(src[i]);
}


static inline void flip112(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 28; i++)
        dest[i] = RH_swap_u32(src[i]);
}

static inline void flip128(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 32; i++)
        dest[i] = RH_swap_u32(src[i]);
}

static inline void flip168(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 42; i++)
        dest[i] = RH_swap_u32(src[i]);
}

static inline void flip180(unsigned char* dest_p, const unsigned char* src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;
    int i;

    for (i = 0; i < 45; i++)
        dest[i] = RH_swap_u32(src[i]);
}


static inline void be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len)
{
    uint32_t i;

    for (i = 0; i < len; i++)
        dst[i] = RHMINER_TO_BIG_ENDIAN32(src[i]);
}


static inline void swap256(void *dest_p, const void *src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;

    dest[0] = src[7];
    dest[1] = src[6];
    dest[2] = src[5];
    dest[3] = src[4];
    dest[4] = src[3];
    dest[5] = src[2];
    dest[6] = src[1];
    dest[7] = src[0];
}

static inline void swab256(void *dest_p, const void *src_p)
{
    uint32_t *dest = (uint32_t *)dest_p;
    const uint32_t *src = (uint32_t *)src_p;

    dest[0] = RH_swap_u32(src[7]);
    dest[1] = RH_swap_u32(src[6]);
    dest[2] = RH_swap_u32(src[5]);
    dest[3] = RH_swap_u32(src[4]);
    dest[4] = RH_swap_u32(src[3]);
    dest[5] = RH_swap_u32(src[2]);
    dest[6] = RH_swap_u32(src[1]);
    dest[7] = RH_swap_u32(src[0]);
}

