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
#include "MinersLib/Pascal/RandomHash_def.h"

#ifdef RHMINER_PLATFORM_CPU
extern int  g_memoryBoostLevel;
extern int  g_sseOptimization;

#endif //CPU

#ifndef RANDOMHASH_CUDA

inline void RH_INPLACE_MEMCPY_128(U8* pDst, U8* pSrc, size_t byteCount)
{
    RH_ASSERT(((size_t)pDst % 32) == 0);
    RH_ASSERT(((size_t)pSrc % 32) == 0);
    if (g_memoryBoostLevel == 1)
    {
        S32 n = RHMINER_CEIL(byteCount, sizeof(__m128i));
        __m128i r0;        
        __m128i* src = (__m128i *)(pSrc);
        __m128i* end = src + n;
        __m128i* dst = (__m128i *)(pDst);
        while (src < end)
        {
            r0 = RH_MM_LOAD128(src++);
            RH_MM_STORE128(dst++, r0);
        }
       //RH_MM_BARRIER();
    }
    else
        memcpy(pDst, pSrc, byteCount);
}
/*
CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_INPLACE_MEMCPY_128_A)(U8* pDst, U8* pSrc, U32 byteCount, MurmurHash3_x86_32_State* accum)
{
    RH_ASSERT(( (size_t)pDst % 8) == 0);
    RH_ASSERT(( (size_t)pSrc % 8) == 0);

    S32 n = (byteCount / sizeof(__m128i)) * sizeof(__m128i);
    U32 m = byteCount % sizeof(__m128i);
    __m128i r0;
    while (n > 0)
    {
        r0 = RH_MM_LOAD128((__m128i *)(pSrc ));
        MurmurHash3_x86_32_Update_16(r0, 16, accum);   
        RH_MM_STORE128((__m128i *)(pDst ), r0);
        pSrc += sizeof(__m128i);
        pDst += sizeof(__m128i);
        n -= sizeof(__m128i);
    }
    if (m)
    {
        r0 = RH_MM_LOAD128((__m128i *)(pSrc));
        RH_MM_STORE128((__m128i *)(pDst ), r0);
        MurmurHash3_x86_32_Update_16(r0, m, accum); 
    }
    RH_MM_BARRIER();
}
*/

#else //!CPU

//TODO: Optmiz - Test memcpy
#define RH_INPLACE_MEMCPY_128(pDst, pSrc, byteCount)                    \
    {U8* end = pDst + byteCount;                                        \
    while(pDst < end)                                                   \
    {                                                                   \
        *(uint4 *)pDst = *(uint4 *)pSrc;                                \
        pDst += 16;                                                     \
        pSrc += 16;                                                     \
    }}

/*
CUDA_DECL_DEVICE
void CUDA_SYM(RH_INPLACE_MEMCPY_128_A)(U8* pDst, U8* pSrc, U32 byteCount, MurmurHash3_x86_32_State* accum)
{
    S32 n = RHMINER_FLOOR(byteCount, sizeof(uint4));
    uint4 data;
    while (n >= sizeof(uint4))
    {
        data = *(uint4 *)pSrc;
        _CM(MurmurHash3_x86_32_Update_16)(data, 16, accum);
        *(uint4 *)pDst = data;
        pDst += sizeof(uint4);
        pSrc += sizeof(uint4);
        n -= sizeof(uint4);
    }
    data = *(uint4 *)pSrc;
    _CM(MurmurHash3_x86_32_Update_16)(data, byteCount % 16, accum);
    *(uint4 *)pDst = data;
}
*/

#endif //CPU

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDE_MEMCPY_ALIGNED_SIZE128)(U8 *pDst, U8 *pSrc, size_t byteCount)
{
    RH_ASSERT(( (size_t)pDst % 8) == 0);
    RH_ASSERT(( (size_t)pSrc % 8) == 0);
    RH_ASSERT(( (size_t)pDst % 32) == 0);
    RH_ASSERT(( (size_t)pSrc % 32) == 0);

    RH_INPLACE_MEMCPY_128(pDst, pSrc, byteCount);
}

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(U8 *pDst, U8 *pSrc, size_t byteCount)
{
    RH_ASSERT(byteCount < RH_StrideSize - 256); 

#ifndef __CUDA_ARCH__
    memcpy(pDst, pSrc, byteCount);
    return;
#endif

    //CUDA and PC
    if (((size_t)pDst)%8 == 0 &&
        ((size_t)pSrc)%8 == 0)
    {
        U8* end = pDst + byteCount;
        while(pDst < end)
        {
            *(U64*)pDst = *(U64*)pSrc;
            pDst += 8;
            pSrc += 8;
        }
    }
    else
    {
        memcpy((void*)(pDst), (void*)(pSrc), byteCount); 
    }
}


#ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3

CUDA_DECL_HOST_AND_DEVICE void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_1)(U8* strideArray, U32 elementIdx)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);
    
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);

    register U32 h1 = mm3_array1->h1;
    RH_ASSERT(size >= sizeof(U64));
    RH_ASSERT(( (size_t)strideArray % 8) == 0);
    S32 n = (size / sizeof(U64)) * sizeof(U64);
    U32 m = size % sizeof(U64);    
    RH_StridePtr lstride_end = lstride + n;
    U64 r0;
    mm3_array1->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(U64*)(lstride);
        lstride += sizeof(U64);
        MURMUR3_BODY((U32)(r0));
        MURMUR3_BODY((U32)(r0 >> 32));        
    }

    mm3_array1->h1 = h1;
    if (m)
    {
        U64 r0 = *((U64 *)(lstride));
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array1);
    }
}

CUDA_DECL_HOST_AND_DEVICE void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_1_boost)(U8* strideArray, U32 elementIdx)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);
    
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);

    register U32 h1 = mm3_array1->h1;
    RH_ASSERT(size >= sizeof(U64));
    RH_ASSERT(( (size_t)strideArray % 8) == 0);
    S32 n = (size / sizeof(U64)) * sizeof(U64);
    U32 m = size % sizeof(U64);    
    RH_StridePtr lstride_end = lstride + n;
    U64 r0;
    mm3_array1->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(U64*)(lstride);
        lstride += sizeof(U64);
        RH_PREFETCH_MEM((const char*)lstride);

        MURMUR3_BODY((U32)(r0));
        MURMUR3_BODY((U32)(r0 >> 32));        
    }

    mm3_array1->h1 = h1;
    if (m)
    {
        U64 r0 = *((U64 *)(lstride));
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array1);
    }
}

#define MURMUR3_BODY2(k) { \
            uint32_t k1 = (k); \
            k1 *= MurmurHash3_x86_32_c1; \
            k1 = ROTL32(k1, 15); \
            k1 *= MurmurHash3_x86_32_c2; \
            h2 ^= k1; \
            h2 = ROTL32(h2, 13); \
            h2 = h2 * 5 + MurmurHash3_x86_32_c3; } 


CUDA_DECL_HOST_AND_DEVICE void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2)(U8* strideArray, U32 elementIdx, U8* array2)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);
    
    MurmurHash3_x86_32_State* mm3_array2 = RH_StrideArrayStruct_GetAccum(array2);
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);
    RH_ASSERT(mm3_array2->idx == 0);

    register U32 h1 = mm3_array1->h1;
    register U32 h2 = mm3_array2->h1;
    RH_ASSERT(size >= sizeof(U64));
    RH_ASSERT(( (size_t)strideArray % 8) == 0);
    S32 n = (size / sizeof(U64)) * sizeof(U64);
    U32 m = size % sizeof(U64);    
    RH_StridePtr lstride_end = lstride + n;
    U64 r0;
    mm3_array1->totalLen += n;
    mm3_array2->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(U64*)(lstride);
        lstride += sizeof(U64);

        MURMUR3_BODY((U32)(r0));
        MURMUR3_BODY2((U32)(r0));
        MURMUR3_BODY((U32)(r0 >> 32));        
        MURMUR3_BODY2((U32)(r0 >> 32));        
    }
    mm3_array1->h1 = h1;
    mm3_array2->h1 = h2;
    if (m)
    {
        U64 r0 = *((U64 *)(lstride));
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array1);
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array2);
    }
}

/*
CUDA_DECL_DEVICE void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2)(U8* strideArray, U32 elementIdx, U8* array2)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);
    
    MurmurHash3_x86_32_State* mm3_array2 = RH_StrideArrayStruct_GetAccum(array2);
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);
    RH_ASSERT(mm3_array2->idx == 0);

    register U32 h1 = mm3_array1->h1;
    register U32 h2 = mm3_array2->h1;
    RH_ASSERT(size >= sizeof(U64));
    RH_ASSERT(( (size_t)strideArray % 8) == 0);
    S32 n = (size / sizeof(U64)) * sizeof(U64);
    U32 m = size % sizeof(U64);    
    RH_StridePtr lstride_end = lstride + n;
    U64 r0;
    mm3_array1->totalLen += n;
    mm3_array2->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(U64*)(lstride);
        lstride += sizeof(U64);

        MURMUR3_BODY((U32)(r0));
        MURMUR3_BODY2((U32)(r0));
        MURMUR3_BODY((U32)(r0 >> 32));        
        MURMUR3_BODY2((U32)(r0 >> 32));        
    }
    mm3_array1->h1 = h1;
    mm3_array2->h1 = h2;
    if (m)
    {
        U64 r0 = *((U64 *)(lstride));
        MurmurHash3_x86_32_Update_8(r0, m, mm3_array1);
        MurmurHash3_x86_32_Update_8(r0, m, mm3_array2);
    }
}
*/
CUDA_DECL_HOST_AND_DEVICE void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2_boost)(U8* strideArray, U32 elementIdx, U8* array2)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);
    
    MurmurHash3_x86_32_State* mm3_array2 = RH_StrideArrayStruct_GetAccum(array2);
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);
    RH_ASSERT(mm3_array2->idx == 0);

    register U32 h1 = mm3_array1->h1;
    register U32 h2 = mm3_array2->h1;
    RH_ASSERT(size >= sizeof(U64));
    RH_ASSERT(( (size_t)strideArray % 8) == 0);
    S32 n = (size / sizeof(U64)) * sizeof(U64);
    U32 m = size % sizeof(U64);    
    RH_StridePtr lstride_end = lstride + n;
    U64 r0;
    mm3_array1->totalLen += n;
    mm3_array2->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(U64*)(lstride);
        lstride += sizeof(U64);
        RH_PREFETCH_MEM(lstride);

        MURMUR3_BODY((U32)(r0));
        MURMUR3_BODY2((U32)(r0));
        MURMUR3_BODY((U32)(r0 >> 32));        
        MURMUR3_BODY2((U32)(r0 >> 32));        
    }
    mm3_array1->h1 = h1;
    mm3_array2->h1 = h2;
    if (m)
    {
        U64 r0 = *((U64 *)(lstride));
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array1);
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array2);
    }
}


#define RH_MURMUR3_BODY_2(k1, hx) \
    hx ^= k1; \
    hx = ROTL32(hx, 13); \
    hx = hx * 5 + MurmurHash3_x86_32_c3;

#if defined(RHMINER_ENABLE_SSE4) /*&& defined(RH_COMPILE_CPU_ONLY)*/ && !defined(__CUDA_ARCH__)

#if defined(RANDOMHASH_CUDA) || defined(RHMINER_NO_SSE4)
//unused yet
static inline __m128i _mm_mullo_epi32_EMU(const __m128i &a, const __m128i &b)
{
    __m128i tmp1 = _mm_mul_epu32(a,b); /* mul 2,0*/
    __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); /* mul 3,1 */
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
}
#define _mm_mullo_epi32_M _mm_mullo_epi32_EMU
#else
#define _mm_mullo_epi32_M _mm_mullo_epi32
#endif

void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41)(U8* strideArray, U32 elementIdx)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);

    RH_MUR3_BACKUP_STATE(RH_StrideArrayStruct_GetAccum(strideArray));
    back_i = 0;   
    RH_ASSERT(back_idx != 0xDEADBEEF)
    RH_ASSERT(back_idx == 0);

    register uint32_t h1 = back_h1;
    RH_ASSERT(size >= sizeof(__m128i));
    RH_ASSERT(( (size_t)strideArray % 32) == 0);
    S32 n = (size / sizeof(__m128i)) * sizeof(__m128i);
    U32 m = size % sizeof(__m128i);   
    RH_StridePtr lstride_end = lstride + n;
    __m128i r0,r1;
    __m128i c1 = _mm_cvtsi32_si128(MurmurHash3_x86_32_c1);  
    __m128i c2 = _mm_cvtsi32_si128(MurmurHash3_x86_32_c2);
    U32 r32;
    back_totalLen += n;
    c1 = _mm_shuffle_epi32(c1, 0);
    c2 = _mm_shuffle_epi32(c2, 0);
    while (lstride != lstride_end)
    {
        r0 = RH_MM_LOAD128((__m128i*)lstride);
        lstride += sizeof(__m128i);

        r0 = _mm_mullo_epi32_M(r0, c1);
        r1 = r0;
        r0 = _mm_slli_epi32(r0, 15);            
        r1 = _mm_srli_epi32(r1, 17);
        r0 = _mm_castps_si128( _mm_or_ps(_mm_castsi128_ps(r0), _mm_castsi128_ps(r1)) ) ;
        r0 = _mm_mullo_epi32_M(r0, c2);
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        r0 = _mm_shuffle_epi32(r0, 0x39); 
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        r0 = _mm_shuffle_epi32(r0, 0x39); 
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        r0 = _mm_shuffle_epi32(r0, 0x39); 
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
    }
    back_h1 = h1;
    if (m)
    {
        RH_ASSERT(m <= sizeof(U64));

        U64 r0 = *((U64 *)(lstride));
        INPLACE_M_MurmurHash3_x86_32_Update_8(r0, m); 
    }
    
    RH_MUR3_RESTORE_STATE(RH_StrideArrayStruct_GetAccum(strideArray));
}

void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41_2)(U8* strideArray, U32 elementIdx, U8* strideArray2)
{    
    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstride);
    lstride = RH_STRIDE_GET_DATA(lstride);

    MurmurHash3_x86_32_State* mm3_array2 = RH_StrideArrayStruct_GetAccum(strideArray2);
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);
    RH_ASSERT(mm3_array2->idx == 0);

    register U32 h1 = mm3_array1->h1;
    register U32 h2 = mm3_array2->h1;
    RH_ASSERT(size >= sizeof(__m128i));
    RH_ASSERT(( (size_t)strideArray % 32) == 0);
    S32 n = (size / sizeof(__m128i)) * sizeof(__m128i);
    U32 m = size % sizeof(__m128i);   
    RH_StridePtr lstride_end = lstride + n;
    __m128i r0,r1;
    __m128i c1 = _mm_cvtsi32_si128(MurmurHash3_x86_32_c1);  
    __m128i c2 = _mm_cvtsi32_si128(MurmurHash3_x86_32_c2);
    U32 r32;
    c1 = _mm_shuffle_epi32(c1, 0);
    c2 = _mm_shuffle_epi32(c2, 0);

    mm3_array1->totalLen += n;
    mm3_array2->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = RH_MM_LOAD128((__m128i*)lstride);
        lstride += sizeof(__m128i);

        r0 = _mm_mullo_epi32_M(r0, c1);           
        r1 = r0;
        r0 = _mm_slli_epi32(r0, 15);            
        r1 = _mm_srli_epi32(r1, 17);
        r0 = _mm_castps_si128( _mm_or_ps(_mm_castsi128_ps(r0), _mm_castsi128_ps(r1)) ) ;
        r0 = _mm_mullo_epi32_M(r0, c2);
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        RH_MURMUR3_BODY_2((U32)(r32), h2);
        r0 = _mm_shuffle_epi32(r0, 0x39); 
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        RH_MURMUR3_BODY_2((U32)(r32), h2);
        r0 = _mm_shuffle_epi32(r0, 0x39); 
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        RH_MURMUR3_BODY_2((U32)(r32), h2);
        r0 = _mm_shuffle_epi32(r0, 0x39); 
        r32 = _mm_cvtsi128_si32(r0);
        RH_MURMUR3_BODY_2((U32)(r32), h1);
        RH_MURMUR3_BODY_2((U32)(r32), h2);
       
    }
    mm3_array1->h1 = h1;
    mm3_array2->h1 = h2;
    if (m)
    {
        RH_ASSERT(m <= sizeof(U64));
        U64 r0 = *((U64 *)(lstride));

        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array1);
        _CM(MurmurHash3_x86_32_Update_8)(r0, m, mm3_array2);
    }
}

#endif  //#if defined(RHMINER_ENABLE_SSE4) && defined(RH_COMPILE_CPU_ONLY)&& !defined(__CUDA_ARCH__)




#ifdef RANDOMHASH_CUDA

CUDA_DECL_DEVICE
void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_DUO)(U8* strideArray, U32 elementIdx, U8* r5p2AccumArray)
{
#ifdef __CUDA_ARCH__
    RH_STRIDEARRAY_PUSHBACK(r5p2AccumArray, RH_STRIDEARRAY_GET(strideArray, elementIdx));
            
    #if defined(RH_USE_CUDA_MEM_BOOST)
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2_boost)(strideArray, elementIdx, r5p2AccumArray);
    #else 
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2)(strideArray, elementIdx, r5p2AccumArray);
    #endif
#endif
}

CUDA_DECL_DEVICE
void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(U8* strideArray, U32 elementIdx)
{
#ifdef __CUDA_ARCH__
    #if defined(RH_USE_CUDA_MEM_BOOST)
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_1_boost)(strideArray, elementIdx);
    #else
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_1)(strideArray, elementIdx);
    #endif
#endif
}

#else  //RANDOMHASH_CUDA

void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(U8* strideArray, U32 elementIdx, U8* r5p2AccumArray = 0)
{
    U32 sseoOtimization = g_sseOptimization;// RH_STRIDEARRAY_GET_EXTRA(strideArray, sseoptimization);
    if (!sseoOtimization)
    {
        if (r5p2AccumArray)
        {   
            RH_STRIDEARRAY_PUSHBACK(r5p2AccumArray, RH_STRIDEARRAY_GET(strideArray, elementIdx));
            
            if (RH_STRIDEARRAY_GET_EXTRA(strideArray, memoryboost))
                _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2_boost)(strideArray, elementIdx, r5p2AccumArray);
            else
                _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_2)(strideArray, elementIdx, r5p2AccumArray);
        }
        else
        {
            if (RH_STRIDEARRAY_GET_EXTRA(strideArray, memoryboost))
                return _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_1_boost)(strideArray, elementIdx);
            else
                return _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_64b_1)(strideArray, elementIdx);
        }
    }
    else if (sseoOtimization == 1)
    {
#if defined(RHMINER_ENABLE_SSE4)
        if (r5p2AccumArray)
        {
            RH_STRIDEARRAY_PUSHBACK(r5p2AccumArray, RH_STRIDEARRAY_GET(strideArray, elementIdx));
            return _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41_2)(strideArray, elementIdx, r5p2AccumArray);
        }
        else
            return _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41)(strideArray, elementIdx);
#else
        RHMINER_ASSERT(sseoOtimization == 0); //impossible
#endif  
    }
#if 0
    else if (sseoOtimization == 2)
    {
        RHMINER_ASSERT(0);
        return _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_AVX2)(strideArray, elementIdx);
    }
#endif

}


inline void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_DUO)(U8* strideArray, U32 elementIdx, U8* r5p2AccumArray)
{
    RH_STRIDE_ARRAY_UPDATE_MURMUR3(strideArray, elementIdx, r5p2AccumArray);
}

#endif //CUDA_ACRH



CUDA_DECL_DEVICE
void CUDA_SYM(RH_STRIDEARRAY_PUSHBACK_MANY_ALL)(U8* strideArrayVar, U8* strideArrayVarSrc)
{
    RH_ASSERT(strideArrayVar != strideArrayVarSrc);
    *RH_StrideArrayStruct_GetAccum(strideArrayVar) = *RH_StrideArrayStruct_GetAccum(strideArrayVarSrc);

    U32 i = 0;
    U32 srcCnt = RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc);
    RH_STRIDEARRAY_SET_SIZE(strideArrayVar, srcCnt);
    while(i < srcCnt)
    {
        ((RH_StrideArrayStruct*)(strideArrayVar))->strides[i] = ((RH_StrideArrayStruct*)(strideArrayVarSrc))->strides[i];
        i++;
    }
}


CUDA_DECL_DEVICE
void CUDA_SYM(RH_STRIDEARRAY_PUSHBACK_MANY_UPDATE)(U8* strideArrayVar, U8* strideArrayVarSrc, U8* r5p2AccumArray)
{
    U32 i = RH_STRIDEARRAY_GET_SIZE(strideArrayVar);
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc) + i);
    U32 j = 0;
    U32 cnt = RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc)+i;
    RH_STRIDEARRAY_SET_SIZE(strideArrayVar, cnt);
    cnt--;
    
    while(i < cnt)
    {
        ((RH_StrideArrayStruct*)(strideArrayVar))->strides[i] = ((RH_StrideArrayStruct*)(strideArrayVarSrc))->strides[j++];
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(strideArrayVar, i);
        i++;
    }
    ((RH_StrideArrayStruct*)(strideArrayVar))->strides[i] = ((RH_StrideArrayStruct*)(strideArrayVarSrc))->strides[j++];
    _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_DUO)(strideArrayVar, i, r5p2AccumArray);
    
}

#else

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDEARRAY_PUSHBACK_MANY)(U8* strideArrayVar, U8* strideArrayVarSrc)
{                                                                               
    RH_ASSERT(strideArrayVar != strideArrayVarSrc);                             
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVar) <= RH_StrideArrayCount);  
    RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc)                            
    {                                                                           
        RH_STRIDEARRAY_PUSHBACK(strideArrayVar, strideItrator);                 
    }                                                                           
    RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc)                              
} 

error;

#endif


#define RH_Accum_8() \
    U32 acc8_idx = 0;  \
    U64 acc8_buf = 0;  \

#define RH_Accum_8_Reset(p) \
    acc8_idx = 0;  \
    acc8_buf = 0;  \

#define RH_Accum_8_Add(chunk8, acc8_ptr)                    \
{                                                           \
    acc8_buf &= ~(0xFFLLU << (U64)(acc8_idx << 3));         \
    acc8_buf |= ((U64)(chunk8) << (U64)(acc8_idx << 3));    \
    acc8_idx++;                                             \
    if (acc8_idx == 8)                                      \
    {                                                       \
        *((U64*)acc8_ptr) = acc8_buf;                       \
        acc8_ptr += 8;                                      \
        acc8_idx = 0;                                       \
        acc8_buf = 0;                                       \
    }                                                       \
}


#define RH_Accum_8_Finish(acc8_ptr)      \
{                                       \
    RH_ASSERT(acc8_idx != 8);           \
    *((U64*)acc8_ptr) = acc8_buf;       \
}

inline void CUDA_SYM_DECL(Transfo0_2)(U8* nextChunk, U32 size, U8* source)
{
    U32 rndState = _CM(MurmurHash3_x86_32_Fast)(source,size);
    if (!rndState)
        rndState = 1;

    {
        U8* head = nextChunk;
        U8* end = head + size;
        while(head < end)
        {
            uint32_t x = rndState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rndState = x;
            *head = source[x % size];
            head++;
        }
    }
}

#ifdef RHMINER_PLATFORM_CPU


#if defined(RHMINER_ENABLE_SSE4) && !defined(__CUDA_ARCH__)
inline void CUDA_SYM_DECL(Transfo0_2_128_SSE4)(U8* nextChunk, U32 size, U8* source)
{
    U32 rndState = _CM(MurmurHash3_x86_32_Fast)(source,size); 
    if (!rndState)
        rndState = 1;

    __m128i r0,r1,r2,r3,r4,r5,r6,r7;
    switch(size/16)
    {
        case 8:
        case 7: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                r6 = RH_MM_LOAD128 ((__m128i *)(source+6*sizeof(__m128i)));
                r7 = RH_MM_LOAD128 ((__m128i *)(source+7*sizeof(__m128i)));
                break;
        case 6: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                r6 = RH_MM_LOAD128 ((__m128i *)(source+6*sizeof(__m128i)));
                break;
        case 5: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                break;
        case 4: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                break;
        case 3: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                break;
        case 2: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                break;
        case 1: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                break;
        case 0: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                break;
        default: RHMINER_ASSERT(false);
    }

    U8* head = nextChunk;
    U8* end = head + size;
    //load work
    while(head < end)
    {
        uint32_t x = rndState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rndState = x;
        U32 d;
        #define RH_GB128_SSE4(chunk128, n)                                         \
            {                                                               \
                d = ((n) & 0x7)*8;                                          \
                switch((n)>>2)                                              \
                {                                                           \
                    case 0:b = _mm_extract_epi32_M(chunk128, 0)>>d; break;  \
                    case 1:b = _mm_extract_epi32_M(chunk128, 1)>>d; break;  \
                    case 2:b = _mm_extract_epi32_M(chunk128, 2)>>d; break;  \
                    case 3:b = _mm_extract_epi32_M(chunk128, 3)>>d; break;  \
                    default:                                                \
                        RHMINER_ASSERT(false);                              \
                };                                                          \
            }

        U8 b;
        U32 val = x % size;
        U32 reg = val / 16;
        U32 n = val % 16;
        switch(reg)
        {
            case 7: RH_GB128_SSE4(r7, n)  break;
            case 6: RH_GB128_SSE4(r6, n)  break;
            case 5: RH_GB128_SSE4(r5, n)  break;
            case 4: RH_GB128_SSE4(r4, n)  break;
            case 3: RH_GB128_SSE4(r3, n)  break;
            case 2: RH_GB128_SSE4(r2, n)  break;
            case 1: RH_GB128_SSE4(r1, n)  break;
            case 0: RH_GB128_SSE4(r0, n)  break;
            default: RHMINER_ASSERT(false);
        }
        
        *head = b;
        head++;
    }
}
#endif  //defined(RHMINER_ENABLE_SSE4)


inline void CUDA_SYM_DECL(Transfo0_2_128_SSE3)(U8* nextChunk, U32 size, U8* source)
{
    U32 rndState = _CM(MurmurHash3_x86_32_Fast)(source,size); 
    if (!rndState)
        rndState = 1;

    __m128i r0,r1,r2,r3,r4,r5,r6,r7;
    switch(size/16)
    {
        case 8:
        case 7: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                r6 = RH_MM_LOAD128 ((__m128i *)(source+6*sizeof(__m128i)));
                r7 = RH_MM_LOAD128 ((__m128i *)(source+7*sizeof(__m128i)));
                break;
        case 6: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                r6 = RH_MM_LOAD128 ((__m128i *)(source+6*sizeof(__m128i)));
                break;
        case 5: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                break;
        case 4: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                break;
        case 3: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                break;
        case 2: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                break;
        case 1: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                break;
        case 0: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                break;
        default: RHMINER_ASSERT(false);
    }

    U8* head = nextChunk;
    U8* end = head + size;
    //load work
    while(head < end)
    {
        uint32_t x = rndState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rndState = x;
        U32 d;
        #define RH_GB128(chunk128, n)                                         \
            {                                                               \
                d = ((n) & 0x7)*8;                                          \
                switch((n)>>2)                                              \
                {                                                           \
                    case 0:b = _mm_extract_epi32_<0>(chunk128)>>d; break;  \
                    case 1:b = _mm_extract_epi32_<1>(chunk128)>>d; break;  \
                    case 2:b = _mm_extract_epi32_<2>(chunk128)>>d; break;  \
                    case 3:b = _mm_extract_epi32_<3>(chunk128)>>d; break;  \
                    default:                                                \
                        RHMINER_ASSERT(false);                              \
                };                                                          \
            }

        U8 b;
        U32 val = x % size;
        U32 reg = val / 16;
        U32 n = val % 16;
        switch(reg)
        {
            case 7: RH_GB128(r7, n)  break;
            case 6: RH_GB128(r6, n)  break;
            case 5: RH_GB128(r5, n)  break;
            case 4: RH_GB128(r4, n)  break;
            case 3: RH_GB128(r3, n)  break;
            case 2: RH_GB128(r2, n)  break;
            case 1: RH_GB128(r1, n)  break;
            case 0: RH_GB128(r0, n)  break;
            default: RHMINER_ASSERT(false);
        }

        *head = b;
        head++;
    }
}

#endif //cpu


void CUDA_SYM_DECL(Transfo1_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 halfSize = size >> 1;
    RH_ASSERT((size % 2) == 0);


    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, outputPtr + halfSize , halfSize);
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk + halfSize, outputPtr, halfSize);
}

void CUDA_SYM_DECL(Transfo2_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 halfSize = size >> 1;
    
    U8* srcHead = outputPtr;
    U8* srcEnd = srcHead + halfSize;
    U8* srcTail = &outputPtr[size - 1];
    U8* tail = &nextChunk[size - 1];
    U8* head = nextChunk;
    while(srcHead < srcEnd)
    {
        *head = *srcTail;
        *tail = *srcHead;
        head++;
        tail--;
        srcHead++;
        srcTail--;
    }
}



void CUDA_SYM_DECL(Transfo3_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ASSERT((size % 2) == 0);

    U32 halfSize = size >> 1;
    U32 left = 0;
    U32 right = (int)halfSize;
    U8* work = nextChunk;
    while(left < halfSize)
    {
        *work = outputPtr[left++];
        work++;
        *work = outputPtr[right++];
        work++;
    }

    RH_ASSERT(size < RH_StrideSize);                
}


void CUDA_SYM_DECL(Transfo4_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ASSERT((size % 2) == 0);
    U32 halfSize = size >> 1;

    U32 left = 0;
    U32 right = halfSize;
    U8* work = nextChunk;
    while(left < halfSize)
    {
        *work = outputPtr[right++];
        work++;
        *work = outputPtr[left++];
        work++;
    }
}


void CUDA_SYM_DECL(Transfo5_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ASSERT((size % 2) == 0);
    const U32 halfSize = size >> 1;
    S32 itt = 0;
    S32 ritt = size-1;
    size = 0;
    while(size < halfSize)
    {
        //first half
        nextChunk[size] = outputPtr[itt] ^ outputPtr[itt + 1];
        itt += 2;
        //second half
        nextChunk[size+halfSize] = outputPtr[size] ^ outputPtr[ritt];
        size++;
        ritt--;
    }
}




void CUDA_SYM_DECL(Transfo6_2)(U8* nextChunk, U32 size, U8* source)
{
    U32 i = 0;
#ifdef __CUDA_ARCH__
    while(i < size)
    {
        nextChunk[i] = ROTL8(source[i], size-i);
        i++;
    }
#else
    U8* end = nextChunk + (size >> 3)*8;
    while(((size_t)source % 8) != 0 && (i < size))
    {
        *nextChunk = ROTL8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
    
    while(nextChunk < end)
    {
        U64 res = 0; 
        U64 buf = *(U64*)source;
        source += 8;

        U32 localSize = size - i;
        U64 b;
        
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*0);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*1);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*2);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*3);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*4);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*5);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*6);
        res |= b;
        b = (U8)(buf);
        b = ROTL8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8*7);
        res |= b;

        i += 8;
        *(U64*)nextChunk = res;
        nextChunk += 8;
    }

    while(i < size)
    {
        *nextChunk = ROTL8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
#endif //cudaarch
}


void CUDA_SYM_DECL(Transfo7_2)(U8* nextChunk, U32 size, U8* source)
{
    U32 i = 0;
#ifdef __CUDA_ARCH__
    while(i < size)
    {
        nextChunk[i] = ROTR8(source[i], size-i);
        i++;
    }
#else
    U8* end = nextChunk + (size >> 3)*8;
    while(((size_t)source % 8) != 0 && (i < size))
    {
        *nextChunk = ROTR8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }

    while(nextChunk < end)
    {
        U64 res = 0;
        U64 buf = *(U64*)source;
        source += 8;

        U32 localSize = size - i;
        U64 b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize);
        localSize--;
        buf >>= 8;
        b <<= (8 * 0);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 1);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 2);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 3);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 4);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 5);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 6);
        res |= b;
        b = (U8)(buf);
        b = ROTR8((U8)b, localSize); 
        localSize--;
        buf >>= 8;
        b <<= (8 * 7);
        res |= b;

        i += 8;
        *(U64*)nextChunk = res;
        nextChunk += 8;
    }
    while(i < size)
    {
        *nextChunk = ROTR8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
#endif //#ifdef __CUDA_ARCH__
}

