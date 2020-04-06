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
#include "corelib/rh_endian.h"

#ifdef RHMINER_PLATFORM_CPU
extern int  g_memoryBoostLevel;

#define _RH_LD_LINE_128(i) r##i = RH_MM_LOAD128((__m128i *)(source)); 
#define RH_LD_LINE_128(i) r##i = RH_MM_LOAD128((__m128i *)(source)); source += sizeof(__m128i);

#endif //CPU


inline void RH_INPLACE_MEMCPY_128(U8* pDst, U8* pSrc, size_t byteCount)
{
    RH_ASSERT(((size_t)pDst % 32) == 0);
    RH_ASSERT(((size_t)pSrc % 32) == 0);
#ifndef RH2_FORCE_NO_INPLACE_MEMCPY_USE_MMX
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
#endif
        memcpy(pDst, pSrc, byteCount);
}



inline void RH_STRIDE_MEMCPY_ALIGNED_SIZE128(U8 *pDst, U8 *pSrc, size_t byteCount)
{
    RH_ASSERT(( (size_t)pDst % 8) == 0);
    RH_ASSERT(( (size_t)pSrc % 8) == 0);
    RH_ASSERT(( (size_t)pDst % 32) == 0);
    RH_ASSERT(( (size_t)pSrc % 32) == 0);

    RH_INPLACE_MEMCPY_128(pDst, pSrc, byteCount);
}



#define RH_STRIDEARRAY_PUSHBACK(strideArrayVar, stride)                     \
{                                                                           \
    U32 _as = RH_STRIDEARRAY_GET_SIZE(strideArrayVar)++;                    \
    RH_ASSERT(_as < RH_STRIDEARRAY_GET_MAXSIZE(strideArrayVar));            \
    (strideArrayVar).strides[_as] = (stride);     \
}


inline void RH_STRIDEARRAY_PUSHBACK_MANY(RH_StrideArrayStruct& strideArrayVar, RH_StrideArrayStruct& strideArrayVarSrc)
{
    U32 ssize = RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc);
    U32 dsize = RH_STRIDEARRAY_GET_SIZE(strideArrayVar);

    RH_ASSERT(&strideArrayVar != &strideArrayVarSrc);
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVar) + RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc) < RH_STRIDEARRAY_GET_MAXSIZE(strideArrayVar));
  #ifdef RH2_STRIDE_USE_MEMCPY
    memcpy(&strideArrayVar.strides[dsize], &strideArrayVarSrc.strides[0], sizeof(strideArrayVarSrc.strides[0]) * ssize);
    RH_STRIDEARRAY_GET_SIZE(strideArrayVar) += ssize;
  #else
    RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc)
    {                                                                           
        RH_STRIDEARRAY_PUSHBACK(strideArrayVar, strideItrator);  
        RH_STRIDE_CHECK_INTEGRITY(strideItrator);
    }                                                                           
    RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc)
  #endif
} 

void RH_STRIDEARRAY_COPY_ALL(RH_StrideArrayStruct& strideArrayVar, RH_StrideArrayStruct& strideArrayVarSrc)
{
    U32 srcCnt = RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc);
    RH_STRIDEARRAY_SET_SIZE(strideArrayVar, srcCnt);
    RH_ASSERT(&strideArrayVar != &strideArrayVarSrc);

  #ifdef RH2_STRIDE_USE_MEMCPY
    memcpy(&(strideArrayVar).strides[0], &(strideArrayVarSrc).strides[0], sizeof((strideArrayVarSrc).strides[0]) * srcCnt);
  #else
    U32 i = 0;
    while(i < srcCnt)
    {
        (strideArrayVar).strides[i] = (strideArrayVarSrc).strides[i];
        RH_STRIDE_CHECK_INTEGRITY((strideArrayVar).strides[i]);
        i++;
    }
  #endif
}



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

#ifdef RH2_ENABLE_TRANSFO0_MMX128
inline void Transfo0_2(U8* nextChunk, U32 size, U8* source)
{
    RH_ASSERT(size <= 128);
    U32 rndState;
    rndState = (*(U32*)(source+size-4));

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
        default: RH_ASSERT(false);
    }

    U8* head = nextChunk;
    U8* end = head + size;
    
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
                        RH_ASSERT(false);                              \
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
            default: RH_ASSERT(false);
        }
        
        *head = b;
        head++;
    }
}


#else


inline void Transfo0_2(U8* nextChunk, U32 size, U8* source)
{
    U32 rndState;
    rndState = (*(U32*)(source+size-4));

    if (!rndState)
        rndState = 1;

    for(U32 i=0; i < size; i++)
    {                
        rndState ^= rndState << 13;
        rndState ^= rndState >> 17;
        rndState ^= rndState << 5;
        *nextChunk = source[rndState % size];
        nextChunk++;
    }
}
#endif


void Transfo1_2(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 halfSize = size >> 1;
    RH_ASSERT((size % 2) == 0);


    memcpy(nextChunk, outputPtr + halfSize , halfSize);
    memcpy(nextChunk + halfSize, outputPtr, halfSize);
}

void Transfo2_2(U8* nextChunk, U32 size, U8* outputPtr)
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


void Transfo3_2(U8* nextChunk, U32 size, U8* outputPtr)
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

#ifdef RH_ENABLE_ASSERT
    {
        RH_ASSERT(size < RH2_StrideSize);
    }
#endif
    
}


void Transfo4_2(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ASSERT((size % 2) == 0);
    U32 halfSize = (size >> 1);
    U8* left = outputPtr;
    U8* right = outputPtr + halfSize;
    U8* work = nextChunk;
    while (halfSize)
    {
        *work = *right;
        work++;
        *work = *left;
        work++;
        right++;
        left++;
        halfSize--;
    }
}

void Transfo5_2(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ASSERT((size % 2) == 0);
    const U32 halfSize = size >> 1;
    S32 itt = 0;
    S32 ritt = size-1;
    size = 0;
    while(size < halfSize)
    {
        
        nextChunk[size] = outputPtr[itt] ^ outputPtr[itt + 1];
        itt += 2;
        
        nextChunk[size+halfSize] = outputPtr[size] ^ outputPtr[ritt];
        size++;
        ritt--;
    }
}



void Transfo6_2(U8* nextChunk, U32 size, U8* source)
{
    U32 i = 0;
#if defined(RH2_ENABLE_COMPACT_TRANSFO_67)
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
#endif 
}


void Transfo7_2(U8* nextChunk, U32 size, U8* source)
{
    U32 i = 0;
#if defined(RH2_ENABLE_COMPACT_TRANSFO_67)
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
#endif 
}

