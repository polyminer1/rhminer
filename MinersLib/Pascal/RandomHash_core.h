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
#include "MinersLib/Pascal/RandomHash_MurMur3_32_def.h"
#include "MinersLib/Pascal/RandomHash_mersenne_twister.h"

enum RandomHash2Algos
{
    RH2_Blake2B_160 = 0,
    RH2_Blake2B_256 = 1,
    RH2_Blake2B_512 = 2,
    RH2_Blake2B_384 = 3,
    RH2_Blake2S_128 = 4,
    RH2_Blake2S_160 = 5,
    RH2_Blake2S_224 = 6,
    RH2_Blake2S_256 = 7,
    RH2_Gost = 8,
    RH2_GOST3411_2012_256 = 9,
    RH2_GOST3411_2012_512 = 10,
    RH2_Grindahl256 = 11,
    RH2_Grindahl512 = 12,
    RH2_HAS160 = 13,
    RH2_Haval_3_128 = 14,
    RH2_Haval_3_160 = 15,
    RH2_Haval_3_192 = 16,
    RH2_Haval_3_224 = 17,
    RH2_Haval_3_256 = 18,
    RH2_Haval_4_128 = 19,
    RH2_Haval_4_160 = 20,
    RH2_Haval_4_192 = 21,
    RH2_Haval_4_224 = 22,
    RH2_Haval_4_256 = 23,
    RH2_Haval_5_128 = 24,
    RH2_Haval_5_160 = 25,
    RH2_Haval_5_192 = 26,
    RH2_Haval_5_224 = 27,
    RH2_Haval_5_256 = 28,
    RH2_Keccak_224 = 29,
    RH2_Keccak_256 = 30,
    RH2_Keccak_288 = 31,
    RH2_Keccak_384 = 32,
    RH2_Keccak_512 = 33,
    RH2_MD2 = 34,
    RH2_MD5 = 35,
    RH2_MD4 = 36,
    RH2_Panama = 37,
    RH2_RadioGatun32 = 38,
    RH2_RIPEMD = 39,
    RH2_RIPEMD128 = 40,
    RH2_RIPEMD160 = 41,
    RH2_RIPEMD256 = 42,
    RH2_RIPEMD320 = 43,
    RH2_SHA0 = 44,
    RH2_SHA1 = 45,
    RH2_SHA2_224 = 46,
    RH2_SHA2_256 = 47,
    RH2_SHA2_384 = 48,
    RH2_SHA2_512 = 49,
    RH2_SHA2_512_224 = 50,
    RH2_SHA2_512_256 = 51,
    RH2_SHA3_224 = 52,
    RH2_SHA3_256 = 53,
    RH2_SHA3_384 = 54,
    RH2_SHA3_512 = 55,
    RH2_Snefru_8_128 = 56,
    RH2_Snefru_8_256 = 57,
    RH2_Tiger_3_128 = 58,
    RH2_Tiger_3_160 = 59,
    RH2_Tiger_3_192 = 60,
    RH2_Tiger_4_128 = 61,
    RH2_Tiger_4_160 = 62,
    RH2_Tiger_4_192 = 63,
    RH2_Tiger_5_128 = 64,
    RH2_Tiger_5_160 = 65,
    RH2_Tiger_5_192 = 66,
    RH2_Tiger2_3_128 = 67,
    RH2_Tiger2_3_160 = 68,
    RH2_Tiger2_3_192 = 69,
    RH2_Tiger2_4_128 = 70,
    RH2_Tiger2_4_160 = 71,
    RH2_Tiger2_4_192 = 72,
    RH2_Tiger2_5_128 = 73,
    RH2_Tiger2_5_160 = 74,
    RH2_Tiger2_5_192 = 75,
    RH2_WhirlPool = 76,
    RH2_MAX_ALGO = 77
};

//------------------------------------------------------------------------------------
typedef U32* RH_StridePtr;

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    #define RH_STRIDE_GET_INDEX(strideVar)                  (0)
    #define RH_STRIDE_SET_INDEX(strideVar, val)             {}
    #define RH_STRIDE_INIT_INTEGRITY(strideVar)         {U64* ic = (U64*)((strideVar)+RH_IDEAL_ALIGNMENT+RH_STRIDE_GET_SIZE(strideVar)); *ic = (U64)0xAABBCCDDEEFF5577LLU;}
    #define RH_STRIDE_CHECK_INTEGRITY(strideVar)        {RH_ASSERT(*(U64*)((strideVar)+RH_IDEAL_ALIGNMENT+RH_STRIDE_GET_SIZE(strideVar)) == (U64)0xAABBCCDDEEFF5577LLU);}
#else
    #define RH_STRIDE_GET_INDEX(strideVar)                  (0)
    #define RH_STRIDE_SET_INDEX(strideVar, val)             {}
    #define RH_STRIDE_INIT_INTEGRITY(strideVar)              {}
    #define RH_STRIDE_CHECK_INTEGRITY(strideVar)   {}
#endif



#define RH_STRIDE_GET_SIZE(strideVar)                   (*strideVar)
#define RH_STRIDE_SET_SIZE(strideVar, val)              {*strideVar = (U32)(val);}

#define RH_STRIDE_RESET(strideVar)                       {*strideVar = 0;}
#define RH_STRIDE_GET_DATA(strideVar)                    ((strideVar) + RH_IDEAL_ALIGNMENT32) 
#define RH_STRIDE_GET_DATA8(strideVar)                   reinterpret_cast<U8*>((strideVar) + RH_IDEAL_ALIGNMENT32)
#define RH_STRIDE_GET_DATA64(strideVar)                  reinterpret_cast<U64*>((strideVar) + RH_IDEAL_ALIGNMENT32)


#define RH_STRIDEARRAY_GET_SIZE(strideArrayVar)         (strideArrayVar).size
#define RH_STRIDEARRAY_SET_SIZE(strideArrayVar, val)    (strideArrayVar).size = (val);
#define RH_STRIDEARRAY_GET_MAXSIZE(strideArrayVar)      (strideArrayVar).maxSize
#define RH_STRIDEARRAY_GET_EXTRA(strideArrayVar, field) (strideArrayVar).field
#define RH_STRIDEARRAY_RESET(strideArrayVar)            {(strideArrayVar).size = 0;}
#define RH_STRIDEARRAY_GET(strideArrayVar, idx)         (strideArrayVar).strides[idx]


#define RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc) \
    U32 cnt = RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc); \
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc) <= RH2_StrideArrayCount); \
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVarSrc) <= RH_STRIDEARRAY_GET_MAXSIZE(strideArrayVarSrc)); \
    RH_StridePtr* endStridePtr = (strideArrayVarSrc).strides + cnt; \
    RH_StridePtr* stridePtr = (strideArrayVarSrc).strides; \
    while (stridePtr != endStridePtr)  \
    { \
        RH_StridePtr strideItrator = *stridePtr; \

#define RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc) \
        stridePtr++; \
    } \

#define CUDA_DECLARE_STATE() RandomHash_State* state = allStates;

#define RH_STRIDE_COPY(dstStride, srcStride)                                                 \
{                                                                                            \
    U32 _ss = RH_STRIDE_GET_SIZE(srcStride);                                                 \
    RH_STRIDE_CHECK_INTEGRITY(srcStride);                                                    \
    RH_STRIDE_MEMCPY_ALIGNED_SIZE128(dstStride, srcStride, _ss + RH_IDEAL_ALIGNMENT32);   \
    RH_STRIDE_CHECK_INTEGRITY(dstStride);                                                    \
}

struct RH_StrideArrayStruct
{
    U32 size;
    RH_StridePtr strides[RH2_StrideArrayCount];
    U32 maxSize;
    U64 memoryboost;
    U64 supportsse41;
    RH_StrideArrayStruct() :size(0)
#ifndef RH_DISABLE_RH_ASSERTS
    ,maxSize(RH2_StrideArrayCount)
#endif
    {}
};
typedef RH_StrideArrayStruct* RH_StrideArrayPtr;


PLATFORM_CONST uint32_t RH_RIPEMD_C1 = 0x50A28BE6;
PLATFORM_CONST uint32_t RH_RIPEMD_C2 = 0x5A827999;
PLATFORM_CONST uint32_t RH_RIPEMD_C3 = 0x5C4DD124;
PLATFORM_CONST uint32_t RH_RIPEMD_C4 = 0x6ED9EBA1;
PLATFORM_CONST uint32_t RH_RIPEMD_C5 = 0x6D703EF3;
PLATFORM_CONST uint32_t RH_RIPEMD_C6 = 0x8F1BBCDC;
PLATFORM_CONST uint32_t RH_RIPEMD_C7 = 0x7A6D76E9;
PLATFORM_CONST uint32_t RH_RIPEMD_C8 = 0xA953FD4E;

enum SHA2_512_MODE
{
    SHA2_512_MODE_384,
    SHA2_512_MODE_512,
    SHA2_512_MODE_512_224,
    SHA2_512_MODE_512_256,
};

PLATFORM_CONST uint64_t Tiger_C1 = 0xA5A5A5A5A5A5A5A5;
PLATFORM_CONST uint64_t Tiger_C2 = 0x0123456789ABCDEF;

struct SHA2_256_SavedState
{
    U32 state[8];
    int32_t len;
    uint32_t nextCut;
    uint32_t endCut;
    uint64_t bits;
};





inline void swap_copy_str_to_u32(const void *src, void *dest, const int32_t length)
{
    uint32_t *lsrc, *ldest, *lend;
    uint8_t *lbsrc;
    int32_t	lLength;

    if (((int32_t((uint8_t *)(dest)-(uint8_t *)(0)) | ((uint8_t *)(src)-(uint8_t *)(0))  | length) & 3) == 0)
    {
        lsrc = (uint32_t *)((uint8_t *)(src));
        lend = (uint32_t *)(((uint8_t *)(src)) + length);
        ldest = (uint32_t *)((uint8_t *)(dest));
        while (lsrc < lend)
        {
            *ldest = RH_swap_u32(*lsrc);
            ldest += 1;
            lsrc += 1;
        } 
    } 

    else
    {
        lbsrc = ((uint8_t *)(src));
        int32_t dest_index = 0;
        lLength = length + dest_index;
        while (dest_index < lLength)
        {
            ((uint8_t *)dest)[dest_index ^ 3] = *lbsrc;

            lbsrc += 1;
            dest_index += 1;
        }
    }
}

inline void swap_copy_str_to_u64(const void *src, void *dest, const int32_t length)
{
    uint64_t *lsrc, *ldest, *lend;
    uint8_t *lbsrc;
    int32_t	lLength;


    if (((int32_t((uint8_t *)(dest)-(uint8_t *)(0)) | ((uint8_t *)(src)-(uint8_t *)(0)) | length) & 7) == 0)
    {
        lsrc = (uint64_t *)((uint8_t *)(src));
        lend = (uint64_t *)(((uint8_t *)(src)) + length);
        ldest = (uint64_t *)((uint8_t *)(dest));
        while (lsrc < lend)
        {
            *ldest = RH_swap_u64(*lsrc);
            ldest += 1;
            lsrc += 1;
        } 
    } 
    else
    {
        lbsrc = ((uint8_t *)(src));
        uint32_t dest_index = 0;
        lLength = length + dest_index;
        while (dest_index < (U32)lLength)
        {
            ((uint8_t *)dest)[dest_index ^ 7] = *lbsrc;

            lbsrc += 1;
            dest_index += 1;
        } 
    } 
}


