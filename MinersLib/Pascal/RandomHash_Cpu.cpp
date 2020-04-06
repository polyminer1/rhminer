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

#include "precomp.h"


////////////////////////////////////////////////////
// opt
#define RH2_STRIDE_USE_MEMCPY
//#define RH2_ENABLE_TRANSFO0_MMX128
#if !defined(_WIN32_WINNT)
    #define RH2_ENABLE_MERSSEN_12_SSE4
    #define RH2_ENABLE_MERSSEN_INTERLEAVE
    #define RH2_ENABLE_EXPAND_MERSSEN_INPLACE
#else
    #define RH2_ENABLE_MERSSEN_12_SSE4 
    #define RH2_ENABLE_EXPAND_MERSSEN_INPLACE
#endif
#define RH2_ENABLE_TRANSFO_INPLACE
#define RH2_ENABLE_COMPACT_TRANSFO_67
#define RH2_ENABLE_MEM_ZERO_X_USE_MMX
#define RH2_FORCE_NO_INPLACE_MEMCPY_USE_MMX
#define RH2_ENABLE_CACHE
#define RH2_ENABLE_PREFLIGHT_CACHE
#define RH2_ENABLE_PREFLIGHT_4X
#define RH2_ENABLE_SHA256_PRERUN

//force UT
#ifdef _DEBUG
    #define RHMINER_DEBUG_RANDOMHASH_UNITTEST
#endif

#ifndef _DEBUG
    #undef RHMINER_DEBUG_RANDOMHASH_UNITTEST
#endif


#include "MinersLib/Pascal/RandomHash.h"
#include "MinersLib/Pascal/RandomHash_MurMur3_32.h"  

#include "MinersLib/Pascal/RandomHash_Blake2b.h" 
#include "MinersLib/Pascal/RandomHash_Blake2s.h"
#include "MinersLib/Pascal/RandomHash_Grindahl512.h"
#include "MinersLib/Pascal/RandomHash_Haval_5_256.h" 
#include "MinersLib/Pascal/RandomHash_MD5.h"
#include "MinersLib/Pascal/RandomHash_RadioGatun32.h"
#include "MinersLib/Pascal/RandomHash_RIPEMD160.h" 
#include "MinersLib/Pascal/RandomHash_RIPEMD256.h"
#include "MinersLib/Pascal/RandomHash_RIPEMD320.h"
#include "MinersLib/Pascal/RandomHash_SHA2_256.h"
#include "MinersLib/Pascal/RandomHash_SHA2_512.h"
#include "MinersLib/Pascal/RandomHash_SHA3_512.h"
#include "MinersLib/Pascal/RandomHash_Snefru_8_256.h"
#include "MinersLib/Pascal/RandomHash_Tiger2_5_192.h"
#include "MinersLib/Pascal/RandomHash_Whirlpool.h"

//--------------- RandomHash2 -------------------
#include "MinersLib/Pascal/RandomHash_Ghost.h"
#include "MinersLib/Pascal/RandomHash_Ghost3411.h"
#include "MinersLib/Pascal/RandomHash_MD2.h"
#include "MinersLib/Pascal/RandomHash_MD4.h"
#include "MinersLib/Pascal/RandomHash_Panama.h"
#include "MinersLib/Pascal/RandomHash_HAS160.h"
#include "MinersLib/Pascal/RandomHash_SHA.h"
#include "MinersLib/Pascal/RandomHash_Grindahl256.h"
#include "MinersLib/Pascal/RandomHash_Haval.h"
#include "MinersLib/Pascal/RandomHash_RIPEMD.h"
#include "MinersLib/Pascal/RandomHash_RIPEMD128.h"


#define RH_FULLDEBUG_CPU 
#ifdef RHMINER_RELEASE
    #undef RH_FULLDEBUG_CPU
#endif

#include "corelib/CommonData.h"

#include "MinersLib/Pascal/RandomHash_inl.h"


void RandomHash_Free(void* ptr)
{
    if (ptr)
        RH_SysFree(ptr);
}

void RandomHash_Alloc(void** out_ptr, size_t size)
{
    *out_ptr = RH_SysAlloc(size);
    RHMINER_ASSERT(*out_ptr);
}


const U32    c_AlgoSize[] = { 20, 32, 64, 48, 16, 20, 28, 32, 32, 32, 64, 32, 64, 20, 16, 20, 24, 28, 32, 
                              16, 20, 24, 28, 32, 16, 20, 24, 28, 32, 28, 32, 36, 48, 64, 16, 16, 16, 32, 
                              32, 16, 16, 20, 32, 40, 20, 20, 28, 32, 48, 64, 28, 32, 28, 32, 48, 64, 16, 
                              32, 16, 20, 24, 16, 20, 24, 16, 20, 24, 16, 20, 24, 16, 20, 24, 16, 20, 24, 64, 28,32 };
static thread_local U64                 c_target = 0xFFFFFFFFFFFFFfFF;
#define GetTarget()                     c_target

void RandomHash_CreateMany(RandomHash_State** outPtr, U32 count)
{
    RandomHash_Alloc((void**)outPtr, sizeof(RandomHash_State)*count);
    RH_CUDA_ERROR_CHECK();

    for (U32 i = 0; i < count; i++)
    {
        RandomHash_Create(&(*outPtr)[i]);
    }
} 


inline void RandomHash_Initialize(RandomHash_State* state)
{
    state->m_workBytes = state->m_precalcPart2;

    state->m_strideID = 0;
    state->m_stideDataFlipFlop = state->m_stideDataFlipFlop ? 0 : 1;
    state->m_pPartiallyComputedOutputsClone = &state->m_stideDBuffer[state->m_stideDataFlipFlop].m_partiallyComputedOutputsClone;
    state->m_pStridesInstances = state->m_stideDBuffer[state->m_stideDataFlipFlop].m_stridesInstances;
    state->m_stridesAllocIndex = 0;

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    memset(state->m_stideDBuffer[state->m_stideDataFlipFlop].m_stridesInstances, (U8)0xBA, RH2_STRIDE_BANK_SIZE);
    U64* check = (U64*)(state->m_stideDBuffer[state->m_stideDataFlipFlop].m_stridesInstances + RH2_STRIDE_BANK_SIZE);
    RHMINER_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif
}


void RandomHash_Create(RandomHash_State* state)
{
    U32 ajust = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    ajust = sizeof(U64);
#endif    
    RandomHash_Alloc((void**)&state->m_stideDBuffer[0].m_stridesInstances, RH2_STRIDE_BANK_SIZE + ajust);
    RandomHash_Alloc((void**)&state->m_stideDBuffer[1].m_stridesInstances, RH2_STRIDE_BANK_SIZE + ajust);
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    U64* check = (U64*)(state->m_stideDBuffer[0].m_stridesInstances + RH2_STRIDE_BANK_SIZE);
    *check = 0xFF55AA44BB8800DDLLU;
    check = (U64*)(state->m_stideDBuffer[1].m_stridesInstances + RH2_STRIDE_BANK_SIZE);
    *check = 0xFF55AA44BB8800DDLLU;
#endif
    RH_CUDA_ERROR_CHECK();
    state->m_stideDataFlipFlop = 0;    

    state->m_isNewHeader = true;
    RandomHash_Initialize(state);
}


void RandomHash_Destroy(RandomHash_State* state)
{
    RandomHash_Free(state->m_stideDBuffer[0].m_stridesInstances);
    RandomHash_Free(state->m_stideDBuffer[1].m_stridesInstances);
}

void RandomHash_DestroyMany(RandomHash_State* stateArray, U32 count)
{
    if (stateArray)
    {
        for (U32 i = 0; i < count; i++)
            RandomHash_Destroy(&stateArray[i]);
        RandomHash_Free(stateArray);
    }
} 


void RandomHash_SetHeader(RandomHash_State* state, U8* sourceHeader, U32 headerSize, U32 nonce2) 
{
    RHMINER_ASSERT(headerSize == PascalHeaderSizeV5); //optimized for 236 header
    U8* targetInput = state->m_header;
    state->m_isNewHeader = true;
    state->m_pascalHeaderSize = headerSize;
#ifdef RH2_ENABLE_CACHE
    state->m_partiallyComputedCount = 0;
    state->m_partiallyComputedRound = U32_Max;
    state->m_partiallyComputedNonceHeader = U32_Max;
#endif
#ifdef RH2_ENABLE_PREFLIGHT_CACHE
    state->m_isCalculatePreflight = 1;
    state->m_preflightData.endCut = 8;
#endif
    memcpy(targetInput, sourceHeader, state->m_pascalHeaderSize);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    inline RH_StridePtr RH_StrideArrayAllocOutput(RandomHash_State* state, U32 initialSize) 
    {
        RH_ASSERT((size_t(state->m_stridesAllocIndex) % 32) == 0);
        RH_ASSERT((initialSize % 4) == 0);
        RH_StridePtr stride = state->m_pStridesInstances + (state->m_stridesAllocIndex/4);
        RH_ASSERT((size_t(stride) % 32) == 0);
    #ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
        if (state->m_strideID)
            RH_ASSERT(*(U32*)((stride - 4)) == 0xBABABABA);
    #endif
        state->m_stridesAllocIndex += initialSize + RH_IDEAL_ALIGNMENT;
        RH_ASSERT(state->m_stridesAllocIndex < RH2_STRIDE_BANK_SIZE);
        RH_STRIDE_INIT_INTEGRITY(stride);
    #ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
        RH_STRIDE_SET_INDEX(stride, state->m_strideID);
        state->m_strideID++;
    #endif
        return stride;
    }

    inline void RH_StrideArrayGrow(RandomHash_State* state, RH_StridePtr stride, U32 growSize)
    {
        RH_ASSERT((growSize % 4) == 0);
        state->m_stridesAllocIndex += growSize;
        RH_ASSERT(state->m_stridesAllocIndex < RH2_STRIDE_BANK_SIZE);
        //CpuYield();
        RH_STRIDE_SET_SIZE(stride, RH_STRIDE_GET_SIZE(stride) + growSize);
        RH_STRIDE_INIT_INTEGRITY(stride);
    }


    inline void RH_StrideArrayClose(RandomHash_State* state, RH_StridePtr stride)
    {
        U32 ss = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
        ss += sizeof(U64);
#endif
        RH_ASSERT(RHMINER_ALIGN(state->m_stridesAllocIndex + ss, 32) > state->m_stridesAllocIndex);
        state->m_stridesAllocIndex = RHMINER_ALIGN(state->m_stridesAllocIndex + ss, 32);
        RH_ASSERT(state->m_stridesAllocIndex < RH2_STRIDE_BANK_SIZE);
        RH_ASSERT((size_t(state->m_stridesAllocIndex) % 32) == 0);
        RH_ASSERT((size_t(state->m_stridesAllocIndex) % 4) == 0);

        RH_STRIDE_CHECK_INTEGRITY(stride);
    }


inline U32 GetLastDWordLE(RH_StridePtr in_stride)
{
    U32 size = RH_STRIDE_GET_SIZE(in_stride);
    RH_ASSERT(size > 4);
    return *(RH_STRIDE_GET_DATA(in_stride) + ((size - 4)/4));
}

inline void SetLastDWordLE(RH_StridePtr in_stride, U32 newNonce)
{
    U32 size = RH_STRIDE_GET_SIZE(in_stride);
    RH_ASSERT(size > 4);
    *(RH_STRIDE_GET_DATA(in_stride) + ((size - 4)/4)) = newNonce;
}


    void RandomHash_Expand(RandomHash_State* state, RH_StridePtr input, U32 seed, int round, int ExpansionFactor)
    {
        U32 inputSize = RH_STRIDE_GET_SIZE(input);
#if defined(RH2_ENABLE_EXPAND_MERSSEN_USING_FAST4)
    #ifndef RH2_ENABLE_EXPAND_MERSSEN_INPLACE
        U32 mt4_sequence = 0;
        {
            U64 mt10;
            U64 mt32;
            U32 pval = (0x6c078965 * (seed ^ seed >> 30) + 1);
            mt10 = ((U64)pval) << 32 | seed;
            
            mt32 = 0x6c078965 * (pval ^ pval >> 30) + 2;
            pval = (U32)mt32;
            pval = 0x6c078965 * (pval ^ pval >> 30) + 3;
            mt32 |= ((U64)pval) << 32;
            pval = 0x6c078965 * (pval ^ pval >> 30) + 4;
            U32 mt5 = pval;

            U32 fval= 5;
            while (fval < MERSENNE_TWISTER_PERIOD)
                pval = 0x6c078965 * (pval ^ pval >> 30) + fval++;

            fval = 0x6c078965 * (pval ^ pval >> 30) + MERSENNE_TWISTER_PERIOD;
            pval = M32((U32)mt10);
            pval |= L31(mt10>>32);
            pval = fval ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

            pval ^= pval >> 11; 
            pval ^= pval << 7 & 0x9d2c5680; 
            pval ^= pval << 15 & 0xefc60000; 
            pval ^= pval >> 18;

            mt4_sequence |= (pval % 8);
            mt4_sequence <<= 8;

            fval = 0x6c078965 * (fval ^ fval >> 30) + MERSENNE_TWISTER_PERIOD+1;
            pval = M32(mt10>>32);
            pval |= L31((U32)mt32);
            pval = fval ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

            pval ^= pval >> 11; 
            pval ^= pval << 7 & 0x9d2c5680; 
            pval ^= pval << 15 & 0xefc60000; 
            pval ^= pval >> 18;

            mt4_sequence |= (pval % 8);
            mt4_sequence <<= 8;

            fval = 0x6c078965 * (fval ^ fval >> 30) + MERSENNE_TWISTER_PERIOD+2;
            pval = M32((U32)mt32);
            pval |= L31(mt32>>32);
            pval = fval ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

            pval ^= pval >> 11; 
            pval ^= pval << 7 & 0x9d2c5680; 
            pval ^= pval << 15 & 0xefc60000; 
            pval ^= pval >> 18;

            mt4_sequence |= (pval % 8);
            mt4_sequence <<= 8;

            fval = 0x6c078965 * (fval ^ fval >> 30) + MERSENNE_TWISTER_PERIOD+3;
            pval = M32(mt32>>32);
            pval |= L31(mt5);
            pval = fval ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

            pval ^= pval >> 11; 
            pval ^= pval << 7 & 0x9d2c5680; 
            pval ^= pval << 15 & 0xefc60000; 
            pval ^= pval >> 18;

            mt4_sequence |= (pval % 8);
            mt4_sequence = RH_swap_u32(mt4_sequence);
        }
    #else
        U32 rndIdx = 0;
        U64 mt10;
        U64 mt32;
        U32 pval = (0x6c078965 * (seed ^ seed >> 30) + 1);
        mt10 = ((U64)pval) << 32 | seed;
            
        mt32 = 0x6c078965 * (pval ^ pval >> 30) + 2;
        pval = (U32)mt32;
        pval = 0x6c078965 * (pval ^ pval >> 30) + 3;
        mt32 |= ((U64)pval) << 32;
        pval = 0x6c078965 * (pval ^ pval >> 30) + 4;
        U32 mt5 = pval;

        U32 fval= 5;
        while (fval < MERSENNE_TWISTER_PERIOD)
            pval = 0x6c078965 * (pval ^ pval >> 30) + fval++;

    #endif 
#else 
    #ifndef RH2_ENABLE_EXPAND_MERSSEN_INPLACE
        mersenne_twister_state   rndGenExpand;
        merssen_twister_seed_fast_partial(seed, &rndGenExpand, 4);

        merssen_twister_rand_fast_partial_4(&rndGenExpand);
        rndGenExpand.index--;
#else
        __m128i r1;
        {
            __m128i f1;
            __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
            __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
            c1 = _mm_shuffle_epi32(c1, 0);
            c2 = _mm_shuffle_epi32(c2, 0);
            U32  pval, pval2, pvaln, ip1_l;
            pval = seed;

            r1 = _mm_cvtsi32_si128(pval);
            pval = 0x6c078965 * (pval ^ pval >> 30) + 1;
            r1 = _mm_insert_epi32_M(r1, pval, 1);
            pval = 0x6c078965 * (pval ^ pval >> 30) + 2;
            r1 = _mm_insert_epi32_M(r1, pval, 2);
            pval = 0x6c078965 * (pval ^ pval >> 30) + 3;
            r1 = _mm_insert_epi32_M(r1, pval, 3);
            ip1_l = 0x6c078965 * (pval ^ pval >> 30) + 4;

            pval2 = 5;
            pval = ip1_l;
            while (pval2 <= MERSENNE_TWISTER_PERIOD)
            {
                pval = 0x6c078965 * (pval ^ pval >> 30) + pval2++;
            }
            
            f1 = _mm_cvtsi32_si128(pval);
            pval = 0x6c078965 * (pval ^ pval >> 30) + 398;
            f1 = _mm_insert_epi32_M(f1, pval, 1);
            pval = 0x6c078965 * (pval ^ pval >> 30) + 399;
            f1 = _mm_insert_epi32_M(f1, pval, 2);
            pval = 0x6c078965 * (pval ^ pval >> 30) + 400;
            f1 = _mm_insert_epi32_M(f1, pval, 3);

            RH_MT_ST(r1, f1, 0, 1);
            RH_MT_ST(r1, f1, 1, 2);
            RH_MT_ST(r1, f1, 2, 3);
            pval = _mm_extract_epi32_M(r1, 3);
            pval2 = ip1_l;
            pvaln = _mm_extract_epi32_M(f1, 3);
            pval = M32(pval) | L31(pval2);
            pval = pvaln ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);
            r1 = _mm_insert_epi32_M(r1, pval, 3);

            f1 = _mm_srli_epi32(r1, 11);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 7);
            f1 = _mm_and_si128(f1, c1);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 15);
            f1 = _mm_and_si128(f1, c2);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_srli_epi32(r1, 18);
            r1 = _mm_xor_si128(r1, f1);
        }
    #endif 
#endif
        size_t sizeExp = inputSize + ExpansionFactor * RH2_M;

        RH_StridePtr output = input;

        S64 bytesToAdd = sizeExp - inputSize;
        RH_ASSERT((bytesToAdd % 2) == 0);
        RH_ASSERT(RH_STRIDE_GET_SIZE(output) != 0);
        RH_ASSERT(RH_STRIDE_GET_SIZE(output) < RH2_StrideSize);
        RH_STRIDE_INIT_INTEGRITY(input);

        U8* outputPtr = RH_STRIDE_GET_DATA8(output);
        
        while (bytesToAdd > 0)
        {
            U32 nextChunkSize = RH_STRIDE_GET_SIZE(output);
            U8* nextChunk = outputPtr + nextChunkSize;
            if (nextChunkSize > bytesToAdd)
            {
                nextChunkSize = (U32)bytesToAdd;
            }

            RH_StrideArrayGrow(state, output, nextChunkSize);

            RH_ASSERT(nextChunk + nextChunkSize < ((U8*)(void*)output) + RH2_StrideSize);
#ifdef RH2_ENABLE_EXPAND_MERSSEN_USING_FAST4
#ifndef RH2_ENABLE_EXPAND_MERSSEN_INPLACE
            U32 r = (U8)mt4_sequence;
            mt4_sequence >>= 8;
#else
            switch (rndIdx)
            {
            case 0:
                fval = 0x6c078965 * (pval ^ pval >> 30) + MERSENNE_TWISTER_PERIOD;
                pval = M32((U32)mt10);
                pval |= L31(mt10 >> 32);
                break;
            case 1:
                fval = 0x6c078965 * (fval ^ fval >> 30) + MERSENNE_TWISTER_PERIOD + 1;
                pval = M32(mt10 >> 32);
                pval |= L31((U32)mt32);
                break;
            case 2:
                fval = 0x6c078965 * (fval ^ fval >> 30) + MERSENNE_TWISTER_PERIOD + 2;
                pval = M32((U32)mt32);
                pval |= L31(mt32 >> 32);
                break;
            case 3:
                fval = 0x6c078965 * (fval ^ fval >> 30) + MERSENNE_TWISTER_PERIOD + 3;
                pval = M32(mt32 >> 32);
                pval |= L31(mt5);
                break;
            }
            rndIdx++;
            pval = fval ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

            pval ^= pval >> 11;
            pval ^= pval << 7 & 0x9d2c5680;
            pval ^= pval << 15 & 0xefc60000;
            pval ^= pval >> 18;
            U32 r = (pval % 8);
#endif
#else
#ifndef RH2_ENABLE_EXPAND_MERSSEN_INPLACE
            U32 random = merssen_twister_rand_fast_partial_4(&rndGenExpand);
            U32 r = random % 8;
#else
            U32 r = _mm_extract_epi32_M(r1, 0);
            r1 = _mm_bsrli_si128(r1, 4);
            r = r % 8;
#endif
#endif
            RH_ASSERT((nextChunkSize & 1) == 0);
            RH_ASSERT(r >= 0 && r <= 7);
#ifdef RH2_ENABLE_TRANSFO_INPLACE

            U32 halfSize = nextChunkSize >> 1;
            switch (r)
            {
                case 0:
                {
                    halfSize = (*(U32*)(outputPtr + nextChunkSize - 4));

                    if (!halfSize)
                        halfSize = 1;

                    for (U32 i = 0; i < nextChunkSize; i++)
                    {
                        halfSize ^= halfSize << 13;
                        halfSize ^= halfSize >> 17;
                        halfSize ^= halfSize << 5;
                        *nextChunk = outputPtr[halfSize % nextChunkSize];
                        nextChunk++;
                    }
                }
                break;
                case 1:
                {
                    memcpy(nextChunk, outputPtr + halfSize, halfSize);
                    memcpy(nextChunk + halfSize, outputPtr, halfSize);
                }
                break;
                case 2:
                {
                    U8* srcHead = outputPtr;
                    U8* srcEnd = srcHead + halfSize;
                    U8* srcTail = &outputPtr[nextChunkSize - 1];
                    U8* tail = &nextChunk[nextChunkSize - 1];
                    while (srcHead < srcEnd)
                    {
                        *nextChunk = *srcTail;
                        *tail = *srcHead;
                        nextChunk++;
                        tail--;
                        srcHead++;
                        srcTail--;
                    }
                }
                break;
                case 3:
                {
                    U32 left = 0;
                    U32 right = (int)halfSize;
                    while (left < halfSize)
                    {
                        *nextChunk = outputPtr[left++];
                        nextChunk++;
                        *nextChunk = outputPtr[right++];
                        nextChunk++;
                    }
                }
                break;
                case 4:
                {
                    U8* left = outputPtr;
                    U8* right = outputPtr + halfSize;
                    while (halfSize)
                    {
                        *nextChunk = *right;
                        nextChunk++;
                        *nextChunk = *left;
                        nextChunk++;
                        right++;
                        left++;
                        halfSize--;
                    }
                }
                break;
                case 5:
                {
                    S32 itt = 0;
                    S32 ritt = nextChunkSize - 1;
                    S32 fitt = 0;
                    while (fitt < halfSize)
                    {
                        nextChunk[fitt] = outputPtr[itt] ^ outputPtr[itt + 1];
                        itt += 2;
                        nextChunk[fitt + halfSize] = outputPtr[fitt] ^ outputPtr[ritt];
                        fitt++;
                        ritt--;
                    }
                }
                break;
                case 6:
                {
                    for (int i = 0; i < nextChunkSize; i++)
                        nextChunk[i] = ROTL8(outputPtr[i], nextChunkSize - i);
                }
                break;
                case 7:
                {
                    for (int i = 0; i < nextChunkSize; i++)
                        nextChunk[i] = ROTR8(outputPtr[i], nextChunkSize - i);
                }
                break;
        }
#else
            switch (r)
            {
            case 0: Transfo0_2(nextChunk, nextChunkSize, outputPtr); break;
            case 1: Transfo1_2(nextChunk, nextChunkSize, outputPtr); break;
            case 2: Transfo2_2(nextChunk, nextChunkSize, outputPtr); break;
            case 3: Transfo3_2(nextChunk, nextChunkSize, outputPtr); break;
            case 4: Transfo4_2(nextChunk, nextChunkSize, outputPtr); break;
            case 5: Transfo5_2(nextChunk, nextChunkSize, outputPtr); break;
            case 6: Transfo6_2(nextChunk, nextChunkSize, outputPtr); break;
            case 7: Transfo7_2(nextChunk, nextChunkSize, outputPtr); break;
            }
#endif

            RH_STRIDE_CHECK_INTEGRITY(output);
            RH_ASSERT(RH_STRIDE_GET_SIZE(output) < RH2_StrideSize);
            bytesToAdd = bytesToAdd - nextChunkSize;
        }
       

        RH_StrideArrayClose(state, output);
        RH_ASSERT(sizeExp == RH_STRIDE_GET_SIZE(output));
        RH_STRIDE_CHECK_INTEGRITY(output);
    }



    void RandomHash_Compress(RandomHash_State* state, RH_StrideArrayStruct& inputs, RH_StridePtr Result, U32 seed)
    {
        U32 rval;
        mersenne_twister_state   rndGenCompress;
        merssen_twister_seed_fast_partial(seed, &rndGenCompress, 204);
        
        merssen_twister_rand_fast_partial_204(&rndGenCompress);

#if defined(RH2_ENABLE_MERSSEN_INTERLEAVE) && !defined(RHMINER_NO_SSE4)
        rndGenCompress.index -= 2;
#else
        rndGenCompress.index--;
#endif

        RH_STRIDE_SET_SIZE(Result, 100);
        U8* resultPtr = RH_STRIDE_GET_DATA8(Result);
        U32 inoutSize = RH_STRIDEARRAY_GET_SIZE(inputs);
        RH_StridePtr source;
        for (size_t i = 0; i < 100; i++)
        {
            source = RH_STRIDEARRAY_GET(inputs, merssen_twister_rand_fast_partial(&rndGenCompress, 204) % inoutSize);
            U32 sourceSize = RH_STRIDE_GET_SIZE(source);
            rval = merssen_twister_rand_fast_partial(&rndGenCompress, 204);
            resultPtr[i] = RH_STRIDE_GET_DATA8(source)[rval % sourceSize];
        }
    }

    inline void ComputeVeneerRound(RandomHash_State* state, RH_StrideArrayStruct& in_strideArray, RH_StridePtr finalHash)
    {
        U32 seed = GetLastDWordLE(RH_STRIDEARRAY_GET(in_strideArray, RH_STRIDEARRAY_GET_SIZE(in_strideArray)-1));
        RandomHash_Compress(state, in_strideArray, state->m_workBytes, seed);
        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) < WorkBytesSize);

        RandomHash_SHA2_256(state->m_workBytes, finalHash, false, false);

    }
    bool CalculateRoundOutputs(RandomHash_State* state, U32 in_round, RH_StrideArrayStruct& roundOutputs)
    {    
        U32 LSeed = 0;
        RH_ASSERT(in_round >= 1 && in_round <= RH2_MAX_N);
        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_roundInput) <= state->m_pascalHeaderSize);

#ifdef RH2_ENABLE_CACHE
        mersenne_twister_state   rndGen; 

        if (state->m_partiallyComputedCount &&
            state->m_partiallyComputedRound == in_round &&
            GetLastDWordLE(state->m_roundInput) == state->m_partiallyComputedNonceHeader)
        {
            RH_STRIDEARRAY_RESET(roundOutputs);

            RH_StrideArrayPtr prevCompClone = &state->m_stideDBuffer[state->m_stideDataFlipFlop ? 0 : 1].m_partiallyComputedOutputsClone;
            RH_STRIDEARRAY_COPY_ALL(roundOutputs, *prevCompClone);

           

            state->m_partiallyComputedCount = 0;
            state->m_partiallyComputedRound = U32_Max;
            state->m_partiallyComputedNonceHeader = U32_Max;
            return 0; 
        }
#endif
        if (in_round == 1)
        {
#ifdef RH2_ENABLE_PREFLIGHT_CACHE
    #ifndef RH2_ENABLE_PREFLIGHT_4X
            RandomHash_SHA2_256_Part2(state->m_roundInput, state->m_preflightData, state->m_workBytes);
    #endif
#else
            RandomHash_SHA2_256(state->m_roundInput, state->m_workBytes);
#endif
            LSeed = GetLastDWordLE(state->m_workBytes);
            merssen_twister_seed_fast_partial(LSeed, &rndGen,12);
            RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_roundInput) == state->m_pascalHeaderSize);

        #ifdef RH2_ENABLE_SHA256_PRERUN
            LSeed = merssen_twister_rand_fast_partial_12(&rndGen) % RH2_MAX_ALGO;
            rndGen.index--;
        #endif


        }
        else
        {
            RH_StrideArrayStruct parenAndNeighbortOutputs;
#ifdef RH2_ENABLE_PREFLIGHT_4X
            if (in_round - 1 == 1)
                RandomHash_SHA2_256_Part2(state->m_roundInput, state->m_preflightData, state->m_workBytes);
#endif
            if (CalculateRoundOutputs(state, in_round - 1, parenAndNeighbortOutputs) == true)
            {
                RH_STRIDEARRAY_RESET(roundOutputs); 
                RH_STRIDEARRAY_COPY_ALL(roundOutputs, parenAndNeighbortOutputs);
                return true;
            }

            LSeed = GetLastDWordLE( RH_STRIDEARRAY_GET(parenAndNeighbortOutputs, RH_STRIDEARRAY_GET_SIZE(parenAndNeighbortOutputs)-1) );
            merssen_twister_seed_fast_partial(LSeed, &rndGen, 12);
            merssen_twister_rand_fast_partial_12(&rndGen);
            rndGen.index--;

            RH_STRIDEARRAY_COPY_ALL(roundOutputs, parenAndNeighbortOutputs); 
            RH_STRIDEARRAY_RESET(parenAndNeighbortOutputs);            
            U32 LNumNeighbours = (merssen_twister_rand_fast_partial_12(&rndGen) % (RH2_MAX_J - RH2_MIN_J)) + RH2_MIN_J;
            
#ifdef RH2_ENABLE_PREFLIGHT_4X
            state->m_workBytes = state->m_precalcPart2;
            U32* MT = &rndGen.MT[rndGen.index];
            if (LNumNeighbours >= 4)
            {
                SetLastDWordLE(state->m_roundInput, *MT);
                RandomHash_SHA2_256_Part2_SSE_4x(state->m_roundInput, state->m_preflightData,
                    MT[1], MT[2], MT[3],
                    state->m_workBytes + 0 * WorkBytesSize32,
                    state->m_workBytes + 1 * WorkBytesSize32,
                    state->m_workBytes + 2 * WorkBytesSize32,
                    state->m_workBytes + 3 * WorkBytesSize32);
                state->m_workBytes += 4*WorkBytesSize32;
                MT += 4;
            }
            switch(LNumNeighbours % 4)
            {
            case 3:
            case 2:
                SetLastDWordLE(state->m_roundInput, *MT);
                RandomHash_SHA2_256_Part2_SSE_4x(state->m_roundInput, state->m_preflightData,
                    MT[1], MT[2], MT[3],
                    state->m_workBytes + 0 * WorkBytesSize32,
                    state->m_workBytes + 1 * WorkBytesSize32,
                    state->m_workBytes + 2 * WorkBytesSize32,
                    state->m_workBytes + 3 * WorkBytesSize32);
                state->m_workBytes += 4*WorkBytesSize32;
                MT += 4;
                break;

            case 1:
                SetLastDWordLE(state->m_roundInput, *MT);
                RandomHash_SHA2_256_Part2(state->m_roundInput, state->m_preflightData, state->m_workBytes);
                state->m_workBytes += WorkBytesSize32;
                MT++;
            }

            state->m_workBytes = state->m_precalcPart2 - WorkBytesSize32;
#endif 


            for (U32 i = 0; i < LNumNeighbours; i++)
            {
                LSeed = merssen_twister_rand_fast_partial_12(&rndGen);
                SetLastDWordLE(state->m_roundInput, LSeed); 
#ifdef RH2_ENABLE_PREFLIGHT_4X
                state->m_workBytes += WorkBytesSize32;
                RH_ASSERT(state->m_workBytes < state->m_precalcPart2 + sizeof(state->m_precalcPart2)/4);
#endif
                bool LNeighbourWasLastRound = CalculateRoundOutputs(state, in_round - 1, parenAndNeighbortOutputs);
                RH_STRIDEARRAY_PUSHBACK_MANY(roundOutputs, parenAndNeighbortOutputs); 
                
#ifdef RH2_ENABLE_CACHE
                if (LNeighbourWasLastRound)
                {
                    if (state->out_hashes->count < RandomHashResult::MaxHashes)
                    {
                        state->out_hashes->nonces[state->out_hashes->count] = LSeed;

                        ComputeVeneerRound(state, parenAndNeighbortOutputs, state->out_hashes->hashes[state->out_hashes->count]);
                        state->out_hashes->count++;
                    }
                    else
                    {
                        PrintOut("Warning: Missed internal hash.\n");
                    }
                }
                else
                {
                    U32 pround = in_round - 1;
                    if ((pround >= 3))
                    {
                        {
                            state->m_partiallyComputedLocalCount++;
                            state->m_partiallyComputedCount++;
                            state->m_partiallyComputedRound = pround;
                            state->m_partiallyComputedNonceHeader = LSeed;


                            RH_STRIDEARRAY_RESET(*(state->m_pPartiallyComputedOutputsClone));
                            RH_STRIDEARRAY_COPY_ALL(*(state->m_pPartiallyComputedOutputsClone), parenAndNeighbortOutputs);

                        }
                    }
                }
#endif
                RH_STRIDEARRAY_RESET(parenAndNeighbortOutputs);
            }
            
            LSeed = merssen_twister_rand_fast_partial_12(&rndGen);
            RandomHash_Compress(state, roundOutputs, state->m_workBytes, LSeed);
            RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) <= 100);
        }
        U32 rndHash = (merssen_twister_rand_fast_partial_12(&rndGen) % RH2_MAX_ALGO);
        RH_StridePtr input = state->m_workBytes;
        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) < WorkBytesSize);
        RH_StridePtr output = RH_StrideArrayAllocOutput(state, c_AlgoSize[rndHash]);
        switch(rndHash)
        {
            case RH2_Blake2B_160:       RandomHash_blake2b(input, output, 160); break; 
            case RH2_Blake2B_256:       RandomHash_blake2b(input, output, 256); break;
            case RH2_Blake2B_512:       RandomHash_blake2b(input, output, 512); break;
            case RH2_Blake2B_384:       RandomHash_blake2b(input, output, 384); break;
            case RH2_Blake2S_128:       RandomHash_blake2s(input, output, 128); break;
            case RH2_Blake2S_160:       RandomHash_blake2s(input, output, 160); break;
            case RH2_Blake2S_224:       RandomHash_blake2s(input, output, 224); break;
            case RH2_Blake2S_256:       RandomHash_blake2s(input, output, 256); break;
            case RH2_Gost              :RandomHash_Ghost(input, output); break;
            case RH2_GOST3411_2012_256 :RandomHash_Ghost3411(input, output, 256); break;
            case RH2_GOST3411_2012_512 :RandomHash_Ghost3411(input, output, 512); break;
            case RH2_Grindahl256       :RandomHash_Grindahl256(input, output); break;
            case RH2_Grindahl512       :RandomHash_Grindahl512(input, output); break;
            case RH2_HAS160            :RandomHash_HAS160(input, output); break;
            case RH2_Haval_3_128       :RandomHash_Haval3 (input, output, 128); break;
            case RH2_Haval_3_160       :RandomHash_Haval3 (input, output, 160); break;
            case RH2_Haval_3_192       :RandomHash_Haval3 (input, output, 192); break;
            case RH2_Haval_3_224       :RandomHash_Haval3 (input, output, 224); break;
            case RH2_Haval_3_256       :RandomHash_Haval3 (input, output, 256); break;
            case RH2_Haval_4_128       :RandomHash_Haval4 (input, output, 128); break;
            case RH2_Haval_4_160       :RandomHash_Haval4 (input, output, 160); break;
            case RH2_Haval_4_192       :RandomHash_Haval4 (input, output, 192); break;
            case RH2_Haval_4_224       :RandomHash_Haval4 (input, output, 224); break;
            case RH2_Haval_4_256       :RandomHash_Haval4 (input, output, 256); break;
            case RH2_Haval_5_128       :RandomHash_Haval_5_256 (input, output, 128); break;
            case RH2_Haval_5_160       :RandomHash_Haval_5_256 (input, output, 160); break;
            case RH2_Haval_5_192       :RandomHash_Haval_5_256 (input, output, 192); break;
            case RH2_Haval_5_224       :RandomHash_Haval_5_256 (input, output, 224); break;
            case RH2_Haval_5_256       :RandomHash_Haval_5_256 (input, output, 256); break;
            case RH2_Keccak_224        :_RandomHash_SHA3_512  (input, output, 224/8, false); break;
            case RH2_Keccak_256        :_RandomHash_SHA3_512  (input, output, 256/8, false); break;
            case RH2_Keccak_288        :_RandomHash_SHA3_512  (input, output, 288/8, false); break;
            case RH2_Keccak_384        :_RandomHash_SHA3_512  (input, output, 384/8, false); break;
            case RH2_Keccak_512        :_RandomHash_SHA3_512  (input, output, 512/8, false); break;
            case RH2_MD2               :RandomHash_MD2         (input, output); break;
            case RH2_MD5               :RandomHash_MD5         (input, output); break;
            case RH2_MD4               :RandomHash_MD4         (input, output); break;
            case RH2_Panama            :RandomHash_Panama      (input, output); break;
            case RH2_RadioGatun32      :RandomHash_RadioGatun32(input, output); break;
            case RH2_RIPEMD            :RandomHash_RIPEMD      (input, output); break;
            case RH2_RIPEMD128         :RandomHash_RIPEMD128   (input, output); break;
            case RH2_RIPEMD160         :RandomHash_RIPEMD160   (input, output); break;
            case RH2_RIPEMD256         :RandomHash_RIPEMD256   (input, output); break;
            case RH2_RIPEMD320         :RandomHash_RIPEMD320   (input, output); break;
            case RH2_SHA0              :RandomHash_SHA0        (input, output); break;
            case RH2_SHA1              :RandomHash_SHA1        (input, output); break;
            case RH2_SHA2_224          :RandomHash_SHA2_256    (input, output, true); break;                 
            case RH2_SHA2_256          :RandomHash_SHA2_256    (input, output, false); break;
            case RH2_SHA2_384          :RandomHash_SHA2_512    (input, output, SHA2_512_MODE_384); break;
            case RH2_SHA2_512          :RandomHash_SHA2_512    (input, output, SHA2_512_MODE_512); break;
            case RH2_SHA2_512_224      :RandomHash_SHA2_512    (input, output, SHA2_512_MODE_512_224); break;
            case RH2_SHA2_512_256      :RandomHash_SHA2_512    (input, output, SHA2_512_MODE_512_256); break;
            case RH2_SHA3_224          :RandomHash_SHA3_224    (input, output); break;
            case RH2_SHA3_256          :RandomHash_SHA3_256    (input, output); break;
            case RH2_SHA3_384          :RandomHash_SHA3_384    (input, output); break;
            case RH2_SHA3_512          :RandomHash_SHA3_512    (input, output); break;
            case RH2_Snefru_8_128      :RandomHash_Snefru_8_256(input, output, 16); break;
            case RH2_Snefru_8_256      :RandomHash_Snefru_8_256(input, output, 32); break;
            case RH2_Tiger_3_128       :RandomHash_Tiger(input, output, 3, 128, false); break;
            case RH2_Tiger_3_160       :RandomHash_Tiger(input, output, 3, 160, false); break;
            case RH2_Tiger_3_192       :RandomHash_Tiger(input, output, 3, 192, false); break;
            case RH2_Tiger_4_128       :RandomHash_Tiger(input, output, 4, 128, false); break;
            case RH2_Tiger_4_160       :RandomHash_Tiger(input, output, 4, 160, false); break;
            case RH2_Tiger_4_192       :RandomHash_Tiger(input, output, 4, 192, false); break;
            case RH2_Tiger_5_128       :RandomHash_Tiger(input, output, 5, 128, false); break;
            case RH2_Tiger_5_160       :RandomHash_Tiger(input, output, 5, 160, false); break;
            case RH2_Tiger_5_192       :RandomHash_Tiger(input, output, 5, 192, false); break;
            case RH2_Tiger2_3_128      :RandomHash_Tiger(input, output, 3, 128, true); break;
            case RH2_Tiger2_3_160      :RandomHash_Tiger(input, output, 3, 160, true); break;
            case RH2_Tiger2_3_192      :RandomHash_Tiger(input, output, 3, 192, true); break;
            case RH2_Tiger2_4_128      :RandomHash_Tiger(input, output, 4, 128, true); break;
            case RH2_Tiger2_4_160      :RandomHash_Tiger(input, output, 4, 160, true); break;
            case RH2_Tiger2_4_192      :RandomHash_Tiger(input, output, 4, 192, true); break;
            case RH2_Tiger2_5_128      :RandomHash_Tiger(input, output, 5, 128, true); break;
            case RH2_Tiger2_5_160      :RandomHash_Tiger(input, output, 5, 160, true); break;
            case RH2_Tiger2_5_192      :RandomHash_Tiger(input, output, 5, 192, true); break;
            case RH2_WhirlPool         :RandomHash_WhirlPool(input, output); break;
        }


        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) <= 100);
        RH_ASSERT(RH_STRIDE_GET_SIZE(output) == c_AlgoSize[rndHash]);


        LSeed = merssen_twister_rand_fast_partial_12(&rndGen);
        RandomHash_Expand(state, 
                               output, 
                               LSeed, 
                               in_round, 
                               RH2_MAX_N - in_round);
        RH_STRIDEARRAY_PUSHBACK(roundOutputs, output);

        bool finalResult = (in_round == RH2_MAX_N) || ((in_round >= RH2_MIN_N) && ((GetLastDWordLE(output) % RH2_MAX_N) == 0));
        return finalResult;
    }


    void RandomHash_Init(RandomHash_State* allStates, U32 startNonce)
    {
        CUDA_DECLARE_STATE();

        RH_ASSERT(state->m_pascalHeaderSize > 0);

        RandomHash_Initialize(state);

        if (state->m_isNewHeader)
        {
            state->m_isNewHeader = false;
            RH_STRIDE_SET_SIZE(state->m_roundInput, state->m_pascalHeaderSize);
            memcpy(RH_STRIDE_GET_DATA(state->m_roundInput), &state->m_header[0], state->m_pascalHeaderSize); 
            RH_STRIDE_INIT_INTEGRITY(state->m_roundInput);

            if (state->m_isCalculatePreflight)
            {
                state->m_isCalculatePreflight = 0;
                RandomHash_SHA2_256_Part1(RH_STRIDE_GET_DATA(state->m_roundInput), RH_STRIDE_GET_SIZE(state->m_roundInput), state->m_preflightData);
            }
        }
        else
        {
            if (state->m_partiallyComputedNonceHeader && state->m_partiallyComputedNonceHeader != U32_Max)
            {
                startNonce = state->m_partiallyComputedNonceHeader;
            }
        }
        state->out_hashes->count = 1;
        state->out_hashes->nonces[0] = startNonce;
        SetLastDWordLE(state->m_roundInput, startNonce);
    }
    
  

    void RandomHash_Search(RandomHash_State* in_state, RandomHashResult& out_hashes, U32 startNonce)
    {
        in_state->m_partiallyComputedLocalCount = 0;
        in_state->out_hashes = &out_hashes;
        RandomHash_Init(in_state, startNonce);
        RH_StrideArrayStruct roundOutputs;
        CalculateRoundOutputs(in_state, RH2_MAX_N, roundOutputs);
        ComputeVeneerRound(in_state, roundOutputs, out_hashes.hashes[0]);
        if(in_state->m_partiallyComputedLocalCount == 0)
        {            
            in_state->m_partiallyComputedCount = 0;
            in_state->m_partiallyComputedRound = U32_Max;
            in_state->m_partiallyComputedNonceHeader = U32_Max;
            RH_STRIDEARRAY_RESET(*in_state->m_pPartiallyComputedOutputsClone);
        }

        RH_ASSERT(out_hashes.count <= RandomHashResult::MaxHashes);
    }
