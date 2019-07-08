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

extern bool g_isSSE3Supported;
extern bool g_isSSE4Supported;
extern bool g_isAVX2Supported;



#include "corelib/CommonData.h"

//--------------------------------------------------------------------------------------------------
const size_t c_round_outputsCounts[6] = {0, 1, 3, 7, 15, 31 };
const size_t c_round_parenAndNeighbortOutputsCounts[6] = { 31, 0, 1, 3, 7, 15 };
const U32    c_AlgoSize[] = { 32,48,64,32,48,64,20,32,40,64,32,24,32,64,32,16,32,64};

static thread_local U64                 c_target = 0xFFFFFFFFFFFFFFFF;
#define GetTarget()                     c_target
#define GetRoundOutputCount(n)          c_round_outputsCounts[n]
#define GetParentRoundOutputCount(n)    c_round_parenAndNeighbortOutputsCounts[n]
#define GetDeviceID()                   0

CUDA_DECL_HOST_AND_DEVICE
inline RH_StridePtr CUDA_SYM(RH_StrideArrayGet)(RH_StridePtrArray strideArrayVar, int idx) 
{
    RH_ASSERT(idx <= (int)RH_STRIDEARRAY_GET_MAXSIZE(strideArrayVar));
    return ((RH_StrideArrayStruct*)strideArrayVar)->strides[idx];
}


CUDA_DECL_HOST_AND_DEVICE
inline RH_StridePtr CUDA_SYM(RH_StrideArrayAllocOutput)(RandomHash_State* state, U32 initialSize) 
{
    if (state->m_isMidStateRound)
    {
        RHMINER_ASSERT(state->m_stridesAllocIndex + initialSize + 8 < state->m_stridesAllocMidstateBarrier);
    }

    RH_ASSERT((size_t(state->m_stridesAllocIndex) % 32) == 0);
    RH_StridePtr stride = ((U8*)state->m_stridesInstances) + (state->m_stridesAllocIndex);
    RH_ASSERT((size_t(stride) % 32) == 0);

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    if (state->m_strideID)
        RH_ASSERT(*(U32*)((stride - 4)) == 0xBABABABA);
#endif

    state->m_stridesAllocIndex += initialSize + RH_IDEAL_ALIGNMENT;
    RH_ASSERT(state->m_stridesAllocIndex < RH_STRIDE_BANK_SIZE);

    RH_STRIDE_SET_SIZE(stride, initialSize);
    RH_STRIDE_INIT_INTEGRITY(stride);

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    RH_STRIDE_SET_INDEX(stride, state->m_strideID);
    state->m_strideID++;
#endif

    return stride;
}

CUDA_DECL_HOST_AND_DEVICE
inline void CUDA_SYM(RH_StrideArrayGrow)(RandomHash_State* state, RH_StridePtr stride, U32 growSize) 
{
    if (state->m_isMidStateRound)
    {
        RHMINER_ASSERT(state->m_stridesAllocIndex + growSize+8 < state->m_stridesAllocMidstateBarrier);
    }

    state->m_stridesAllocIndex += growSize;
    RH_ASSERT(state->m_stridesAllocIndex < RH_STRIDE_BANK_SIZE);
    
    RH_STRIDE_SET_SIZE(stride, RH_STRIDE_GET_SIZE(stride) + growSize);
    RH_STRIDE_INIT_INTEGRITY(stride);
}

CUDA_DECL_HOST_AND_DEVICE
inline void CUDA_SYM(RH_StrideArrayClose)(RandomHash_State* state, RH_StridePtr stride) 
{
    if (state->m_isMidStateRound)
    {
        RHMINER_ASSERT(RHMINER_ALIGN(state->m_stridesAllocIndex, 32) + 8 < state->m_stridesAllocMidstateBarrier);
    }

    U32 ss = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    ss += sizeof(U64);
#endif

    RH_ASSERT(RHMINER_ALIGN(state->m_stridesAllocIndex + ss, 32) > state->m_stridesAllocIndex);
    state->m_stridesAllocIndex = RHMINER_ALIGN(state->m_stridesAllocIndex + ss, 32);
    RH_ASSERT(state->m_stridesAllocIndex < RH_STRIDE_BANK_SIZE);
    RH_ASSERT((size_t(state->m_stridesAllocIndex) % 32) == 0);

    RH_STRIDE_CHECK_INTEGRITY(stride);
}

CUDA_DECL_HOST_AND_DEVICE
inline void CUDA_SYM(RH_StrideArrayReset)(RH_StridePtrArray strideArrayVar) 
{
    RH_StrideArrayStruct* arr = (RH_StrideArrayStruct*)strideArrayVar;
    arr->size = 0;
    static_assert(sizeof(MurmurHash3_x86_32_State) == 2 * sizeof(U64), "Incorrect struct size");
    U64* accum = (U64*)&arr->accum;
    accum[0] = 0;
    accum[1] = 0;

    static_assert(sizeof(void*) == sizeof(U64), "Incorrect ptr size");
}

#include "MinersLib/Pascal/RandomHash_inl.h"

inline CUDA_DECL_HOST_AND_DEVICE U32 CUDA_SYM(GetNextRnd)(mersenne_twister_state* gen) 
{    
    return _CM(merssen_twister_rand)(gen);
}

//--------------------------------------------------------------------------------------------------
void CUDA_DECL_HOST_AND_DEVICE CUDA_SYM(RandomHash_RoundDataInit)(RH_RoundData* rd, int round)
{
    RH_ASSERT(rd->roundOutputs || (round == 0 && rd->roundOutputs == 0));
    if (GetRoundOutputCount(round))
    {
        RH_STRIDEARRAY_RESET(rd->roundOutputs);
    }

    if (GetParentRoundOutputCount(round))
    {
        if (round != 5)
            RH_STRIDEARRAY_RESET( rd->parenAndNeighbortOutputs );
    }
}

inline void CUDA_DECL_HOST_AND_DEVICE CUDA_SYM(RandomHash_Initialize)(RandomHash_State* state)
{
    CUDA_SYM(RandomHash_RoundDataInit)(&state->m_data[0], 0);
    CUDA_SYM(RandomHash_RoundDataInit)(&state->m_data[1], 1);
    CUDA_SYM(RandomHash_RoundDataInit)(&state->m_data[2], 2);
    CUDA_SYM(RandomHash_RoundDataInit)(&state->m_data[3], 3);
    CUDA_SYM(RandomHash_RoundDataInit)(&state->m_data[4], 4);
    CUDA_SYM(RandomHash_RoundDataInit)(&state->m_data[5], 5);
    

    RH_ASSERT(RH_WorkSize == RH_StrideSize);
    state->m_strideID = 0;
    RH_STRIDEARRAY_RESET(state->m_round5Phase2PrecalcArray);
    
    if (state->m_isCachedOutputs)
    {
        state->m_isMidStateRound = true;

        RH_ASSERT(state->m_stridesAllocIndex);
        if (state->m_stridesAllocMidstateBarrierNext != RH_STRIDE_BANK_SIZE) 
        {
            state->m_stridesAllocIndex = 0;
            state->m_stridesAllocMidstateBarrier = state->m_stridesAllocMidstateBarrierNext;

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
            memset(state->m_stridesInstances + 0, (U8)0xBA, state->m_stridesAllocMidstateBarrierNext); 
            U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
            RHMINER_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif            
        }
        else 
        {
            const U32 ReqDelta = 4096;
            state->m_stridesAllocIndex = RHMINER_ALIGN(state->m_stridesAllocIndex, 4096) + ReqDelta;
            state->m_stridesAllocMidstateBarrierNext = state->m_stridesAllocMidstateBarrier;
            state->m_stridesAllocMidstateBarrier = RH_STRIDE_BANK_SIZE;


#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
            memset(state->m_stridesInstances + state->m_stridesAllocIndex, (U8)0xBA, RH_STRIDE_BANK_SIZE - state->m_stridesAllocIndex);   
            U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
            RHMINER_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif
        }
    }
    else
    {
        state->m_stridesAllocMidstateBarrierNext = RH_STRIDE_BANK_SIZE;
        state->m_stridesAllocMidstateBarrier = 0;
        state->m_stridesAllocIndex = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
        memset(state->m_stridesInstances, (U8)0xBA, RH_STRIDE_BANK_SIZE);
        U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
        RHMINER_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif
    }
}


#define CUDA_DECLARE_STATE() RandomHash_State* state = allStates;

void CUDA_SYM(RandomHash_SetTarget)(uint64_t target)
{
    c_target =  target;
}


void CUDA_SYM(RandomHash_Destroy)(RandomHash_State* state)
{
    _CM(RandomHash_RoundDataUnInit)(&state->m_data[0], 0);
    _CM(RandomHash_RoundDataUnInit)(&state->m_data[1], 1);
    _CM(RandomHash_RoundDataUnInit)(&state->m_data[2], 2);
    _CM(RandomHash_RoundDataUnInit)(&state->m_data[3], 3);
    _CM(RandomHash_RoundDataUnInit)(&state->m_data[4], 4);
    _CM(RandomHash_RoundDataUnInit)(&state->m_data[5], 5);
   
    state->m_isCachedOutputs = false;
    _CM(RandomHash_Free)(state->m_stridesInstances);
}

void CUDA_SYM(RandomHash_DestroyMany)(RandomHash_State* stateArray, U32 count)
{
    if (stateArray)
    {
        for (U32 i = 0; i < count; i++)
            _CM(RandomHash_Destroy)(&stateArray[i]);
        _CM(RandomHash_Free)(stateArray);
    }
} 


void CUDA_SYM(AllocateArray)(U8*& arrayData, int count)
{
    U32 size = sizeof(RH_StrideArrayStruct);
    _CM(RandomHash_Alloc)((void**)&arrayData, size);
    PLATFORM_MEMSET(arrayData, 0, size);

    RH_STRIDEARRAY_GET_MAXSIZE(arrayData) = count;
    RH_STRIDEARRAY_GET_EXTRA(arrayData, memoryboost) = g_memoryBoostLevel;
    RH_STRIDEARRAY_GET_EXTRA(arrayData, sseoptimization) = g_sseOptimization;
    RH_CUDA_ERROR_CHECK();
}


void CUDA_SYM(RandomHash_RoundDataAlloc)(RH_RoundData* rd, int round)
{
    PLATFORM_MEMSET(rd, 0, sizeof(RH_RoundData));

    if (GetRoundOutputCount(round) > 0)
    {
        _CM(AllocateArray)(rd->roundOutputs, (int)GetRoundOutputCount(round));
    }
        
    if (GetParentRoundOutputCount(round))
    {
        _CM(AllocateArray)(rd->parenAndNeighbortOutputs, (int)GetParentRoundOutputCount(round));
    }
}


void CUDA_SYM(RandomHash_Create)(RandomHash_State* state)
{
    U32 ajust = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    ajust = sizeof(U64);
#endif    

    _CM(RandomHash_Alloc)((void**)&state->m_stridesInstances, RH_STRIDE_BANK_SIZE + ajust);
    
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
    *check = 0xFF55AA44BB8800DDLLU;
#endif

    RH_CUDA_ERROR_CHECK();
    state->m_stridesAllocIndex = 0;
    state->m_stridesAllocMidstateBarrierNext = 0;

    _CM(RandomHash_RoundDataAlloc)(&state->m_data[0], 0);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[1], 1);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[2], 2);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[3], 3);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[4], 4);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[5], 5);

    _CM(AllocateArray)(state->m_round5Phase2PrecalcArray, 31);
    RH_STRIDEARRAY_RESET(state->m_round5Phase2PrecalcArray);
    
    
    state->m_isCachedOutputs = false;
    state->m_midStateNonce = 0xFFFFFFFF;
    state->m_isNewHeader = true;
    state->m_isMidStateRound = false;
    state->m_data[1].first_round_consume = false;
    state->m_data[2].first_round_consume = false;
    state->m_data[3].first_round_consume = false;
    state->m_data[4].first_round_consume = false;
    state->m_data[5].first_round_consume = false;

    _CM(RandomHash_Initialize)(state);
}

void CUDA_SYM(RandomHash_CreateMany)(RandomHash_State** outPtr, U32 count)
{
    _CM(RandomHash_Alloc)((void**)outPtr, sizeof(RandomHash_State)*count);
    RH_CUDA_ERROR_CHECK();

    for (U32 i = 0; i < count; i++)
    {
        _CM(RandomHash_Create)(&(*outPtr)[i]);
    }

} 

void CUDA_SYM(RandomHash_RoundDataUnInit)(RH_RoundData* rd, int round)
{
    _CM(RandomHash_Free)(rd->roundOutputs);
}

void CUDA_SYM(RandomHash_SetHeader)(RandomHash_State* state, U8* sourceHeader, U32 nonce2)
{
    U8* targetInput = state->m_header;
    
    state->m_isCachedOutputs = false;
    state->m_isNewHeader = true;
    state->m_isMidStateRound = false;
    state->m_data[1].first_round_consume = false;
    state->m_data[2].first_round_consume = false;
    state->m_data[3].first_round_consume = false;
    state->m_data[4].first_round_consume = false;
    state->m_data[5].first_round_consume = false;

    RHMINER_ASSERT(sourceHeader);
    RHMINER_ASSERT(PascalHeaderSize <= PascalHeaderSize);
    memcpy(targetInput, sourceHeader, PascalHeaderSize);

}


inline void CUDA_SYM_DECL(RandomHash_Reseed)(mersenne_twister_state& rndGen, U32 seed) 
{
    _CM(merssen_twister_seed)(seed, &rndGen);
}


inline U32 CUDA_SYM_DECL(RandomHash_Checksum)(RH_StridePtr input)
{
    U32 csum  = 0;
    csum = _CM(MurmurHash3_x86_32_Fast)(RH_STRIDE_GET_DATA(input), RH_STRIDE_GET_SIZE(input));
    return csum;
}
    
U32 CUDA_SYM_DECL(RandomHash_ChecksumArray)(RH_StridePtrArray inputs)
{
    U32 csum;
    MurmurHash3_x86_32_State state;
    _CM(MurmurHash3_x86_32_Init)(0, &state);

    RH_STRIDEARRAY_FOR_EACH_BEGIN(inputs)
    {
        _CM(MurmurHash3_x86_32_Update)(RH_STRIDE_GET_DATA(strideItrator), RH_STRIDE_GET_SIZE(strideItrator), &state);
    }
    RH_STRIDEARRAY_FOR_EACH_END(inputs)
    csum = _CM(MurmurHash3_x86_32_Finalize)(&state);
	return csum;
}

void CUDA_SYM_DECL(RandomHash_Expand)(RandomHash_State* state, RH_StridePtr input, int round, int ExpansionFactor, U8* strideArray)
{
    U32 inputSize = RH_STRIDE_GET_SIZE(input);
    U32 seed = _CM(RandomHash_Checksum)(input);
    _CM(RandomHash_Reseed)(state->m_rndGenExpand, seed);
    size_t sizeExp = inputSize + ExpansionFactor * RH_M;

    RH_StridePtr output = input;

    S64 bytesToAdd = sizeExp - inputSize;    
    RH_ASSERT((bytesToAdd % 2) == 0);
    RH_ASSERT(RH_STRIDE_GET_SIZE(output) != 0);
    RH_ASSERT(RH_STRIDE_GET_SIZE(output) < RH_StrideSize);

    U8* outputPtr = RH_STRIDE_GET_DATA(output);
    while (bytesToAdd > 0)
    {
        U32 nextChunkSize = RH_STRIDE_GET_SIZE(output);
        U8* nextChunk = outputPtr + nextChunkSize;  
        if (nextChunkSize > bytesToAdd)
        {
            nextChunkSize = (U32)bytesToAdd; 
        }

        _CM(RH_StrideArrayGrow)(state, output, nextChunkSize);

        RH_ASSERT(nextChunk + nextChunkSize < output + RH_StrideSize);
        U32 random = _CM(GetNextRnd)(&state->m_rndGenExpand);
        U32 r = random % 8;
        RH_ASSERT((nextChunkSize & 1) == 0);
        switch(r)
        {
            case 0: 
#if defined(RHMINER_ENABLE_SSE4) && defined(RH_ENABLE_AVX)
                    if (nextChunkSize <= 512 && (g_sseOptimization == 2))
                    {
                        extern void Transfo0_2_AVX(U8* nextChunk, U32 size, U8* source);
                        Transfo0_2_AVX(nextChunk, nextChunkSize, outputPtr);
                    }
                    else
#endif
                    if (nextChunkSize <= 128 && (size_t(nextChunk) & 0x0f) == 0)
                    {
#if defined(RHMINER_ENABLE_SSE4)
    #if defined(RHMINER_COND_SSE4)
                        if (g_isSSE4Supported)
                            _CM(Transfo0_2_128_SSE4)(nextChunk, nextChunkSize,  outputPtr);
                        else if (g_isSSE3Supported)
                            _CM(Transfo0_2_128_SSE3)(nextChunk, nextChunkSize,  outputPtr);
                        else
                            _CM(Transfo0_2)(nextChunk, nextChunkSize,  outputPtr); 
    #else
                        _CM(Transfo0_2_128_SSE4)(nextChunk, nextChunkSize,  outputPtr);
                            
    #endif //defined(RHMINER_COND_SSE4)
#else
                        if (g_isSSE3Supported)
                            _CM(Transfo0_2_128_SSE3)(nextChunk, nextChunkSize,  outputPtr);
                        else
                            _CM(Transfo0_2)(nextChunk, nextChunkSize,  outputPtr); 
#endif
                    }
                    else
                        _CM(Transfo0_2)(nextChunk, nextChunkSize,  outputPtr); 
                    break;
                    
            case 1: _CM(Transfo1_2)(nextChunk, nextChunkSize,  outputPtr); break;
            case 2: _CM(Transfo2_2)(nextChunk, nextChunkSize,  outputPtr); break;
            case 3: _CM(Transfo3_2)(nextChunk, nextChunkSize,  outputPtr); break;
            case 4: 
            {
#if defined(RHMINER_ENABLE_SSE4)
    #if defined(RH_ENABLE_AVX)
                if (g_sseOptimization == 2 && nextChunkSize > 32)
                {
                    extern void Transfo4_2_AVX(U8* nextChunk, U32 size, U8* outputPtr);
                    Transfo4_2_AVX(nextChunk, nextChunkSize, outputPtr);
                }
                else
    #endif //defined(RH_ENABLE_AVX)
#if 0
                if (nextChunkSize <= 128 && (size_t(nextChunk) & 0x0f) && g_isSSE4Supported)
                {
                    _CM(Transfo4_2_128_SSE4)(nextChunk, nextChunkSize, outputPtr);
                }
                else
#endif
#endif //#if defined(RHMINER_ENABLE_SSE4)
                    _CM(Transfo4_2)(nextChunk, nextChunkSize, outputPtr);
            }break;
            case 5: _CM(Transfo5_2)(nextChunk, nextChunkSize,  outputPtr); break; 
            case 6: _CM(Transfo6_2)(nextChunk, nextChunkSize,  outputPtr); break;
            case 7: _CM(Transfo7_2)(nextChunk, nextChunkSize,  outputPtr); break;
        }

        RH_STRIDE_CHECK_INTEGRITY(output);
        RH_ASSERT(RH_STRIDE_GET_SIZE(output) < RH_StrideSize);
        bytesToAdd = bytesToAdd - nextChunkSize;
    }
    _CM(RH_StrideArrayClose)(state, output);
    RH_ASSERT(sizeExp == RH_STRIDE_GET_SIZE(output))
    RH_STRIDE_CHECK_INTEGRITY(output);
}

void CUDA_SYM_DECL(RandomHash_Compress)(RandomHash_State* state, RH_StridePtrArray inputs, RH_StridePtr Result, U32 in_round)
{
    U32 rval;

#ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3
    MurmurHash3_x86_32_State tstate = *RH_StrideArrayStruct_GetAccum(inputs);
    U32 seed = _CM(MurmurHash3_x86_32_Finalize)(&tstate);

#else
    U32 seed = _CM(RandomHash_ChecksumArray)(inputs);
#endif
    _CM(RandomHash_Reseed)(state->m_rndGenCompress, seed);

    RH_STRIDE_SET_SIZE(Result, 100);
    U8* resultPtr = RH_STRIDE_GET_DATA(Result);
    U32 inoutSize = RH_STRIDEARRAY_GET_SIZE(inputs);

    for (size_t i = 0; i < 100; i++)
    {
        RH_StridePtr source = RH_STRIDEARRAY_GET(inputs, _CM(GetNextRnd)(&state->m_rndGenCompress) % inoutSize);
        U32 sourceSize = RH_STRIDE_GET_SIZE(source);

        rval = _CM(GetNextRnd)(&state->m_rndGenCompress);
        resultPtr[i] = RH_STRIDE_GET_DATA(source)[rval % sourceSize];
    }
} 

inline void CUDA_SYM_DECL(RandomHash_MiddlePoint)(RandomHash_State* state)
{
    state->m_midStateNonce = *(U32*)(RH_STRIDE_GET_DATA(state->m_roundInput)+PascalHeaderNoncePosV4(PascalHeaderSize));
    

    if (!state->m_isMidStateRound)
    {
        const U32 ReqDelta = 4096;
        state->m_stridesAllocIndex = RHMINER_ALIGN(state->m_stridesAllocIndex, 4096) + ReqDelta;
        state->m_stridesAllocMidstateBarrierNext = state->m_stridesAllocIndex;
    }

}

inline void CUDA_SYM_DECL(RandomHash_start)(RandomHash_State* state, U32 in_round)
{
    RH_ASSERT(in_round >= 1 && in_round <= RH_N);
    RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_roundInput) <= PascalHeaderSize);
    U32 seed = _CM(RandomHash_Checksum)(state->m_roundInput);
    _CM(RandomHash_Reseed)(state->m_data[in_round].rndGen, seed);
}

inline void CUDA_SYM_DECL(RandomHash_Phase_1_push)(RandomHash_State* state, int in_round)
{
    if (in_round == RH_N)
    {
        if (in_round == 5)
        {
            if (state->m_midStateNonce == *(U32*)(RH_STRIDE_GET_DATA(state->m_roundInput) + PascalHeaderNoncePosV4(PascalHeaderSize)))
            {
                state->m_skipPhase1 = 1;
                return;
            }
        }
    }

    state->m_data[in_round-1].backup_io_results = state->m_data[in_round-1].io_results;
    
    if (in_round == RH_N)
        state->m_data[in_round - 1].io_results = state->m_data[RH_N].parenAndNeighbortOutputs;
    else
        state->m_data[in_round - 1].io_results = state->m_data[in_round].parenAndNeighbortOutputs;
}

inline void CUDA_SYM_DECL(RandomHash_Phase_1_pop)(RandomHash_State* state, int in_round)
{
    RH_StridePtrArray pano;
    bool skipLastUpdate = false;
    if (in_round == RH_N)
    {    
        if (state->m_skipPhase1)
        {
            RH_StridePtrArray testCache = state->m_data[RH_N].parenAndNeighbortOutputs;

            
            state->m_isCachedOutputs = false;
            state->m_skipPhase1 = 0;

            skipLastUpdate = true;
        }
        
        pano = state->m_data[RH_N].parenAndNeighbortOutputs;
    }
    else
    {   
        state->m_data[in_round-1].io_results = state->m_data[in_round-1].backup_io_results;
        
        pano = state->m_data[in_round].parenAndNeighbortOutputs;
    }

#ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3
    U32 seed;

    {

        if (skipLastUpdate)
        {
            _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(pano, RH_STRIDEARRAY_GET_SIZE(pano) - 1); 

            RH_STRIDEARRAY_PUSHBACK_MANY_ALL(state->m_round5Phase2PrecalcArray, pano);
        }
        else
        {
            _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_DUO)(pano, RH_STRIDEARRAY_GET_SIZE(pano) - 1, state->m_round5Phase2PrecalcArray);
        }
    }

    MurmurHash3_x86_32_State tstate = *RH_StrideArrayStruct_GetAccum(pano);
    seed = _CM(MurmurHash3_x86_32_Finalize)(&tstate);


#else
    U32 seed = _CM(RandomHash_ChecksumArray)(state->m_data[in_round].parenAndNeighbortOutputs);
#endif
    _CM(RandomHash_Reseed)(state->m_data[in_round].rndGen, seed);

    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) == 0);
    RH_STRIDEARRAY_PUSHBACK_MANY_ALL(state->m_data[in_round].roundOutputs, pano); 

    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(pano) <= GetParentRoundOutputCount(in_round));
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));
   
    RH_STRIDEARRAY_RESET(pano);
}

inline void CUDA_SYM_DECL(RandomHash_Phase_2_push)(RandomHash_State* state, int in_round)
{
    U32 newNonce = _CM(GetNextRnd)(&state->m_data[in_round].rndGen);
    *(U32*)(RH_STRIDE_GET_DATA(state->m_roundInput)+PascalHeaderNoncePosV4(PascalHeaderSize)) = newNonce; 
    
    state->m_data[in_round-1].backup_io_results = state->m_data[in_round-1].io_results;
    if (in_round == RH_N)
        state->m_data[in_round - 1].io_results = state->m_data[RH_N].parenAndNeighbortOutputs;
    else
        state->m_data[in_round - 1].io_results = state->m_data[in_round].parenAndNeighbortOutputs;
}


void CUDA_SYM_DECL(RandomHash_Phase_2_pop)(RandomHash_State* state, int in_round)         
{
    state->m_data[in_round-1].io_results = state->m_data[in_round-1].backup_io_results;
    RH_StridePtrArray pano;
    if (in_round == RH_N)
    {
        pano = state->m_data[RH_N].parenAndNeighbortOutputs;

        state->m_isCachedOutputs = true;
    }
    else
        pano = state->m_data[in_round].parenAndNeighbortOutputs;

    RH_ASSERT( RH_STRIDE_GET_SIZE(state->m_data[in_round].roundOutputs) + RH_STRIDE_GET_SIZE(pano) < GetRoundOutputCount(in_round) );
    
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) != 0);
    
    if (in_round == 5)
    {
        {
            RH_STRIDEARRAY_PUSHBACK(state->m_round5Phase2PrecalcArray, RH_STRIDEARRAY_GET(pano, RH_STRIDEARRAY_GET_SIZE(pano)-1));

            _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(state->m_round5Phase2PrecalcArray, RH_STRIDEARRAY_GET_SIZE(state->m_round5Phase2PrecalcArray) - 1);
            _CM(RH_STRIDEARRAY_PUSHBACK_MANY_ALL)(state->m_data[in_round].roundOutputs, state->m_round5Phase2PrecalcArray);
        }
    }
    else
    {
        if (!state->m_data[in_round].first_round_consume)
        {
            state->m_data[in_round].first_round_consume = true;


            RH_STRIDEARRAY_PUSHBACK(state->m_round5Phase2PrecalcArray, RH_STRIDEARRAY_GET(pano, RH_STRIDEARRAY_GET_SIZE(pano)-1));
            _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(state->m_round5Phase2PrecalcArray, RH_STRIDEARRAY_GET_SIZE(state->m_round5Phase2PrecalcArray) - 1);
            _CM(RH_STRIDEARRAY_PUSHBACK_MANY_ALL)(state->m_data[in_round].roundOutputs, state->m_round5Phase2PrecalcArray);

        }
        else
        {
            _CM(RH_STRIDEARRAY_PUSHBACK_MANY_UPDATE)(state->m_data[in_round].roundOutputs, pano, state->m_round5Phase2PrecalcArray);
        }
    }

    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(pano) <= GetParentRoundOutputCount(in_round));
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));

    _CM(RandomHash_Compress)(state, state->m_data[in_round].roundOutputs, state->m_workBytes, in_round);  
    RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) <= 100);
        
    if (in_round != RH_N)
    {
        if (in_round == 4 && state->m_isMidStateRound)
        {
            if (state->m_stridesAllocMidstateBarrier != RH_STRIDE_BANK_SIZE)
                state->m_stridesAllocMidstateBarrierNext = RH_STRIDE_BANK_SIZE;
        }
        RH_STRIDEARRAY_RESET(pano);
    }
}

inline void CUDA_SYM_DECL(RandomHash_Phase_init)(RandomHash_State* state, int in_round)
{    
    RH_STRIDEARRAY_RESET(state->m_data[in_round].roundOutputs);
}

inline void CUDA_SYM_DECL(RandomHash)(RandomHash_State* state, int in_round)
{
    RH_StridePtr input;
    if (in_round == 1)
    {
        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_roundInput) <= PascalHeaderSize);
        input = state->m_roundInput;
    }
    else
    {
        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) <= 100);
        input = state->m_workBytes;
    }

    U32 rndHash = _CM(GetNextRnd)(&state->m_data[in_round].rndGen) % 18;

    RH_StridePtr output = _CM(RH_StrideArrayAllocOutput)(state, c_AlgoSize[rndHash]);
    RH_STRIDEARRAY_PUSHBACK(state->m_data[in_round].roundOutputs, output);
    RH_ASSERT( RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));
    
    switch(rndHash)
    {
        case RandomHashAlgos::RH_SHA2_256     :
        {
            _CM(RandomHash_SHA2_256)(input, output);
        } break;
        case RandomHashAlgos::RH_SHA2_384     :
        {
            _CM(RandomHash_SHA2_384)(input, output);
        } break;
        case RandomHashAlgos::RH_SHA2_512     :
        {
            _CM(RandomHash_SHA2_512)(input, output);
        } break;
        case RandomHashAlgos::RH_SHA3_256     :
        {
            _CM(RandomHash_SHA3_256)(input, output);
        } break;
        case RandomHashAlgos::RH_SHA3_384     :
        {
            _CM(RandomHash_SHA3_384)(input, output);
        } break;
        case RandomHashAlgos::RH_SHA3_512     :
        {
            _CM(RandomHash_SHA3_512)(input, output);
        } break;
        case RandomHashAlgos::RH_RIPEMD160    :
        {
            _CM(RandomHash_RIPEMD160)(input, output);
        } break;
        case RandomHashAlgos::RH_RIPEMD256    :
        {
            _CM(RandomHash_RIPEMD256)(input, output);
        } break;
        case RandomHashAlgos::RH_RIPEMD320    :
        {
            _CM(RandomHash_RIPEMD320)(input, output);
        } break;
        case RandomHashAlgos::RH_Blake2b      :
        {
            _CM(RandomHash_blake2b)(input, output);
        } break;
        case RandomHashAlgos::RH_Blake2s      :
        {            
            _CM(RandomHash_blake2s)(input, output);
        } break;
        case RandomHashAlgos::RH_Tiger2_5_192 :
        {
            _CM(RandomHash_Tiger2_5_192)(input, output);
        } break;
        case RandomHashAlgos::RH_Snefru_8_256 :
        {
            _CM(RandomHash_Snefru_8_256)(input, output);
        } break;
        case RandomHashAlgos::RH_Grindahl512  :
        {
            _CM(RandomHash_Grindahl512)(input, output);
        } break;
        case RandomHashAlgos::RH_Haval_5_256  :
        {
            _CM(RandomHash_Haval_5_256)(input, output);
        } break;
        case RandomHashAlgos::RH_MD5          :
        {
            _CM(RandomHash_MD5)(input, output);
        } break;
        case RandomHashAlgos::RH_RadioGatun32 :
        {
            _CM(RandomHash_RadioGatun32)(input, output);
        } break;
        case RandomHashAlgos::RH_Whirlpool    :
        {
            _CM(RandomHash_WhirlPool)(input, output);
        } break;
    }

    RH_STRIDE_CHECK_INTEGRITY(output);
}

inline void CUDA_SYM_DECL(RandomHash_end)(RandomHash_State* state, int in_round)
{    
    RH_StridePtr output = RH_STRIDEARRAY_GET(state->m_data[in_round].roundOutputs, RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) - 1);
    _CM(RandomHash_Expand)(state, output, in_round, RH_N - in_round, state->m_data[in_round].roundOutputs);
    RH_STRIDEARRAY_RESET(state->m_data[in_round].io_results);

    RH_STRIDEARRAY_PUSHBACK_MANY_ALL(state->m_data[in_round].io_results, state->m_data[in_round].roundOutputs); 
    
    if (in_round == 5)
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3_DUO)(state->m_data[5].roundOutputs, RH_STRIDEARRAY_GET_SIZE(state->m_data[5].roundOutputs) - 1, state->m_round5Phase2PrecalcArray);
    
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));
}

inline void CUDA_SYM_DECL(RandomHash_FirstCall_push)(RandomHash_State* state, int in_round)
{
    state->m_data[5].io_results = state->m_data[0].parenAndNeighbortOutputs;
    state->m_skipPhase1 = 0;    
}

//-------------------------------------------------------------------------------------------------------------------------------------
#define RH_CALL_KERNEL_BLOCK(N)    CUDA_SYM(RandomHash_Block##N)(allStates);

#define RH_DEFINE_KERNEL_BLOCK(N)  CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Block##N)(RandomHash_State* allStates) \
                                    { \
                                        CUDA_DECLARE_STATE(); \
                                        RH_B##N \
                                    }

CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Block0)(RandomHash_State* allStates)
{
    CUDA_DECLARE_STATE();
    /*#define RH_B0*/     
    RandomHash_FirstCall_push(state, 5);
    RandomHash_Phase_init(state, 5);
    RandomHash_Phase_1_push(state, 5);
    if (!state->m_skipPhase1) 
    {
        RandomHash_Phase_init(state, 4);
        RandomHash_Phase_1_push(state, 4);
        RandomHash_Phase_init(state, 3);
        RandomHash_Phase_1_push(state, 3);
        RandomHash_Phase_init(state, 2);
        RandomHash_Phase_1_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_1_pop(state, 2);


        RandomHash_Phase_2_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_2_pop(state, 2);

        RandomHash(state, 2);

        RandomHash_end(state, 2);
        RandomHash_Phase_1_pop(state, 3);


        RandomHash_Phase_2_push(state, 3);
        RandomHash_Phase_init(state, 2);
        RandomHash_Phase_1_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_1_pop(state, 2);


        RandomHash_Phase_2_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_2_pop(state, 2);

        RandomHash(state, 2);

        RandomHash_end(state, 2);
        RandomHash_Phase_2_pop(state, 3);

        RandomHash(state, 3);

        RandomHash_end(state, 3);
        RandomHash_Phase_1_pop(state, 4);


        RandomHash_Phase_2_push(state, 4);
        RandomHash_Phase_init(state, 3);
        RandomHash_Phase_1_push(state, 3);
        RandomHash_Phase_init(state, 2);
        RandomHash_Phase_1_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_1_pop(state, 2);


        RandomHash_Phase_2_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_2_pop(state, 2);

        RandomHash(state, 2);

        RandomHash_end(state, 2);
        RandomHash_Phase_1_pop(state, 3);


        RandomHash_Phase_2_push(state, 3);
        RandomHash_Phase_init(state, 2);
        RandomHash_Phase_1_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_1_pop(state, 2);


        RandomHash_Phase_2_push(state, 2);
        RandomHash_Phase_init(state, 1);
        RandomHash_start(state, 1);
        RandomHash(state, 1);

        RandomHash_end(state, 1);
        RandomHash_Phase_2_pop(state, 2);

        RandomHash(state, 2);

        RandomHash_end(state, 2);
        RandomHash_Phase_2_pop(state, 3);

        RandomHash(state, 3);

        RandomHash_end(state, 3);
        RandomHash_Phase_2_pop(state, 4);

        RandomHash(state, 4);

        RandomHash_end(state, 4);
    }
    RandomHash_Phase_1_pop(state, 5);


    RandomHash_Phase_2_push(state, 5);
    RandomHash_Phase_init(state, 4);
    RandomHash_Phase_1_push(state, 4);
    RandomHash_Phase_init(state, 3);
    RandomHash_Phase_1_push(state, 3);
    RandomHash_Phase_init(state, 2);
    RandomHash_Phase_1_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash_MiddlePoint(state);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_1_pop(state, 2);


    RandomHash_Phase_2_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_2_pop(state, 2);

    RandomHash(state, 2);

    RandomHash_end(state, 2);
    RandomHash_Phase_1_pop(state, 3);


    RandomHash_Phase_2_push(state, 3);
    RandomHash_Phase_init(state, 2);
    RandomHash_Phase_1_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_1_pop(state, 2);


    RandomHash_Phase_2_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_2_pop(state, 2);

    RandomHash(state, 2);

    RandomHash_end(state, 2);
    RandomHash_Phase_2_pop(state, 3);

    RandomHash(state, 3);

    RandomHash_end(state, 3);
    RandomHash_Phase_1_pop(state, 4);


    RandomHash_Phase_2_push(state, 4);
    RandomHash_Phase_init(state, 3);
    RandomHash_Phase_1_push(state, 3);
    RandomHash_Phase_init(state, 2);
    RandomHash_Phase_1_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_1_pop(state, 2);


    RandomHash_Phase_2_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_2_pop(state, 2);

    RandomHash(state, 2);

    RandomHash_end(state, 2);
    RandomHash_Phase_1_pop(state, 3);


    RandomHash_Phase_2_push(state, 3);
    RandomHash_Phase_init(state, 2);
    RandomHash_Phase_1_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_1_pop(state, 2);


    RandomHash_Phase_2_push(state, 2);
    RandomHash_Phase_init(state, 1);
    RandomHash_start(state, 1);
    RandomHash(state, 1);

    RandomHash_end(state, 1);
    RandomHash_Phase_2_pop(state, 2);

    RandomHash(state, 2);

    RandomHash_end(state, 2);
    RandomHash_Phase_2_pop(state, 3);

    RandomHash(state, 3);

    RandomHash_end(state, 3);
    RandomHash_Phase_2_pop(state, 4);

    RandomHash(state, 4);

    RandomHash_end(state, 4);
    RandomHash_Phase_2_pop(state, 5);

    RandomHash(state, 5);

    RandomHash_end(state, 5);
}

#define RH_CALL_ALL_KERNEL_BLOCKS \
RH_CALL_KERNEL_BLOCK(0) \

//-------------------------------------------------------------------------------------------------------------------------------------
CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Init)(RandomHash_State* allStates, U8* out_hash, U32 startNonce)
{
    CUDA_DECLARE_STATE();

    _CM(RandomHash_Initialize)(state);
    
    if (state->m_isNewHeader)
    {
        state->m_isNewHeader = false;
        (*(U32*)(state->m_roundInput)) = PascalHeaderSize;
        _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(RH_STRIDE_GET_DATA(state->m_roundInput), &state->m_header[0], PascalHeaderSize); 
    }
    
    if (state->m_isCachedOutputs)
        startNonce = state->m_midStateNonce;

#ifdef RH_SCREEN_SAVER_MODE
    extern void ScreensaverFeed(U32 nonce);
    ScreensaverFeed(startNonce);
#endif
    
    state->m_startNonce = startNonce;
    *(U32*)(RH_STRIDE_GET_DATA(state->m_roundInput) + PascalHeaderNoncePosV4(PascalHeaderSize)) = startNonce;
}

#ifdef __CUDA_ARCH__
//CUDA_DECL_KERNEL 
#endif
void CUDA_SYM(RandomHash_Finalize)(RandomHash_State* allStates, U8* output) 
{
    CUDA_DECLARE_STATE();

    RH_STRIDE_CHECK_INTEGRITY(RH_STRIDEARRAY_GET(state->m_data[5].roundOutputs, 30));

    _CM(RandomHash_Compress)(state, state->m_data[5].roundOutputs, state->m_workBytes, 0);

    RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) <= 100);

    U8 tempStride[RH_IDEAL_ALIGNMENT + 256];
    _CM(RandomHash_SHA2_256)(state->m_workBytes, &tempStride[0]);

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
    RHMINER_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif

    memcpy(output, RH_STRIDE_GET_DATA(tempStride), 32);
}

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

void RandomHash_Search(RandomHash_State* in_state, U8* out_hash, U32 startNonce)
{

    RandomHash_State* allStates = in_state;
    RandomHash_Init(allStates, out_hash, startNonce);
    RH_CALL_ALL_KERNEL_BLOCKS
    RandomHash_Finalize(allStates, out_hash);
}


