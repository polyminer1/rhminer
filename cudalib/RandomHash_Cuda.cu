/**
 * RandomHash CUDA source code implementation
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
#if !defined(RH_COMPILE_CPU_ONLY)

#define RH_ENABLE_MID_ROUND_OPT_CUDA
#define RH_USE_CUDA_MEM_BOOST

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

struct RH_StrideArrayStruct
{
    U32 size;
    U32 maxSize;
    U64 memoryboost;
    U64 supportsse41;
    U64 sseoptimization;
    MurmurHash3_x86_32_State accum;
    U8  dummy2[(RH_IDEAL_ALIGNMENT/2) - sizeof(MurmurHash3_x86_32_State)];
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    U8* strides[RH_StrideArrayCount + 1];
#else
    U8* strides[RH_StrideArrayCount];
#endif
};
#define RH_StrideArrayStruct_GetAccum(strideArray) (&((RH_StrideArrayStruct*)strideArray)->accum)


//--------------------------------------------------------------------------------------------------
const size_t c_round_outputsCounts[6] = {0, 1, 3, 7, 15, 31 };
const size_t c_round_parenAndNeighbortOutputsCounts[6] = { 31, 0, 1, 3, 7, 15 };
__constant__ const U32    c_AlgoSize[] = { 32,48,64,32,48,64,20,32,40,64,32,24,32,64,32,16,32,64};

__constant__ static size_t RH_ALIGN(16) d_round_outputsCounts[6] = {0, 1, 3, 7, 15, 31 };
__constant__ static size_t RH_ALIGN(16) d_round_parenAndNeighbortOutputsCounts[6] = { 31, 0, 1, 3, 7, 15 };
__constant__ __device__ U64 d_target = 0xFFFFFFFFFFFFFFFF;
__constant__ __device__ U32 d_deviceID = 0xFFFFFFFF;
thread_local U32            c_deviceID = 0xFFFFFFFF;

#ifdef __CUDA_ARCH__
    #define GetRoundOutputCount(n)          d_round_outputsCounts[n]
    #define GetParentRoundOutputCount(n)    d_round_parenAndNeighbortOutputsCounts[n]
    #define GetDeviceID()                   d_deviceID
    #define GetTarget()                     d_target
#else
    static thread_local U64                 c_target = 0xFFFFFFFFFFFFFFFF;
    #define GetTarget()                     c_target
    #define GetRoundOutputCount(n)          c_round_outputsCounts[n]
    #define GetParentRoundOutputCount(n)    c_round_parenAndNeighbortOutputsCounts[n]
    #define GetDeviceID()                   c_deviceID
#endif

CUDA_DECL_HOST_AND_DEVICE
inline RH_StridePtr CUDA_SYM(RH_StrideArrayGet)(RH_StridePtrArray strideArrayVar, int idx) 
{
    RH_ASSERT(idx <= (int)RH_STRIDEARRAY_GET_MAXSIZE(strideArrayVar));
    return ((RH_StrideArrayStruct*)strideArrayVar)->strides[idx];
}


CUDA_DECL_DEVICE
inline RH_StridePtr CUDA_SYM(RH_StrideArrayAllocOutput)(RandomHash_State* state, U32 initialSize) 
{
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    if (state->m_isMidStateRound)
    {
        RH_ASSERT(state->m_stridesAllocIndex + initialSize + 8 < state->m_stridesAllocMidstateBarrier);
    }
#endif

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
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    if (state->m_isMidStateRound)
    {
        RH_ASSERT(state->m_stridesAllocIndex + growSize+8 < state->m_stridesAllocMidstateBarrier);
    }
#endif

    state->m_stridesAllocIndex += growSize;
    RH_ASSERT(state->m_stridesAllocIndex < RH_STRIDE_BANK_SIZE);
    
    RH_STRIDE_SET_SIZE(stride, RH_STRIDE_GET_SIZE(stride) + growSize);
    RH_STRIDE_INIT_INTEGRITY(stride);
}

CUDA_DECL_HOST_AND_DEVICE
inline void CUDA_SYM(RH_StrideArrayClose)(RandomHash_State* state, RH_StridePtr stride) 
{
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    if (state->m_isMidStateRound)
    {
        RH_ASSERT(RHMINER_ALIGN(state->m_stridesAllocIndex, 32) + 8 < state->m_stridesAllocMidstateBarrier);
    }
#endif

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
    cuSetMember<U32>(arr, RH_GET_MEMBER_POS(RH_StrideArrayStruct, size), 0);

    static_assert(sizeof(MurmurHash3_x86_32_State) == 2 * sizeof(U64), "Incorrect struct size");
    
    cuSetMember<U64>(arr, RH_GET_MEMBER_POS(RH_StrideArrayStruct, accum), 0);

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

#if !defined(RH_ENABLE_MID_ROUND_OPT_CUDA)
    state->m_isCachedOutputs = false;
    state->m_isMidStateRound = false;
    state->m_skipPhase1 = 0;
#endif
    
    
    
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    
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
            RH_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
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
            RH_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif
        }
    }
    else
#endif 
    {
        state->m_stridesAllocMidstateBarrierNext = RH_STRIDE_BANK_SIZE;
        state->m_stridesAllocMidstateBarrier = 0;
        state->m_stridesAllocIndex = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
        memset(state->m_stridesInstances, (U8)0xBA, RH_STRIDE_BANK_SIZE);
        U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
        RH_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif
    }
}

#define CUDA_DECLARE_STATE() RandomHash_State* state = &allStates[KERNEL_GET_GID()];
thread_local RandomHash_State*  g_threadsData = 0;
thread_local size_t             g_threadsDataSize = 0;

void CUDA_SYM(RandomHash_SetTarget)(uint64_t target)
{
#ifndef __CUDA_ARCH__
    c_target =  target;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_target, &c_target, sizeof(d_target)));
#endif	
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
    _CM(RandomHash_Alloc)((void**)&arrayData, sizeof(RH_StrideArrayStruct));
    PLATFORM_MEMSET(arrayData, 0, sizeof(RH_StrideArrayStruct));
    cuSetMember<U32>(arrayData, RH_GET_MEMBER_POS(RH_StrideArrayStruct, maxSize), count); 
}



void CUDA_SYM(RandomHash_RoundDataAlloc)(RH_RoundData* rd, int round)
{
    PLATFORM_MEMSET(rd, 0, sizeof(RH_RoundData));

    U8* roundArray = 0;
    if (GetRoundOutputCount(round) > 0)
    {
        _CM(AllocateArray)(roundArray, GetRoundOutputCount(round));
        cuSetMember<U8*>(rd, RH_GET_MEMBER_POS(RH_RoundData, roundOutputs), roundArray); 
    }
        
    if (GetParentRoundOutputCount(round))
    {
        roundArray = 0;
        _CM(AllocateArray)(roundArray, GetParentRoundOutputCount(round));
        cuSetMember<U8*>(rd, RH_GET_MEMBER_POS(RH_RoundData, parenAndNeighbortOutputs), roundArray);        
    }
}


void CUDA_SYM(RandomHash_Create)(RandomHash_State* state)
{
    

    U32 ajust = 0;
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    ajust = sizeof(U64);
#endif    

    U8* devPtr = 0;
    _CM(RandomHash_Alloc)((void**)&devPtr, RH_STRIDE_BANK_SIZE + ajust);
    cuSetMember<U8*>(state, RH_GET_MEMBER_POS(RandomHash_State, m_stridesInstances), devPtr);
    
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    
    cuSetMember<U64>(devPtr, RH_STRIDE_BANK_SIZE, 0xFF55AA44BB8800DDLLU);
#endif

    RH_CUDA_ERROR_CHECK();
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_stridesAllocIndex), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_stridesAllocMidstateBarrierNext), 0);

    _CM(RandomHash_RoundDataAlloc)((RH_RoundData *)cuGetMemberPtr(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[0])), 0);
    _CM(RandomHash_RoundDataAlloc)((RH_RoundData *)cuGetMemberPtr(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[1])), 1);
    _CM(RandomHash_RoundDataAlloc)((RH_RoundData *)cuGetMemberPtr(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[2])), 2);
    _CM(RandomHash_RoundDataAlloc)((RH_RoundData *)cuGetMemberPtr(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[3])), 3);
    _CM(RandomHash_RoundDataAlloc)((RH_RoundData *)cuGetMemberPtr(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[4])), 4);
    _CM(RandomHash_RoundDataAlloc)((RH_RoundData *)cuGetMemberPtr(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[5])), 5);

    devPtr = 0;
    _CM(AllocateArray)(devPtr, 31);
    cuSetMember<U8*>(state, RH_GET_MEMBER_POS(RandomHash_State, m_round5Phase2PrecalcArray), devPtr);
    
    
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_isCachedOutputs), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_midStateNonce), 0xFFFFFFFF);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_isNewHeader), 1);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_isMidStateRound), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[1].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[2].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[3].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[4].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[5].first_round_consume), 0);

    RH_CUDA_ERROR_CHECK();
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

thread_local bool tls_isNewHEader = true;
void CUDA_SYM(RandomHash_SetHeader)(RandomHash_State* state, U8* sourceHeader, U32 nonce2)
{
    U8 host_targetInput[PascalHeaderSize];
    U8* targetInput = host_targetInput;
    
    
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_isCachedOutputs), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_isNewHeader), 1);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_isMidStateRound), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[1].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[2].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[3].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[4].first_round_consume), 0);
    cuSetMember<U32>(state, RH_GET_MEMBER_POS(RandomHash_State, m_data[5].first_round_consume), 0);
    
    RH_ASSERT(sourceHeader);
    RH_ASSERT(PascalHeaderSize <= PascalHeaderSize);
    memcpy(targetInput, sourceHeader, PascalHeaderSize);
    
    tls_isNewHEader = true;

    cudaMemcpy(state->m_header, targetInput, PascalHeaderSize, cudaMemcpyHostToDevice);
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
            case 0: _CM(Transfo0_2)(nextChunk, nextChunkSize,  outputPtr);break;
            case 1: _CM(Transfo1_2)(nextChunk, nextChunkSize,  outputPtr);break;
            case 2: _CM(Transfo2_2)(nextChunk, nextChunkSize,  outputPtr);break;
            case 3: _CM(Transfo3_2)(nextChunk, nextChunkSize,  outputPtr);break;
            case 4: _CM(Transfo4_2)(nextChunk, nextChunkSize,  outputPtr);break; 
            case 5: _CM(Transfo5_2)(nextChunk, nextChunkSize,  outputPtr);break;
            case 6: _CM(Transfo6_2)(nextChunk, nextChunkSize,  outputPtr);break;
            case 7: _CM(Transfo7_2)(nextChunk, nextChunkSize,  outputPtr);break;
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

        rval = _CM(GetNextRnd)(&state->m_rndGenCompress );
        resultPtr[i] = RH_STRIDE_GET_DATA(source)[rval % sourceSize];
    }
} 
        
inline void CUDA_SYM_DECL(RandomHash_MiddlePoint)(RandomHash_State* state)
{
    state->m_midStateNonce = *(U32*)(RH_STRIDE_GET_DATA(state->m_roundInput)+PascalHeaderNoncePosV4(PascalHeaderSize));
    

#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    
    if (!state->m_isMidStateRound)
    {
        
        
        const U32 ReqDelta = 4096;
        state->m_stridesAllocIndex = RHMINER_ALIGN(state->m_stridesAllocIndex, 4096) + ReqDelta;
        state->m_stridesAllocMidstateBarrierNext = state->m_stridesAllocIndex;
    }
#endif

} 

inline void CUDA_SYM_DECL(RandomHash_start)(RandomHash_State* state, U32 in_round)
{
    RH_ASSERT(in_round >= 1 && in_round <= RH_N);
    RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_roundInput) <= PascalHeaderSize);
    U32 seed = _CM(RandomHash_Checksum)(state->m_roundInput);
    _CM(RandomHash_Reseed)(state->m_data[in_round].rndGen, seed );
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
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
        if (state->m_skipPhase1)
        {
            
            state->m_isCachedOutputs = false;
            state->m_skipPhase1 = 0;

            skipLastUpdate = true;
        }
#endif  

        
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

            
            _CM(RH_STRIDEARRAY_PUSHBACK_MANY_ALL)(state->m_round5Phase2PrecalcArray, pano);
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
    _CM(RH_STRIDEARRAY_PUSHBACK_MANY_ALL)(state->m_data[in_round].roundOutputs, pano); 
    
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
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
        if (in_round == 4 && state->m_isMidStateRound)
        {
            
            if (state->m_stridesAllocMidstateBarrier != RH_STRIDE_BANK_SIZE)
                state->m_stridesAllocMidstateBarrierNext = RH_STRIDE_BANK_SIZE;
        }
#endif

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

    _CM(RH_STRIDEARRAY_PUSHBACK_MANY_ALL)(state->m_data[in_round].io_results, state->m_data[in_round].roundOutputs); 

    
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

#define RH_CALL_KERNEL_BLOCK(N) CUDA_SYM(RandomHash_Block##N)<<<threadsPerBlock, blocks>>>(allStates);

#define RH_SKIP_PHASE1_TEST 

#define RH_DEFINE_KERNEL_BLOCK(N)  CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Block##N)(RandomHash_State* allStates) \
                                    { \
                                        CUDA_DECLARE_STATE(); \
                                        RH_B##N \
                                    }


#define RH_B0         cuda_RandomHash_FirstCall_push(state, 5); \
                      cuda_RandomHash_Phase_init(state, 5);     \
                      cuda_RandomHash_Phase_1_push(state, 5);   

#define RH_B1         cuda_RandomHash_Phase_init(state, 4);     \
                      cuda_RandomHash_Phase_1_push(state, 4);   \
                      cuda_RandomHash_Phase_init(state, 3);     \
                      cuda_RandomHash_Phase_1_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B2                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B3         cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B4                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B5         cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B6                                                           \
                      cuda_RandomHash_end(state, 2);            

#define RH_B7                       cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B8                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B9         cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B10                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B11        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B12                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B13                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B14                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B15                      cuda_RandomHash_Phase_1_pop(state, 4);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 4);   \
                      cuda_RandomHash_Phase_init(state, 3);     \
                      cuda_RandomHash_Phase_1_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B16                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B17        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B18                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B19        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B20                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B21                      cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B22                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B23        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B24                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B25        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B26                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B27                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B28                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B29                      cuda_RandomHash_Phase_2_pop(state, 4);    \
                                                                        \
                      cuda_RandomHash(state, 4);                

#define RH_B30                                                          \
                      cuda_RandomHash_end(state, 4);            


#define RH_B31                      cuda_RandomHash_Phase_1_pop(state, 5);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 5);   \
                      cuda_RandomHash_Phase_init(state, 4);     \
                      cuda_RandomHash_Phase_1_push(state, 4);   \
                      cuda_RandomHash_Phase_init(state, 3);     \
                      cuda_RandomHash_Phase_1_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash_MiddlePoint(state);     \
                      cuda_RandomHash(state, 1);                

#define RH_B32                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B33        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B34                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B35        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B36                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B37                      cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B38                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B39        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B40                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B41        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B42                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B43                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B44                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B45                      cuda_RandomHash_Phase_1_pop(state, 4);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 4);   \
                      cuda_RandomHash_Phase_init(state, 3);     \
                      cuda_RandomHash_Phase_1_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B46                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B47        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B48                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B49        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B50                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B51                      cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B52                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B53        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B54                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B55        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B56                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B57                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B58                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B59                      cuda_RandomHash_Phase_2_pop(state, 4);    \
                                                                        \
                      cuda_RandomHash(state, 4);                

#define RH_B60                                                          \
                      cuda_RandomHash_end(state, 4);            

#define RH_B61                      cuda_RandomHash_Phase_2_pop(state, 5);    \
                                                                        \
                      cuda_RandomHash(state, 5);                

#define RH_B62                                                          \
                      cuda_RandomHash_end(state, 5);            




RH_DEFINE_KERNEL_BLOCK(0)
RH_DEFINE_KERNEL_BLOCK(1)
RH_DEFINE_KERNEL_BLOCK(2)
RH_DEFINE_KERNEL_BLOCK(3)
RH_DEFINE_KERNEL_BLOCK(4)
RH_DEFINE_KERNEL_BLOCK(5)
RH_DEFINE_KERNEL_BLOCK(6)
RH_DEFINE_KERNEL_BLOCK(7)
RH_DEFINE_KERNEL_BLOCK(8)
RH_DEFINE_KERNEL_BLOCK(9)
RH_DEFINE_KERNEL_BLOCK(10)
RH_DEFINE_KERNEL_BLOCK(11)
RH_DEFINE_KERNEL_BLOCK(12)
RH_DEFINE_KERNEL_BLOCK(13)
RH_DEFINE_KERNEL_BLOCK(14)
RH_DEFINE_KERNEL_BLOCK(15)
RH_DEFINE_KERNEL_BLOCK(16)
RH_DEFINE_KERNEL_BLOCK(17)
RH_DEFINE_KERNEL_BLOCK(18)
RH_DEFINE_KERNEL_BLOCK(19)
RH_DEFINE_KERNEL_BLOCK(20)
RH_DEFINE_KERNEL_BLOCK(21)
RH_DEFINE_KERNEL_BLOCK(22)
RH_DEFINE_KERNEL_BLOCK(23)
RH_DEFINE_KERNEL_BLOCK(24)
RH_DEFINE_KERNEL_BLOCK(25)
RH_DEFINE_KERNEL_BLOCK(26)
RH_DEFINE_KERNEL_BLOCK(27)
RH_DEFINE_KERNEL_BLOCK(28)
RH_DEFINE_KERNEL_BLOCK(29)
RH_DEFINE_KERNEL_BLOCK(30)
RH_DEFINE_KERNEL_BLOCK(31)
RH_DEFINE_KERNEL_BLOCK(32)
RH_DEFINE_KERNEL_BLOCK(33)
RH_DEFINE_KERNEL_BLOCK(34)
RH_DEFINE_KERNEL_BLOCK(35)
RH_DEFINE_KERNEL_BLOCK(36)
RH_DEFINE_KERNEL_BLOCK(37)
RH_DEFINE_KERNEL_BLOCK(38)
RH_DEFINE_KERNEL_BLOCK(39)
RH_DEFINE_KERNEL_BLOCK(40)
RH_DEFINE_KERNEL_BLOCK(41)
RH_DEFINE_KERNEL_BLOCK(42)
RH_DEFINE_KERNEL_BLOCK(43)
RH_DEFINE_KERNEL_BLOCK(44)
RH_DEFINE_KERNEL_BLOCK(45)
RH_DEFINE_KERNEL_BLOCK(46)
RH_DEFINE_KERNEL_BLOCK(47)
RH_DEFINE_KERNEL_BLOCK(48)
RH_DEFINE_KERNEL_BLOCK(49)
RH_DEFINE_KERNEL_BLOCK(50)
RH_DEFINE_KERNEL_BLOCK(51)
RH_DEFINE_KERNEL_BLOCK(52)
RH_DEFINE_KERNEL_BLOCK(53)
RH_DEFINE_KERNEL_BLOCK(54)
RH_DEFINE_KERNEL_BLOCK(55)
RH_DEFINE_KERNEL_BLOCK(56)
RH_DEFINE_KERNEL_BLOCK(57)
RH_DEFINE_KERNEL_BLOCK(58)
RH_DEFINE_KERNEL_BLOCK(59)
RH_DEFINE_KERNEL_BLOCK(60)
RH_DEFINE_KERNEL_BLOCK(61)
RH_DEFINE_KERNEL_BLOCK(62)


#define RH_CALL_ALL_KERNEL_BLOCKS \
RH_CALL_KERNEL_BLOCK(0) \
RH_CALL_KERNEL_BLOCK(1) \
RH_CALL_KERNEL_BLOCK(2) \
RH_CALL_KERNEL_BLOCK(3) \
RH_CALL_KERNEL_BLOCK(4) \
RH_CALL_KERNEL_BLOCK(5) \
RH_CALL_KERNEL_BLOCK(6) \
RH_CALL_KERNEL_BLOCK(7) \
RH_CALL_KERNEL_BLOCK(8) \
RH_CALL_KERNEL_BLOCK(9) \
RH_CALL_KERNEL_BLOCK(10) \
RH_CALL_KERNEL_BLOCK(11) \
RH_CALL_KERNEL_BLOCK(12) \
RH_CALL_KERNEL_BLOCK(13) \
RH_CALL_KERNEL_BLOCK(14) \
RH_CALL_KERNEL_BLOCK(15) \
RH_CALL_KERNEL_BLOCK(16) \
RH_CALL_KERNEL_BLOCK(17) \
RH_CALL_KERNEL_BLOCK(18) \
RH_CALL_KERNEL_BLOCK(19) \
RH_CALL_KERNEL_BLOCK(20) \
RH_CALL_KERNEL_BLOCK(21) \
RH_CALL_KERNEL_BLOCK(22) \
RH_CALL_KERNEL_BLOCK(23) \
RH_CALL_KERNEL_BLOCK(24) \
RH_CALL_KERNEL_BLOCK(25) \
RH_CALL_KERNEL_BLOCK(26) \
RH_CALL_KERNEL_BLOCK(27) \
RH_CALL_KERNEL_BLOCK(28) \
RH_CALL_KERNEL_BLOCK(29) \
RH_CALL_KERNEL_BLOCK(30) \
RH_CALL_KERNEL_BLOCK(31) \
RH_CALL_KERNEL_BLOCK(32) \
RH_CALL_KERNEL_BLOCK(33) \
RH_CALL_KERNEL_BLOCK(34) \
RH_CALL_KERNEL_BLOCK(35) \
RH_CALL_KERNEL_BLOCK(36) \
RH_CALL_KERNEL_BLOCK(37) \
RH_CALL_KERNEL_BLOCK(38) \
RH_CALL_KERNEL_BLOCK(39) \
RH_CALL_KERNEL_BLOCK(40) \
RH_CALL_KERNEL_BLOCK(41) \
RH_CALL_KERNEL_BLOCK(42) \
RH_CALL_KERNEL_BLOCK(43) \
RH_CALL_KERNEL_BLOCK(44) \
RH_CALL_KERNEL_BLOCK(45) \
RH_CALL_KERNEL_BLOCK(46) \
RH_CALL_KERNEL_BLOCK(47) \
RH_CALL_KERNEL_BLOCK(48) \
RH_CALL_KERNEL_BLOCK(49) \
RH_CALL_KERNEL_BLOCK(50) \
RH_CALL_KERNEL_BLOCK(51) \
RH_CALL_KERNEL_BLOCK(52) \
RH_CALL_KERNEL_BLOCK(53) \
RH_CALL_KERNEL_BLOCK(54) \
RH_CALL_KERNEL_BLOCK(55) \
RH_CALL_KERNEL_BLOCK(56) \
RH_CALL_KERNEL_BLOCK(57) \
RH_CALL_KERNEL_BLOCK(58) \
RH_CALL_KERNEL_BLOCK(59) \
RH_CALL_KERNEL_BLOCK(60) \
RH_CALL_KERNEL_BLOCK(61) \
RH_CALL_KERNEL_BLOCK(62) \

#define RH_CALL_ALL_KERNEL_BLOCKS_SKIP \
RH_CALL_KERNEL_BLOCK(0) \
RH_CALL_KERNEL_BLOCK(31) \
RH_CALL_KERNEL_BLOCK(32) \
RH_CALL_KERNEL_BLOCK(33) \
RH_CALL_KERNEL_BLOCK(34) \
RH_CALL_KERNEL_BLOCK(35) \
RH_CALL_KERNEL_BLOCK(36) \
RH_CALL_KERNEL_BLOCK(37) \
RH_CALL_KERNEL_BLOCK(38) \
RH_CALL_KERNEL_BLOCK(39) \
RH_CALL_KERNEL_BLOCK(40) \
RH_CALL_KERNEL_BLOCK(41) \
RH_CALL_KERNEL_BLOCK(42) \
RH_CALL_KERNEL_BLOCK(43) \
RH_CALL_KERNEL_BLOCK(44) \
RH_CALL_KERNEL_BLOCK(45) \
RH_CALL_KERNEL_BLOCK(46) \
RH_CALL_KERNEL_BLOCK(47) \
RH_CALL_KERNEL_BLOCK(48) \
RH_CALL_KERNEL_BLOCK(49) \
RH_CALL_KERNEL_BLOCK(50) \
RH_CALL_KERNEL_BLOCK(51) \
RH_CALL_KERNEL_BLOCK(52) \
RH_CALL_KERNEL_BLOCK(53) \
RH_CALL_KERNEL_BLOCK(54) \
RH_CALL_KERNEL_BLOCK(55) \
RH_CALL_KERNEL_BLOCK(56) \
RH_CALL_KERNEL_BLOCK(57) \
RH_CALL_KERNEL_BLOCK(58) \
RH_CALL_KERNEL_BLOCK(59) \
RH_CALL_KERNEL_BLOCK(60) \
RH_CALL_KERNEL_BLOCK(61) \
RH_CALL_KERNEL_BLOCK(62) \




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
    
#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    if (state->m_isCachedOutputs)
        startNonce = state->m_midStateNonce;
    else
#endif 
        startNonce += KERNEL_GET_GID();
    
    state->m_startNonce = startNonce;
    *(U32*)(RH_STRIDE_GET_DATA(state->m_roundInput) + PascalHeaderNoncePosV4(PascalHeaderSize)) = startNonce;
}

CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Finalize)(RandomHash_State* allStates, U8* output) 
{
    CUDA_DECLARE_STATE();

    RH_STRIDE_CHECK_INTEGRITY(RH_STRIDEARRAY_GET(state->m_data[5].roundOutputs, 30));

    _CM(RandomHash_Compress)(state, state->m_data[5].roundOutputs, state->m_workBytes, 0);

    RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_workBytes) <= 100);

    U8 tempStride[RH_IDEAL_ALIGNMENT + 256];
    _CM(RandomHash_SHA2_256)(state->m_workBytes, &tempStride[0]);

#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    U64* check = (U64*)(state->m_stridesInstances + RH_STRIDE_BANK_SIZE);
    RH_ASSERT(*check == 0xFF55AA44BB8800DDLLU);
#endif

#ifdef __CUDA_ARCH__
    U64 stateHigh64 = cuda_swab64(*(U64*)RH_STRIDE_GET_DATA(tempStride));

    if (stateHigh64 <= GetTarget())
    {
	    uint32_t index = atomicInc((U32*)output, 0xffffffff) + 1;
	    if (index > CUDA_SEARCH_RESULT_BUFFER_SIZE -1) 
            return;
        ((U32*)output)[index] = state->m_startNonce;
    }
  #ifdef RHMINER_DEBUG_DEV
    if (KERNEL_GET_GID() == 0)
    {
        KERNEL_LOG("Output : ");
        _CM(RandomHash_PrintHex)((void*)RH_STRIDE_GET_DATA(tempStride), 32, false, true);
    }
    memcpy(state->m_workBytes, RH_STRIDE_GET_DATA(tempStride), 32);
  #endif

#else
    memcpy(output, RH_STRIDE_GET_DATA(tempStride), 32);
#endif
}

void CUDA_SYM(RandomHash_Free)(void* ptr)
{
    if (ptr)
        cudaFree(&ptr);
}

thread_local size_t g_totAlloc = 0;
void CUDA_SYM(RandomHash_Alloc)(void** outPtr, size_t size)
{
    g_totAlloc += size;
    cudaError_t err = cudaMalloc(outPtr, size); 
    if (err != cudaSuccess)
    {
        size_t f, t;
        cudaMemGetInfo (&f, &t);
        KERNEL_LOG("Allocation error '%s' on GPU%d while allocating memory. \n", cudaGetErrorString(err), GetDeviceID());
        if (f < RHMINER_MB(100))
            KERNEL_LOG("Hint : Try a lower thread count.\n");
        else
            KERNEL_LOG("Hint : Try raising virtual memory size\n");

        exit(-1);
    }
}


__host__ void cuda_randomhash_create(uint32_t blocks, uint32_t threadsPerBlock, uint32_t* input, U32 deviceID)
{
    if (!g_threadsData)
    {
        c_deviceID = deviceID;
        cudaMemcpyToSymbol(d_deviceID, &c_deviceID, sizeof(d_deviceID));

        g_threadsDataSize = threadsPerBlock * blocks;
        _CM(RandomHash_CreateMany)(&g_threadsData, threadsPerBlock*blocks);
        
        size_t total_byte = 0;
        size_t free_byte = 0;
        auto err = cudaMemGetInfo(&free_byte, &total_byte );
        PrintOut("CUDA: Using %.2f mb (%.2f mb per threads). Free memory %.2f mb. \n", 
            g_totAlloc/1024/1024.0f, 
            (g_totAlloc/1024/1024.0f)/(float)g_threadsDataSize, 
            err == cudaSuccess ? (free_byte/1024.0f/1024.0f):-1.0f);
    }
}


#ifndef RHMINER_DEBUG_RANDOMHASH_UNITTEST_CUDA
__host__ void cuda_randomhash_init(uint32_t* input, U32 nonce2)
{
    
    input[PascalHeaderNoncePosV4(PascalHeaderSize) / 4] = 0;

    for(int i=0; i < g_threadsDataSize; i++)
    {
        CUDA_SYM(RandomHash_SetHeader)(&g_threadsData[i], (U8*)input, nonce2);
    }
}


__host__ void cuda_randomhash_search(uint32_t blocks, uint32_t threadsPerBlock, cudaStream_t stream, uint32_t* input, uint32_t* output, U32 startNonce)
{  
    RH_ASSERT((threadsPerBlock*blocks == g_threadsDataSize));    
    RandomHash_State* allStates = g_threadsData;

    CUDA_SYM(RandomHash_Init)<<<threadsPerBlock, blocks>>>(allStates, (uint8_t*)output, startNonce);

#ifdef RH_ENABLE_MID_ROUND_OPT_CUDA
    if (!tls_isNewHEader)
    {
        RH_CALL_ALL_KERNEL_BLOCKS_SKIP
    }
    else
#endif
    {
        RH_CALL_ALL_KERNEL_BLOCKS
        tls_isNewHEader = false;
    }
    CUDA_SYM(RandomHash_Finalize)<<<threadsPerBlock, blocks>>>(allStates, (uint8_t*)output);

    
}
#endif
#else

#ifdef _DEBUG
__host__ void cuda_randomhash_create(uint32_t blocks, uint32_t threadsPerBlock, uint32_t* input, U32 deviceID) {}
void cuda_RandomHash_SetTarget(uint64_t target){}
__host__ void cuda_randomhash_init(uint32_t* input, U32 nonce2){}
__host__ void cuda_randomhash_search(uint32_t blocks, uint32_t threadsPerBlock, cudaStream_t stream, uint32_t* input, uint32_t* output, U32 startNonce){}
#endif


#endif //RH_COMPILE_CPU_ONLY

