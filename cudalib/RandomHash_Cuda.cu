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
#ifndef RH_COMPILE_CPU_ONLY

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
    U8  dummy[(RH_IDEAL_ALIGNMENT/2) - 8];
    MurmurHash3_x86_32_State accum;
    U8  dummy2[(RH_IDEAL_ALIGNMENT/2) - sizeof(MurmurHash3_x86_32_State)];
    U8* strides[RH_StrideArrayCount];
};
#define RH_StrideArrayStruct_GetAccum(strideArray) (&((RH_StrideArrayStruct*)strideArray)->accum)

//--------------------------------------------------------------------------------------------------
const size_t c_round_outputsCounts[6] = {0, 1, 3, 7, 15, 31 };
const size_t c_round_parenAndNeighbortOutputsCounts[6] = { 31, 0, 1, 3, 7, 15 };

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

    for (int i = 0; i < RH_StrideArrayCount; i++)
        arr->strides[i] = 0;    
}

CUDA_DECL_HOST_AND_DEVICE
inline RH_StridePtr CUDA_SYM(RH_StrideArrayAllocOutput)(RandomHash_State* state) 
{
    RH_StridePtr stride = ((U8*)state->m_stridesInstances) + state->m_stridesAllocIndex*RH_StrideSize;
    state->m_stridesAllocIndex++;
    RH_ASSERT(state->m_stridesAllocIndex < RH_TOTAL_STRIDES_INSTANCES);
    RH_ASSERT( (((size_t)stride) % 32) == 0);
    _CM(RH_StrideArrayReset(stride));
    return stride;
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
        RH_STRIDEARRAY_RESET(rd->parenAndNeighbortOutputs);
    }

    if (rd->otherNonceHeader && round > 0)
    { 
        RH_STRIDE_INIT(rd->otherNonceHeader);
    }

    RH_ASSERT(rd->roundInput);
    RH_STRIDE_INIT(rd->roundInput);
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
    state->m_stridesAllocIndex = 0;
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

    U8* oldArray = arrayData;
    RH_StrideArrayStruct hostArray;
    memset(&hostArray, 0, sizeof(hostArray));
    arrayData = (U8*)&hostArray;

    RH_STRIDEARRAY_GET_MAXSIZE(arrayData) = count;
    RH_STRIDEARRAY_INIT(arrayData);
    RH_CUDA_ERROR_CHECK();

    arrayData = oldArray;
    cudaMemcpy(arrayData, &hostArray, sizeof(RH_StrideArrayStruct), cudaMemcpyHostToDevice);
}



void CUDA_SYM(RandomHash_RoundDataAlloc)(RH_RoundData* rd, int round)
{
    memset(rd, 0, sizeof(RH_RoundData));

    if (GetRoundOutputCount(round) > 0)
    {
        _CM(AllocateArray)(rd->roundOutputs, GetRoundOutputCount(round));
    }
        
    if (GetParentRoundOutputCount(round))
    {
        _CM(AllocateArray)(rd->parenAndNeighbortOutputs, GetParentRoundOutputCount(round));
    }
    
    _CM(RandomHash_Alloc)((void**)&rd->roundInput, RH_StrideSize);
    RH_CUDA_ERROR_CHECK();

    if (round > 0) 
    {
        _CM(RandomHash_Alloc)((void**)&rd->otherNonceHeader, RH_StrideSize);
        RH_CUDA_ERROR_CHECK();
    }    
}


void CUDA_SYM(RandomHash_Create)(RandomHash_State* state)
{
    RandomHash_State hostState;
    RandomHash_State* deviceState = state;
    state = &hostState;

    _CM(RandomHash_Alloc)((void**)&state->m_stridesInstances, RH_StrideSize * RH_TOTAL_STRIDES_INSTANCES);
    RH_CUDA_ERROR_CHECK();
    state->m_stridesAllocIndex = 0;

    _CM(RandomHash_RoundDataAlloc)(&state->m_data[0], 0);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[1], 1);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[2], 2);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[3], 3);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[4], 4);
    _CM(RandomHash_RoundDataAlloc)(&state->m_data[5], 5);
    
    _CM(RandomHash_Alloc)((void**)&state->m_cachedHheader, PascalHeaderSize);
    PLATFORM_MEMSET(state->m_cachedHheader, 0, PascalHeaderSize);

    _CM(AllocateArray)(state->m_cachedOutputs, GetRoundOutputCount(5));

    cudaMemcpy(deviceState, state, sizeof(RandomHash_State), cudaMemcpyHostToDevice);
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
    _CM(RandomHash_Free)(rd->parenAndNeighbortOutputs);
    _CM(RandomHash_Free)(rd->otherNonceHeader);
    _CM(RandomHash_Free)(rd->roundInput);
}

void CUDA_SYM(RandomHash_SetHeader)(RandomHash_State* state, U8* sourceHeader, U32 nonce2)
{
    U8 host_targetInput[PascalHeaderSize];
    U8* targetInput = host_targetInput;
    
    RHMINER_ASSERT(sourceHeader);
    memcpy(targetInput, sourceHeader, PascalHeaderSize);

    cudaMemcpy(state->m_header, targetInput, PascalHeaderSize, cudaMemcpyHostToDevice);
}


inline void CUDA_SYM_DECL(RandomHash_Reseed)(mersenne_twister_state& rndGen, U32 seed) 
{
    _CM(merssen_twister_seed)(seed, &rndGen);
}


inline U32 CUDA_SYM_DECL(RandomHash_Checksum)(RH_StridePtr input)
{
    U32 csum  = 0;
    csum = _CM(MurmurHash3_x86_32_Fast)((const void *)RH_STRIDE_GET_DATA(input), RH_STRIDE_GET_SIZE(input), 0);
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

void CUDA_SYM_DECL(RandomHash_Expand)(RandomHash_State* state, RH_StridePtr input, int round, int ExpansionFactor/*, RH_StridePtr Result*/)
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

        RH_ASSERT(nextChunk + nextChunkSize < output + RH_StrideSize);
        U32 random = _CM(GetNextRnd)(&state->m_rndGenExpand);
        U8* workBytes = state->m_workBytes;
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

        RH_STRIDE_GET_SIZE(output) += nextChunkSize;
        RH_ASSERT(RH_STRIDE_GET_SIZE(output) < RH_StrideSize);

        bytesToAdd = bytesToAdd - nextChunkSize;
    }

    RH_ASSERT(sizeExp == RH_STRIDE_GET_SIZE(output))
    RH_STRIDE_CHECK_INTEGRITY(output);
}

void CUDA_SYM_DECL(RandomHash_Compress)(RandomHash_State* state, RH_StridePtrArray inputs, RH_StridePtr Result)
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
        
    RH_STRIDE_CHECK_INTEGRITY(Result);
} 

inline void CUDA_SYM_DECL(RandomHash_start)(RandomHash_State* state, U32 in_round)
{
    RH_ASSERT(in_round >= 1 && in_round <= RH_N);
    U32 seed = _CM(RandomHash_Checksum)(state->m_data[in_round].in_blockHeader);      
    _CM(RandomHash_Reseed)(state->m_data[in_round].rndGen, seed );

    RH_STRIDE_COPY(state->m_data[in_round].roundInput, state->m_data[in_round].in_blockHeader);
}

inline void CUDA_SYM_DECL(RandomHash_Phase_1_push)(RandomHash_State* state, int in_round)
{
    state->m_data[in_round-1].backup_in_blockHeader = state->m_data[in_round-1].in_blockHeader;
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round].in_blockHeader;
    state->m_data[in_round-1].backup_io_results = state->m_data[in_round-1].io_results;
    state->m_data[in_round-1].io_results = state->m_data[in_round].parenAndNeighbortOutputs;
}

inline void CUDA_SYM_DECL(RandomHash_Phase_1_pop)(RandomHash_State* state, int in_round)
{
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round-1].backup_in_blockHeader;
    state->m_data[in_round-1].io_results = state->m_data[in_round-1].backup_io_results;

#ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3
    MurmurHash3_x86_32_State tstate = *RH_StrideArrayStruct_GetAccum(state->m_data[in_round].parenAndNeighbortOutputs);
    U32 seed = _CM(MurmurHash3_x86_32_Finalize)(&tstate);

#else
    U32 seed = _CM(RandomHash_ChecksumArray)(state->m_data[in_round].parenAndNeighbortOutputs);
#endif
    _CM(RandomHash_Reseed)(state->m_data[in_round].rndGen, seed);

    _CM(RH_STRIDEARRAY_PUSHBACK_MANY)(state->m_data[in_round].roundOutputs, state->m_data[in_round].parenAndNeighbortOutputs);
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].parenAndNeighbortOutputs) <= GetParentRoundOutputCount(in_round));
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));
   
    RH_STRIDEARRAY_RESET(state->m_data[in_round].parenAndNeighbortOutputs);
       
}

inline void CUDA_SYM_DECL(RandomHash_Phase_2_push)(RandomHash_State* state, int in_round)
{
    RH_STRIDE_COPY(state->m_data[in_round].otherNonceHeader, state->m_data[in_round].in_blockHeader);
    U32 newNonce = _CM(GetNextRnd)(&state->m_data[in_round].rndGen);

    *(U32*)(RH_STRIDE_GET_DATA(state->m_data[in_round].otherNonceHeader)+PascalHeaderNoncePosV4(PascalHeaderSize)) = newNonce;
    
    state->m_data[in_round-1].backup_in_blockHeader = state->m_data[in_round-1].in_blockHeader;
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round].otherNonceHeader;
    state->m_data[in_round-1].backup_io_results = state->m_data[in_round-1].io_results;
    state->m_data[in_round-1].io_results = state->m_data[in_round].parenAndNeighbortOutputs;
}


void CUDA_SYM_DECL(RandomHash_Phase_2_pop)(RandomHash_State* state, int in_round)         
{
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round-1].backup_in_blockHeader;
    state->m_data[in_round-1].io_results = state->m_data[in_round-1].backup_io_results;

    RH_ASSERT( RH_STRIDE_GET_SIZE(state->m_data[in_round].roundOutputs) + RH_STRIDE_GET_SIZE(state->m_data[in_round].parenAndNeighbortOutputs) < GetRoundOutputCount(in_round) );
    
    _CM(RH_STRIDEARRAY_PUSHBACK_MANY)(state->m_data[in_round].roundOutputs, state->m_data[in_round].parenAndNeighbortOutputs);
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].parenAndNeighbortOutputs) <= GetParentRoundOutputCount(in_round));
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));

    _CM(RandomHash_Compress)(state, state->m_data[in_round].roundOutputs, state->m_data[in_round].roundInput);

    RH_STRIDEARRAY_RESET(state->m_data[in_round].parenAndNeighbortOutputs);
}

inline void CUDA_SYM_DECL(RandomHash_Phase_init)(RandomHash_State* state, int in_round)
{
    RH_STRIDEARRAY_RESET(state->m_data[in_round].roundOutputs);
    RH_STRIDE_RESET(state->m_data[in_round].roundInput)
}


inline void CUDA_SYM_DECL(RandomHash)(RandomHash_State* state, int in_round)
{
    RH_StridePtr output = _CM(RH_StrideArrayAllocOutput)(state);
    RH_STRIDEARRAY_PUSHBACK(state->m_data[in_round].roundOutputs, output);
    RH_ASSERT( RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));
    
    U32 rndHash = _CM(GetNextRnd)(&state->m_data[in_round].rndGen) % 18;
    
    switch(rndHash)
    {
        case RandomHashAlgos::RH_SHA2_256     :
        {
            _CM(RandomHash_SHA2_256)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_SHA2_384     :
        {
            _CM(RandomHash_SHA2_384)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_SHA2_512     :
        {
            _CM(RandomHash_SHA2_512)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_SHA3_256     :
        {
            _CM(RandomHash_SHA3_256)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_SHA3_384     :
        {
            _CM(RandomHash_SHA3_384)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_SHA3_512     :
        {
            _CM(RandomHash_SHA3_512)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_RIPEMD160    :
        {
            _CM(RandomHash_RIPEMD160)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_RIPEMD256    :
        {
            _CM(RandomHash_RIPEMD256)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_RIPEMD320    :
        {
            _CM(RandomHash_RIPEMD320)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Blake2b      :
        {
            _CM(RandomHash_blake2b)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Blake2s      :
        {            
            _CM(RandomHash_blake2s)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Tiger2_5_192 :
        {
            _CM(RandomHash_Tiger2_5_192)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Snefru_8_256 :
        {
            _CM(RandomHash_Snefru_8_256)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Grindahl512  :
        {
            _CM(RandomHash_Grindahl512)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Haval_5_256  :
        {
            _CM(RandomHash_Haval_5_256)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_MD5          :
        {
            _CM(RandomHash_MD5)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_RadioGatun32 :
        {
            _CM(RandomHash_RadioGatun32)(state->m_data[in_round].roundInput, output);
        } break;
        case RandomHashAlgos::RH_Whirlpool    :
        {
            _CM(RandomHash_WhirlPool)(state->m_data[in_round].roundInput, output);
        } break;
    }

    RH_STRIDE_CHECK_INTEGRITY(output);
    RH_STRIDE_CHECK_INTEGRITY(state->m_data[in_round].roundInput);
}

inline void CUDA_SYM_DECL(RandomHash_end)(RandomHash_State* state, int in_round)
{
    RH_StridePtr output = RH_STRIDEARRAY_GET(state->m_data[in_round].roundOutputs, RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) - 1);
    _CM(RandomHash_Expand)(state, output, in_round, RH_N - in_round);

    #ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3
        _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(state->m_data[in_round].roundOutputs, RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) - 1);
    #endif

    RH_STRIDEARRAY_RESET(state->m_data[in_round].io_results);
    _CM(RH_STRIDEARRAY_PUSHBACK_MANY)(state->m_data[in_round].io_results, state->m_data[in_round].roundOutputs); 
    
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(state->m_data[in_round].roundOutputs) <= GetRoundOutputCount(in_round));
}

inline void CUDA_SYM_DECL(RandomHash_FirstCall_push)(RandomHash_State* state, int in_round)
{
    state->m_data[5].in_blockHeader = state->m_data[0].roundInput;     //blockHerader
    state->m_data[5].io_results = state->m_data[0].parenAndNeighbortOutputs;       //allOutputs
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
                      cuda_RandomHash_Phase_1_push(state, 5);   \
                      cuda_RandomHash_Phase_init(state, 4);     \
                      cuda_RandomHash_Phase_1_push(state, 4);   \
                      cuda_RandomHash_Phase_init(state, 3);     \
                      cuda_RandomHash_Phase_1_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B1                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B2         cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B3                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B4         cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B5                                                           \
                      cuda_RandomHash_end(state, 2);            

#define RH_B6                       cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B7                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B8         cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B9                                                           \
                      cuda_RandomHash_end(state, 1);            

#define RH_B10        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B11                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B12                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B13                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B14                      cuda_RandomHash_Phase_1_pop(state, 4);    \
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

#define RH_B15                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B16        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B17                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B18        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B19                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B20                      cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B21                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B22        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B23                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B24        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B25                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B26                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B27                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B28                      cuda_RandomHash_Phase_2_pop(state, 4);    \
                                                                        \
                      cuda_RandomHash(state, 4);                

#define RH_B29                                                          \
                      cuda_RandomHash_end(state, 4);            

#define RH_B30                      cuda_RandomHash_Phase_1_pop(state, 5);    \
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
                      cuda_RandomHash(state, 1);                

#define RH_B31                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B32        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B33                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B34        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B35                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B36                      cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B37                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B38        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B39                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B40        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B41                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B42                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B43                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B44                      cuda_RandomHash_Phase_1_pop(state, 4);    \
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

#define RH_B45                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B46        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B47                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B48        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B49                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B50                      cuda_RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 3);   \
                      cuda_RandomHash_Phase_init(state, 2);     \
                      cuda_RandomHash_Phase_1_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B51                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B52        cuda_RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      cuda_RandomHash_Phase_2_push(state, 2);   \
                      cuda_RandomHash_Phase_init(state, 1);     \
                      cuda_RandomHash_start(state, 1);          \
                      cuda_RandomHash(state, 1);                

#define RH_B53                                                          \
                      cuda_RandomHash_end(state, 1);            

#define RH_B54        cuda_RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      cuda_RandomHash(state, 2);                

#define RH_B55                                                          \
                      cuda_RandomHash_end(state, 2);            

#define RH_B56                      cuda_RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      cuda_RandomHash(state, 3);                

#define RH_B57                                                          \
                      cuda_RandomHash_end(state, 3);            

#define RH_B58                      cuda_RandomHash_Phase_2_pop(state, 4);    \
                                                                        \
                      cuda_RandomHash(state, 4);                

#define RH_B59                                                          \
                      cuda_RandomHash_end(state, 4);            

#define RH_B60                      cuda_RandomHash_Phase_2_pop(state, 5);    \
                                                                        \
                      cuda_RandomHash(state, 5);                

#define RH_B61                                                          \
                      cuda_RandomHash_end(state, 5);            



//Kernels definitions
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


//-------------------------------------------------------------------------------------------------------------------------------------
//
CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Init)(RandomHash_State* allStates, U8* out_hash, U32 startNonce)
{
    CUDA_DECLARE_STATE();

    _CM(RandomHash_Initialize)(state);
    
    (*(U32*)(state->m_data[0].roundInput)) = PascalHeaderSize;
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(RH_STRIDE_GET_DATA(state->m_data[0].roundInput), &state->m_header[0], PascalHeaderSize); 
    
    RH_STRIDE_INIT(state->m_workBytes);

    startNonce += KERNEL_GET_GID();
    
    state->m_startNonce = startNonce;
    *(U32*)(RH_STRIDE_GET_DATA(state->m_data[0].roundInput) + PascalHeaderNoncePosV4(PascalHeaderSize)) = startNonce;
}

CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Finalize)(RandomHash_State* allStates, U8* output) 
{
    CUDA_DECLARE_STATE();

    RH_STRIDE_CHECK_INTEGRITY(RH_STRIDEARRAY_GET(state->m_data[5].roundOutputs, 30));

    _CM(RandomHash_Compress)(state, state->m_data[5].roundOutputs, state->m_data[5].roundInput);

    RH_STRIDE_RESET(state->m_workBytes)
    _CM(RandomHash_SHA2_256)(state->m_data[5].roundInput, state->m_workBytes);

#ifdef __CUDA_ARCH__
    U64 stateHigh64 = cuda_swab64(*(U64*)RH_STRIDE_GET_DATA(state->m_workBytes));

    if (stateHigh64 <= GetTarget())
    {
	    uint32_t index = atomicInc((U32*)output, 0xffffffff) + 1;
	    if (index > CUDA_SEARCH_RESULT_BUFFER_SIZE -1) 
            return;
        ((U32*)output)[index] = state->m_startNonce;
    }
#else
    memcpy(output, RH_STRIDE_GET_DATA(state->m_workBytes), 32);
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
        auto err = cudaMemGetInfo(&free_byte, &total_byte ) ;
        PrintOut("CUDA: Using %.2f mb (%.2f mb per threads). Free memory %.2f mb. \n", 
            g_totAlloc/1024/1024.0f, 
            (g_totAlloc/1024/1024.0f)/(float)g_threadsDataSize, 
            err == cudaSuccess ? (free_byte/1024.0f/1024.0f):-1.0f);
    }
}

__host__ void cuda_randomhash_init(uint32_t* input, U32 nonce2)
{
    for(int i=0; i < g_threadsDataSize; i++)
    {
        CUDA_SYM(RandomHash_SetHeader)(&g_threadsData[i], (U8*)input, nonce2);
    }
}


__host__ void cuda_randomhash_search(uint32_t blocks, uint32_t threadsPerBlock, cudaStream_t stream, uint32_t* input, uint32_t* output, U32 startNonce)
{  
    RHMINER_ASSERT((threadsPerBlock*blocks == g_threadsDataSize));
    
    RandomHash_State* allStates = g_threadsData;

    CUDA_SYM(RandomHash_Init)<<<threadsPerBlock, blocks>>>(allStates, (uint8_t*)output, startNonce);
    RH_CALL_ALL_KERNEL_BLOCKS
    CUDA_SYM(RandomHash_Finalize)<<<threadsPerBlock, blocks>>>(allStates, (uint8_t*)output);
}

#endif //RH_COMPILE_CPU_ONLY

