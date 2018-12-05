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

extern bool g_disableCachedNonceReuse;

struct RH_StrideArrayStruct
{
    U32 size;
    U32 maxSize;
    U8  dummy[(RH_IDEAL_ALIGNMENT/2) - 8];
    MurmurHash3_x86_32_State accum;
    U8  dummy2[(RH_IDEAL_ALIGNMENT/2) - sizeof(MurmurHash3_x86_32_State)];
#ifdef RHMINER_DEBUG_STRIDE_INTEGRITY_CHECK
    U8* strides[RH_StrideArrayCount + 1];
#else
    U8* strides[RH_StrideArrayCount];
#endif
};
#define RH_StrideArrayStruct_GetAccum(strideArray) (&((RH_StrideArrayStruct*)strideArray)->accum)

#include "corelib/CommonData.h"

//--------------------------------------------------------------------------------------------------
const size_t c_round_outputsCounts[6] = {0, 1, 3, 7, 15, 31 };
const size_t c_round_parenAndNeighbortOutputsCounts[6] = { 31, 0, 1, 3, 7, 15 };


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
inline RH_StridePtr CUDA_SYM(RH_StrideArrayAllocOutput)(RandomHash_State* state) 
{
    RH_StridePtr stride = ((U8*)state->m_stridesInstances) + state->m_stridesAllocIndex*RH_StrideSize;

    state->m_stridesAllocIndex++;
    RHMINER_ASSERT(state->m_stridesAllocIndex < RH_TOTAL_STRIDES_INSTANCES);
    RHMINER_ASSERT( (((size_t)stride) % 32) == 0);

    PLATFORM_MEMSET(stride, 0, RH_StrideSize);
    RH_STRIDE_INIT(stride);

    return stride;
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
    PLATFORM_MEMSET(arr->strides, 0, RH_StrideArrayCount * sizeof(void*));
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
   
   _CM(RandomHash_Free)(state->m_cachedOutputs);
   _CM(RandomHash_Free)(state->m_cachedHheader);

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
    RH_STRIDEARRAY_INIT(arrayData);
    RH_CUDA_ERROR_CHECK();
}


void CUDA_SYM(RandomHash_RoundDataAlloc)(RH_RoundData* rd, int round)
{
    PLATFORM_MEMSET(rd, 0, sizeof(RH_RoundData));

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

    //no need of this buffer for round #0
    if (round > 0) 
    {
        _CM(RandomHash_Alloc)((void**)&rd->otherNonceHeader, RH_StrideSize);
        RH_CUDA_ERROR_CHECK();
    }    
}


void CUDA_SYM(RandomHash_Create)(RandomHash_State* state)
{
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
    _CM(RandomHash_Free)(rd->parenAndNeighbortOutputs);
    _CM(RandomHash_Free)(rd->roundInput);
    _CM(RandomHash_Free)(rd->otherNonceHeader);
}

void CUDA_SYM(RandomHash_SetHeader)(RandomHash_State* state, U8* sourceHeader, U32 nonce2)
{
    U8* targetInput = state->m_header;
    
    RH_STRIDEARRAY_RESET(state->m_cachedOutputs);
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

extern bool g_disableFastTransfo;
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
        U32 r = random % 8;
        RH_ASSERT((nextChunkSize & 1) == 0);

        if (g_disableFastTransfo)
        {
            U8* workBytes = state->m_workBytes;
            _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, outputPtr, nextChunkSize); 
            RH_STRIDE_CHECK_INTEGRITY(output);
            switch(r)
            {
                case 0: _CM(Transfo0)(nextChunk, nextChunkSize,  workBytes); break;
                case 1: _CM(Transfo1)(nextChunk, nextChunkSize,  workBytes); break;
                case 2: _CM(Transfo2)(nextChunk, nextChunkSize,  workBytes); break; 
                case 3: _CM(Transfo3)(nextChunk, nextChunkSize,  workBytes); break;
                case 4: _CM(Transfo4)(nextChunk, nextChunkSize,  workBytes); break; 
                case 5: _CM(Transfo5)(nextChunk, nextChunkSize,  workBytes); break;
                case 6: _CM(Transfo6)(nextChunk, nextChunkSize); break;
                case 7: _CM(Transfo7)(nextChunk, nextChunkSize); break;
            }
        }
        else
        {
            switch(r)
            {
                case 0: _CM(Transfo0_2)(nextChunk, nextChunkSize,  outputPtr); break;
                case 1: _CM(Transfo1_2)(nextChunk, nextChunkSize,  outputPtr); break;
                case 2: _CM(Transfo2_2)(nextChunk, nextChunkSize,  outputPtr); break;
                case 3: _CM(Transfo3_2)(nextChunk, nextChunkSize,  outputPtr); break;
                case 4: _CM(Transfo4_2)(nextChunk, nextChunkSize,  outputPtr); break;
                case 5: _CM(Transfo5_2)(nextChunk, nextChunkSize,  outputPtr); break; 
                case 6: _CM(Transfo6_2)(nextChunk, nextChunkSize,  outputPtr); break;
                case 7: _CM(Transfo7_2)(nextChunk, nextChunkSize,  outputPtr); break;
            }
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

        rval = _CM(GetNextRnd)(&state->m_rndGenCompress);
        resultPtr[i] = RH_STRIDE_GET_DATA(source)[rval % sourceSize];
    }
        
    RH_STRIDE_CHECK_INTEGRITY(Result);
} 

inline void CUDA_SYM_DECL(RandomHash_start)(RandomHash_State* state, U32 in_round)
{
    RH_ASSERT(in_round >= 1 && in_round <= RH_N);
    U32 seed = _CM(RandomHash_Checksum)(state->m_data[in_round].in_blockHeader);      // can hash in_blockHeader first, but not required
    _CM(RandomHash_Reseed)(state->m_data[in_round].rndGen, seed);

    RH_STRIDE_COPY(state->m_data[in_round].roundInput, state->m_data[in_round].in_blockHeader);
}

inline void CUDA_SYM_DECL(RandomHash_Phase_1_push)(RandomHash_State* state, int in_round)
{
    if (in_round == RH_N && !g_disableCachedNonceReuse)
    {
        RH_ASSERT((RH_STRIDE_GET_SIZE(state->m_data[RH_N].in_blockHeader) <= PascalHeaderSize));
        U32* headPtr = (U32*)RH_STRIDE_GET_DATA(state->m_data[RH_N].in_blockHeader);        

        U32* tailPtr = headPtr + (RH_STRIDE_GET_SIZE(state->m_data[RH_N].in_blockHeader)/4) - 1; 
        U32* cachedtailPtr = (U32*)(state->m_cachedHheader + PascalHeaderSize - 4);

        if (*headPtr == *(U32*)(state->m_cachedHheader))
        {
            while (*tailPtr == *cachedtailPtr && tailPtr != headPtr)
            {
                tailPtr--;
                cachedtailPtr--;
            }
        
            //found in cache
            if (tailPtr == headPtr)
            {
                //Raise Skip flag
                state->m_skipPhase1 = 1;
                return;
            }
        }
    }

    state->m_data[in_round-1].backup_in_blockHeader = state->m_data[in_round-1].in_blockHeader;
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round].in_blockHeader;
    state->m_data[in_round-1].backup_io_results = state->m_data[in_round-1].io_results;
    state->m_data[in_round-1].io_results = state->m_data[in_round].parenAndNeighbortOutputs;
}

inline void CUDA_SYM_DECL(RandomHash_Phase_1_pop)(RandomHash_State* state, int in_round)
{
    if (in_round == RH_N && state->m_skipPhase1)
    {
        _CM(RH_STRIDEARRAY_CLONE)(state->m_data[RH_N].parenAndNeighbortOutputs, state->m_cachedOutputs, state);

        RH_STRIDEARRAY_CHECK_INTEGRITY(state->m_cachedOutputs);
        RH_STRIDEARRAY_RESET(state->m_cachedOutputs);
        state->m_skipPhase1 = 0;
    }
    else
    {   
        state->m_data[in_round-1].in_blockHeader = state->m_data[in_round-1].backup_in_blockHeader;
        state->m_data[in_round-1].io_results = state->m_data[in_round-1].backup_io_results;
    }

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
    
    if (in_round == RH_N && !g_disableCachedNonceReuse)
    {
        RH_ASSERT(RH_STRIDE_GET_SIZE(state->m_data[RH_N].otherNonceHeader) <= PascalHeaderSize);
        _CM(RH_STRIDE_MEMCPY_ALIGNED_SIZE128)(state->m_cachedHheader, RH_STRIDE_GET_DATA(state->m_data[RH_N].otherNonceHeader), PascalHeaderSize);
    }

    state->m_data[in_round-1].backup_in_blockHeader = state->m_data[in_round-1].in_blockHeader;
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round].otherNonceHeader;
    state->m_data[in_round-1].backup_io_results = state->m_data[in_round-1].io_results;
    state->m_data[in_round-1].io_results = state->m_data[in_round].parenAndNeighbortOutputs;
}


void CUDA_SYM_DECL(RandomHash_Phase_2_pop)(RandomHash_State* state, int in_round)         
{
    state->m_data[in_round-1].in_blockHeader = state->m_data[in_round-1].backup_in_blockHeader;
    state->m_data[in_round-1].io_results = state->m_data[in_round-1].backup_io_results;

    if (in_round == RH_N && !g_disableCachedNonceReuse)
    {
        _CM(RH_STRIDEARRAY_CLONE)(state->m_cachedOutputs, state->m_data[RH_N].parenAndNeighbortOutputs, state);
    }

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
    state->m_data[5].in_blockHeader = state->m_data[0].roundInput;     
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
#define RH_B0         RandomHash_FirstCall_push(state, 5); \
                      RandomHash_Phase_init(state, 5);     \
                      RandomHash_Phase_1_push(state, 5);   \
                      if (!state->m_skipPhase1) {   \
                      RandomHash_Phase_init(state, 4);     \
                      RandomHash_Phase_1_push(state, 4);   \
                      RandomHash_Phase_init(state, 3);     \
                      RandomHash_Phase_1_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      RandomHash(state, 3);                \
                                                                        \
                      RandomHash_end(state, 3);            \
                      RandomHash_Phase_1_pop(state, 4);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 4);   \
                      RandomHash_Phase_init(state, 3);     \
                      RandomHash_Phase_1_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      RandomHash(state, 3);                \
                                                                        \
                      RandomHash_end(state, 3);            \
                      RandomHash_Phase_2_pop(state, 4);    \
                                                                        \
                      RandomHash(state, 4);                \
                                                                        \
                      RandomHash_end(state, 4);            \
                            } \
                      RandomHash_Phase_1_pop(state, 5);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 5);   \
                      RandomHash_Phase_init(state, 4);     \
                      RandomHash_Phase_1_push(state, 4);   \
                      RandomHash_Phase_init(state, 3);     \
                      RandomHash_Phase_1_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      RandomHash(state, 3);                \
                                                                        \
                      RandomHash_end(state, 3);            \
                      RandomHash_Phase_1_pop(state, 4);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 4);   \
                      RandomHash_Phase_init(state, 3);     \
                      RandomHash_Phase_1_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_1_pop(state, 3);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 3);   \
                      RandomHash_Phase_init(state, 2);     \
                      RandomHash_Phase_1_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_1_pop(state, 2);    \
                                                                        \
                                                                        \
                      RandomHash_Phase_2_push(state, 2);   \
                      RandomHash_Phase_init(state, 1);     \
                      RandomHash_start(state, 1);          \
                      RandomHash(state, 1);                \
                                                                        \
                      RandomHash_end(state, 1);            \
                      RandomHash_Phase_2_pop(state, 2);    \
                                                                        \
                      RandomHash(state, 2);                \
                                                                        \
                      RandomHash_end(state, 2);            \
                      RandomHash_Phase_2_pop(state, 3);    \
                                                                        \
                      RandomHash(state, 3);                \
                                                                        \
                      RandomHash_end(state, 3);            \
                      RandomHash_Phase_2_pop(state, 4);    \
                                                                        \
                      RandomHash(state, 4);                \
                                                                        \
                      RandomHash_end(state, 4);            \
                      RandomHash_Phase_2_pop(state, 5);    \
                                                                        \
                      RandomHash(state, 5);                \
                                                                        \
                      RandomHash_end(state, 5);            \



RH_DEFINE_KERNEL_BLOCK(0)

#define RH_CALL_ALL_KERNEL_BLOCKS \
RH_CALL_KERNEL_BLOCK(0) \

//-------------------------------------------------------------------------------------------------------------------------------------
CUDA_DECL_KERNEL void CUDA_SYM(RandomHash_Init)(RandomHash_State* allStates, U8* out_hash, U32 startNonce)
{
    CUDA_DECLARE_STATE();

    _CM(RandomHash_Initialize)(state);
    
    (*(U32*)(state->m_data[0].roundInput)) = PascalHeaderSize;
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(RH_STRIDE_GET_DATA(state->m_data[0].roundInput), &state->m_header[0], PascalHeaderSize); 

    RH_STRIDE_INIT(state->m_workBytes);

    if (RH_STRIDEARRAY_GET_SIZE(state->m_cachedOutputs) && !g_disableCachedNonceReuse)
    {
        startNonce = *(U32*)(state->m_cachedHheader + PascalHeaderSize - 4);
    }

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

    memcpy(output, RH_STRIDE_GET_DATA(state->m_workBytes), 32);
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
