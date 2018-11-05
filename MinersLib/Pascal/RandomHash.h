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

#ifndef RANDOM_HASH_OPTIMIZED_H
#define RANDOM_HASH_OPTIMIZED_H

#include "MinersLib/Pascal/RandomHash_core.h"


using namespace std;


struct RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_RoundData
{
    //Random hash variable instances (device ptrs)
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtr             roundInput;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtr             otherNonceHeader;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) mersenne_twister_state   rndGen; 

    //ptr to stride instances
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtrArray        roundOutputs;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtrArray        parenAndNeighbortOutputs; 

    //Input args pointers (Pushed/Poped)
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtr             in_blockHeader;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtrArray        io_results;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtr             backup_in_blockHeader;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtrArray        backup_io_results;
};

struct RH_ALIGN(RH_IDEAL_ALIGNMENT) RandomHash_State
{
    RH_ALIGN(RH_IDEAL_ALIGNMENT) U8                       m_header[PascalHeaderSize]; 
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_RoundData             m_data[RH_N+1];
    RH_ALIGN(RH_IDEAL_ALIGNMENT) U8                       m_workBytes[RH_WorkSize];
    RH_ALIGN(RH_IDEAL_ALIGNMENT) mersenne_twister_state   m_rndGenCompress;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) mersenne_twister_state   m_rndGenExpand;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) U32                      m_startNonce;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtr             m_stridesInstances;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) U32                      m_stridesAllocIndex;


    //CPU ONLY
    RH_ALIGN(RH_IDEAL_ALIGNMENT) U8*                      m_cachedHheader;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) RH_StridePtrArray        m_cachedOutputs;
    RH_ALIGN(RH_IDEAL_ALIGNMENT) U32                      CUDA_SYM(m_skipPhase1);
};

//External API functions
extern void             CUDA_SYM(RandomHash_CreateMany)(RandomHash_State** outPtr, U32 count);
extern void             CUDA_SYM(RandomHash_DestroyMany)(RandomHash_State* stateArray, U32 count);
extern void             CUDA_SYM(RandomHash_Create)(RandomHash_State* state);
extern void             CUDA_SYM(RandomHash_Destroy)(RandomHash_State* state);
extern void             CUDA_SYM(RandomHash_SetTarget)(uint64_t target);

//External alloc func
extern void             CUDA_SYM(RandomHash_RoundDataAlloc)(RH_RoundData* rd, int round);
extern void             CUDA_SYM(RandomHash_RoundDataUnInit)(RH_RoundData* rd, int round);
extern void             CUDA_SYM(RandomHash_Free)(void* ptr);
extern void             CUDA_SYM(RandomHash_Alloc)(void** out_ptr, size_t size);
extern void             CUDA_SYM(RandomHash_SetHeader)(RandomHash_State* state, U8* sourceHeader, U32 nonce2);

#ifndef RANDOMHASH_CUDA
    extern void RandomHash_Search(RandomHash_State* state, U8* out_hash, U32 startNonce);
#endif


#endif //RANDOM_HASH_OPTIMIZED_H

