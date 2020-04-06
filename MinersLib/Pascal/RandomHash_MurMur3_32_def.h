#ifndef RANDOM_HASH_MurMur3_32_DEF_h
#define RANDOM_HASH_MurMur3_32_DEF_h

//-----------------------------------------------------------------------------
//https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
//
//* Copyright 2018 Polyminer1 <https://github.com/polyminer1>

#include "MinersLib/Pascal/RandomHash_def.h"
#include "corelib/basetypes.h"


//----------------------------------------------------------------------------------------------------
PLATFORM_CONST uint32_t MurmurHash3_x86_32_c1 = 0xcc9e2d51;
PLATFORM_CONST uint32_t MurmurHash3_x86_32_c2 = 0x1b873593;
PLATFORM_CONST uint32_t MurmurHash3_x86_32_c3 = 0xe6546b64;
PLATFORM_CONST uint32_t MurmurHash3_x86_32_c4 = 0x85ebca6b;
PLATFORM_CONST uint32_t MurmurHash3_x86_32_c5 = 0xc2b2ae35;

struct MurmurHash3_x86_32_State
{
    union
	{
		uint8_t         buf[4]; 
    	uint32_t		buf32;
	} U;
    uint32_t        idx;    
    uint32_t        totalLen = 0;
    uint32_t        h1 = 0;
};


#define RH_MUR3_BACKUP_STATE(state)                 \
register uint32_t        back_buf = (state)->U.buf32;     \
register uint32_t        back_idx = (state)->idx;              \
register uint32_t        back_totalLen = (state)->totalLen;    \
register uint32_t        back_h1 = (state)->h1;                \
register uint32_t        back_i = 0;

#define RH_MUR3_RESTORE_STATE(state)                 \
(state)->U.buf32 = back_buf; \
(state)->idx = back_idx; \
(state)->totalLen = back_totalLen; \
(state)->h1 = back_h1; 

#define MURMUR3_BODY(k) { \
            uint32_t k1 = (k); \
            k1 *= MurmurHash3_x86_32_c1; \
            k1 = ROTL32(k1, 15); \
            k1 *= MurmurHash3_x86_32_c2; \
            h1 ^= k1; \
            h1 = ROTL32(h1, 13); \
            h1 = h1 * 5 + MurmurHash3_x86_32_c3; } 

#define RH_MURMUR3_BODY_2(k1, hx) \
    hx ^= k1; \
    hx = ROTL32(hx, 13); \
    hx = hx * 5 + MurmurHash3_x86_32_c3;


inline void MurmurHash3_x86_32_Init(uint32_t seed, MurmurHash3_x86_32_State* state)
{
    state->U.buf32 = 0;
    state->idx = 0;
    state->totalLen = 0;
    state->h1 = 0;
}

#define INPLACE_M_MurmurHash3_x86_32_Update_1(chunk8)  \
{                                                      \
    RH_ASSERT(back_idx != 0xDEADBEEF)                \
    RH_ASSERT(back_idx != 4);                        \
    back_totalLen++;                                   \
    back_buf &= ~(0xFF << (back_idx*8));               \
    back_buf |= (chunk8 << (back_idx*8));              \
    back_idx++;                                        \
    if (back_idx == 4)                                 \
    {                                                  \
        back_buf *= MurmurHash3_x86_32_c1; \
        back_buf = ROTL32(back_buf, 15); \
        back_buf *= MurmurHash3_x86_32_c2; \
        back_h1 ^= back_buf; \
        back_h1 = ROTL32(back_h1, 13); \
        back_h1 = back_h1 * 5 + MurmurHash3_x86_32_c3; \
        back_idx = 0;                                  \
        back_buf = 0; \
    }                                                  \
}

#define INPLACE_M_MurmurHash3_x86_32_Update_1_NL(b)         \
{                                                           \
    U32 chunk8 = b;                                         \
    chunk8 = chunk8 << (back_idx * 8);                      \
    back_buf |= chunk8;                                     \
    back_idx++;                                             \
    if (back_idx == 4)                                      \
    {                                                       \
        back_buf *= MurmurHash3_x86_32_c1;                  \
        back_buf = ROTL32(back_buf, 15);                    \
        back_buf *= MurmurHash3_x86_32_c2;                  \
        back_h1 ^= back_buf;                                \
        back_h1 = ROTL32(back_h1, 13);                      \
        back_h1 = back_h1 * 5 + MurmurHash3_x86_32_c3;      \
        back_idx = 0;                                       \
        back_buf = 0;                                       \
    }                                                       \
}



#endif //RANDOM_HASH_MurMur3_32_h
