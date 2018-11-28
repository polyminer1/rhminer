#ifndef RANDOM_HASH_MurMur3_32_h
#define RANDOM_HASH_MurMur3_32_h

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
#include "MinersLib/Pascal/PascalCommon.h"


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
    uint8_t         buf[4]; //pending
    	uint32_t		buf32;
	} U;
    uint32_t        idx;    //pending size
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

inline void CUDA_SYM_DECL(MurmurHash3_x86_32_Init)(uint32_t seed, MurmurHash3_x86_32_State* state)
{
    state->U.buf32 = 0;
    state->idx = 0;
    state->totalLen = 0;
    state->h1 = 0;
}

#define MURMUR3_BODY(k) { \
            uint32_t k1 = (k); \
            k1 *= MurmurHash3_x86_32_c1; \
            k1 = ROTL32(k1, 15); \
            k1 *= MurmurHash3_x86_32_c2; \
            h1 ^= k1; \
            h1 = ROTL32(h1, 13); \
            h1 = h1 * 5 + MurmurHash3_x86_32_c3; } 



#define INPLACE_M_MurmurHash3_x86_32_Update_1(chunk8)  \
{                                                      \
    RH_ASSERT(back_idx != 0xDEADBEEF)                \
    RH_ASSERT(back_idx != 4);                        \
    back_totalLen++;                                   \
    uint32_t h1 = back_h1;                             \
    back_buf &= ~(0xFF << (back_idx*8));               \
    back_buf |= (chunk8 << (back_idx*8));              \
    back_idx++;                                        \
    if (back_idx == 4)                                 \
    {                                                  \
        MURMUR3_BODY(back_buf)                         \
        back_idx = 0;                                  \
    }                                                  \
    back_h1 = h1;                                      \
}



void CUDA_SYM_DECL(MurmurHash3_x86_32_Update_8)(U64 chunk64, uint32_t len, MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(len < S32_Max);
    RH_ASSERT(state->idx != 0xDEADBEEF)

    state->totalLen += len;
    uint32_t h1 = state->h1;
    int i = 0;

    //sonsume pending bytes
    if (state->idx)
    {
        while(len)
        {
            while (state->idx < 4 && len)
            {
                U32 b = (U8)(chunk64 >> (i*8)); //TODO: Manage endianness
                state->U.buf[state->idx] = b;
                state->idx++;
                len--;
                i++;
            }
            
            //update buf
            if (state->idx == 4)
            {
                MURMUR3_BODY(state->U.buf32)
                state->idx = 0;
            }
        }
    }
    else
    {
        const int nblocks = len >> 2;
        //TODO: optimiz - UNROLL 2 !!!
        while (i < nblocks)
        {
            U32 block = (U32)(chunk64 >> (i*32));   //TODO: Manage endianness
            MURMUR3_BODY(block);
            i++;
            RH_ASSERT(i <= 2);
        }

        //save pending end bytes
        i = (nblocks * 4);
	    while (i < (int)len)
        {
            RH_ASSERT(state->idx < 4);
            U32 b = (U8)(chunk64 >> (i*8)); //TODO: Manage endianness
            state->U.buf[state->idx] = b;
            state->idx++;
            i++;
        
            //update buf
            if (state->idx == 4)
            {
                MURMUR3_BODY(state->U.buf32)
                state->idx = 0;
            }
        }
    }
    
    state->h1 = h1;
}

#define INPLACE_M_MurmurHash3_x86_32_Update_8(chunk64, _len)         \
{                                                                    \
    RH_ASSERT(back_idx != 0xDEADBEEF)                              \
    U32 len = _len;                                                  \
    back_totalLen += len;                                            \
    uint32_t h1 = back_h1;                                           \
    back_i = 0;                                                      \
    if (back_idx)                                                    \
    {                                                                \
        while(len)                                                   \
        {                                                            \
            while (back_idx < 4 && len)                              \
            {                                                        \
                U32 b = (U8)(chunk64 >> (back_i*8));                      \
                back_buf &= ~(0xFF << (back_idx*8));                 \
                back_buf |= (b << (back_idx*8));                     \
                back_idx++;                                          \
                len--;                                               \
                back_i++;                                                 \
            }                                                        \
            if (back_idx == 4)                                       \
            {                                                        \
                MURMUR3_BODY(back_buf)                               \
                back_idx = 0;                                        \
            }                                                        \
        }                                                            \
    }                                                                \
    else                                                             \
    {                                                                \
        const U32 nblocks = len >> 2;                                \
        while (back_i < nblocks)                                          \
        {                                                            \
            U32 block = (U32)(chunk64 >> (back_i*32));                    \
            MURMUR3_BODY(block);                                     \
            back_i++;                                                     \
            RH_ASSERT(back_i <= 2);                                     \
        }                                                            \
                                                                     \
        back_i = (nblocks * 4);                                           \
	    while (back_i < len)                                         \
        {                                                            \
            RH_ASSERT(back_idx < 4);                               \
            U32 b = (U8)(chunk64 >> (back_i*8));                          \
            back_buf &= ~(0xFF << (back_idx*8));                     \
            back_buf |= (b << (back_idx*8));                         \
            back_idx++;                                              \
            back_i++;                                                     \
            if (back_idx == 4)                                       \
            {                                                        \
                MURMUR3_BODY(back_buf)                               \
                back_idx = 0;                                        \
            }                                                        \
        }                                                            \
    }                                                                \
                                                                     \
    back_h1 = h1;                                                    \
}

#ifdef RANDOMHASH_CUDA

inline void CUDA_SYM_DECL(MurmurHash3_x86_32_Update_16)(uint4 data, uint32_t len, MurmurHash3_x86_32_State* state)
{
    U64 r0, r1;
    devectorize4(data, r0, r1);
    if (len > 8)
    {
        len -= 8;
        _CM(MurmurHash3_x86_32_Update_8)(r0, 8, state);
        _CM(MurmurHash3_x86_32_Update_8)(r1, len, state);
    }
    else
        _CM(MurmurHash3_x86_32_Update_8)(r0, len, state);
}

#else

#define RH_M3_GET_BYTE(chunk128, n, b, d)                               \
        d = ((n) & 0x7)*8;                                              \
        switch((n)>>2)                                                  \
        {                                                               \
            case 0:b = _mm_extract_epi32_M(chunk128, 0)>>d; break;       \
            case 1:b = _mm_extract_epi32_M(chunk128, 1)>>d; break;       \
            case 2:b = _mm_extract_epi32_M(chunk128, 2)>>d; break;       \
            case 3:b = _mm_extract_epi32_M(chunk128, 3)>>d; break;       \
            default:                                                    \
                RH_ASSERT(false);                                  \
        }

#define RH_M3_GET_WORD(chunk128, n, w)                              \
        switch((n)>>2)                                              \
        {                                                           \
            case 0:w = _mm_extract_epi32_M(chunk128, 0); break;       \
            case 1:w = _mm_extract_epi32_M(chunk128, 1); break;       \
            case 2:w = _mm_extract_epi32_M(chunk128, 2); break;       \
            case 3:w = _mm_extract_epi32_M(chunk128, 3); break;       \
            default:                                                \
                RH_ASSERT(false);                              \
        }

inline void CUDA_SYM_DECL(MurmurHash3_x86_32_Update_16)(__m128i chunk128, uint32_t len, MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(len < S32_Max);
    RH_ASSERT(state->idx != 0xDEADBEEF)
        
    state->totalLen += len;
    uint32_t h1 = state->h1;
    int i = 0;
    U8 b;
    U32 d;

    //sonsume pending bytes
    if (state->idx)
    {
        while(len)
        {
            while (state->idx < 4 && len)
            {
                RH_M3_GET_BYTE(chunk128, i, b, d);
                state->U.buf[state->idx++] = b;
                len--;
                i++;
            }
            
            //update buf
            if (state->idx == 4)
            {
                MURMUR3_BODY(state->U.buf32)
                state->idx = 0;
            }
        }
    }
    else
    {
        const int nblocks = len >> 2;
        //TODO: optimiz - UNROLL 4 !!!
        while (i < nblocks)
        {
            RH_M3_GET_WORD(chunk128, i, d);
            MURMUR3_BODY(d);
            i++;
            RH_ASSERT(i <= 4);
        }

        //save pending end bytes
        i = (nblocks * 4);
	    while (i < (int)len)
        {
            RH_ASSERT(state->idx < 4);
            RH_M3_GET_BYTE(chunk128, i, b, d);
            state->U.buf[state->idx++] = b;
            i++;
        
            //update buf
            if (state->idx == 4)
            {
                MURMUR3_BODY(state->U.buf32)
                state->idx = 0;
            }
        }
    }

    state->h1 = h1;
}

#define INPLACE_M_MurmurHash3_x86_32_Update_16(chunk128, _len)                      \
{                                                                                   \
    RH_ASSERT(back_idx != 0xDEADBEEF)                                             \
    U32 len = _len;                                                                 \
    back_totalLen += len;                                                           \
    uint32_t h1 = back_h1;                                                          \
    U8 b;                                                                           \
    U32 d;                                                                          \
    back_i = 0;                                                                     \
    if (back_idx)                                                                   \
    {                                                                               \
        while(len)                                                                  \
        {                                                                           \
            while (back_idx < 4 && len)                                             \
            {                                                                       \
                RH_M3_GET_BYTE(chunk128, back_i, b, d);                      \
                back_buf &= ~(0xFF << (back_idx*8));                                \
                back_buf |= (b << (back_idx*8));                                    \
                back_idx++;                                                         \
                len--;                                                              \
                back_i++;                                                                \
            }                                                                       \
            if (back_idx == 4)                                                      \
            {                                                                       \
                MURMUR3_BODY(back_buf)                                              \
                back_idx = 0;                                                       \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        const int nblocks = len >> 2;                                               \
        while (back_i < nblocks)                                                         \
        {                                                                           \
            RH_M3_GET_WORD(chunk128, back_i, d);                                \
            MURMUR3_BODY(d);                                                    \
            back_i++;                                                                    \
            RH_ASSERT(back_i <= 4);                                                    \
        }                                                                           \
        back_i = (nblocks * 4);                                                          \
	    while (back_i < (int)len)                                                        \
        {                                                                           \
            RH_ASSERT(back_idx < 4);                                              \
            U8 b = RH_M3_GET_BYTE(chunk128, back_i, b, d);                          \
            back_buf &= ~(0xFF << (back_idx*8));                                    \
            back_buf |= (b << (back_idx*8));                                        \
            back_idx++;                                                             \
            back_i++;                                                                    \
            if (back_idx == 4)                                                      \
            {                                                                       \
                MURMUR3_BODY(back_buf)                                              \
                back_idx = 0;                                                       \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    back_h1 = h1;                                                                   \
}
#endif

void CUDA_SYM_DECL(MurmurHash3_x86_32_Update)( const uint8_t* data, int len, MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(state->idx != 0xDEADBEEF)
    
    state->totalLen += len;
    uint32_t h1 = state->h1;
    uint32_t a_index = 0;

    //sonsume pending bytes
    if (state->idx && len)
    {
        while (state->idx < 4 && len)
        {
            state->U.buf[state->idx++] = *(data + a_index);
            a_index++;
            len--;
        }
            
        //update buf
        if (state->idx == 4)
        {
            MURMUR3_BODY(state->U.buf32)
            state->idx = 0;
        }
    }
    else
    {
        //assert(0);
    }
/*
    //allign ptr for cuda
    while (((((size_t)data + a_index)) % 8) && len)
    {
        while (state->idx < 4 && len)
        {
            state->U.buf[state->idx++] = *(data+ a_index);
            a_index++;
            len--;
        }
            
        //update buf
        if (state->idx == 4)
        {
            MURMUR3_BODY(*(uint32_t*)state->U.buf)
            state->idx = 0; 
        }
    }
*/
    //----------
    // body
    int i = 0;
    const int nblocks = len >> 2;
    const uint32_t * blocks = (const uint32_t *)(data + a_index);
    while (i < nblocks)
    {
        MURMUR3_BODY(blocks[i]);  //TODO: optimiz use 64 bit reg with 1 slr in 2 steps
        i++;
    }

    //save pending end bytes
    uint32_t offset = a_index + (nblocks * 4);
	while (offset < (len + a_index))
    {
        RH_ASSERT(state->idx < 4);
        state->U.buf[state->idx++] = *(data + offset++);
        
        //update buf
        if (state->idx == 4)
        {
            MURMUR3_BODY(state->U.buf32)
            state->idx = 0;
        }
    }

    state->h1 = h1;
}

inline uint32_t CUDA_SYM_DECL(MurmurHash3_x86_32_Finalize)( MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(state->idx != 0xDEADBEEF)

    //----------
    // tail / finish
    const uint8_t * tail = (const uint8_t*)(state->U.buf);
    uint32_t h1 = state->h1;
    uint32_t k1 = 0;
    switch (state->idx & 3)
    {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
        k1 *= MurmurHash3_x86_32_c1; 
        k1 = ROTL32(k1, 15); 
        k1 *= MurmurHash3_x86_32_c2; 
        h1 ^= k1;
    };

    //----------
    // Finalization
    h1 ^= state->totalLen;

    // Finalization mix - force all bits of a hash block to avalanche
    h1 ^= h1 >> 16;
    h1 *= MurmurHash3_x86_32_c4;
    h1 ^= h1 >> 13;
    h1 *= MurmurHash3_x86_32_c5;
    h1 ^= h1 >> 16;

#ifdef _DEBUG
    state->idx = 0xDEADBEEF;
#endif

    return h1;
}


uint32_t CUDA_SYM_DECL(MurmurHash3_x86_32_Fast)(const void * key, int len, uint32_t h1)
{
    const uint8_t * data = (const uint8_t*)key;

    //----------
    // body    
    int i = 0;
    const int nblocks = len >> 2;
    const uint32_t * blocks = (const uint32_t *)(data);
    while (i < nblocks)
    {
        MURMUR3_BODY(blocks[i]);
        i++;
    }
    
    //----------
    // tail / finish

    const uint8_t * tail = (const uint8_t*)(data + nblocks * 4);

    uint32_t k1 = 0;
    switch (len & 3)
    {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
        k1 *= MurmurHash3_x86_32_c1; 
        k1 = ROTL32(k1, 15); 
        k1 *= MurmurHash3_x86_32_c2; 
        h1 ^= k1;
    };
    
    //----------
    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= MurmurHash3_x86_32_c4;
    h1 ^= h1 >> 13;
    h1 *= MurmurHash3_x86_32_c5;
    h1 ^= h1 >> 16;

    return h1;
}


#endif //RANDOM_HASH_MurMur3_32_h
