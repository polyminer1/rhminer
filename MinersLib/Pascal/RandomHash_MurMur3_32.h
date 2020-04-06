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

#include "MinersLib/Pascal/RandomHash_MurMur3_32_def.h"
#include "MinersLib/Pascal/PascalCommon.h"

void MurmurHash3_x86_32_Update_8(U64 chunk64, uint32_t len, MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(len < S32_Max)
    RH_ASSERT(state->idx != 0xDEADBEEF)
    RH_ASSERT(len <= sizeof(U64))

    state->totalLen += len;
    uint32_t h1 = state->h1;
    int i = 0;

    if (state->idx)
    {
        while(len)
        {
            while (state->idx < 4 && len)       
            {
                U32 b = (U8)(chunk64 >> (i*8)); 
                state->U.buf[state->idx] = b;
                state->idx++;
                len--;
                i++;
            }
            
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
        while (i < nblocks)     
        {
            U32 block = (U32)(chunk64 >> (i*32));   
            MURMUR3_BODY(block);
            i++;
            RH_ASSERT(i <= 2);
        }

        
        i = (nblocks * 4);
	    while (i < (int)len)        
        {
            RH_ASSERT(state->idx < 4);
            U32 b = (U8)(chunk64 >> (i*8)); 
            state->U.buf[state->idx] = b;
            state->idx++;
            i++;
        
            
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
            RH_ASSERT(back_i <= 2)                                     \
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

void MurmurHash3_x86_32_Update( const uint8_t* data, int len, MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(state->idx != 0xDEADBEEF)
    
    state->totalLen += len;
    uint32_t h1 = state->h1;
    uint32_t a_index = 0;

    
    if (state->idx && len)
    {
        while (state->idx < 4 && len)
        {
            state->U.buf[state->idx++] = *(data + a_index);
            a_index++;
            len--;
        }
            
        
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

    int i = 0;
    const int nblocks = len >> 2;
    const uint32_t * blocks = (const uint32_t *)(data + a_index);
    while (i < nblocks)
    {
        MURMUR3_BODY(blocks[i]);
        i++;
    }

    uint32_t offset = a_index + (nblocks * 4);
	while (offset < (len + a_index))
    {
        RH_ASSERT(state->idx < 4);
        state->U.buf[state->idx++] = *(data + offset++);
        
        if (state->idx == 4)
        {
            MURMUR3_BODY(state->U.buf32)
            state->idx = 0;
        }
    }

    state->h1 = h1;
}

inline uint32_t MurmurHash3_x86_32_Finalize( MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(state->idx != 0xDEADBEEF)

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

    h1 ^= state->totalLen;

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

inline uint32_t MurmurHash3_x86_32_Fast(const U8* key, int len)
{
    RH_ASSERT((size_t(key)% 8) == 0);
    uint32_t h1=0;

    S32 n = (len / sizeof(U64)) * sizeof(U64);
    U32 m = len % sizeof(U64);
    const U8* keyEnd = key + n;
    U64 r0; 
    while (key != keyEnd)
    {
        r0 = *(U64*)(key);
		key += sizeof(U64);
#if defined(RANDOMHASH_CUDA)
		RH_PREFETCH_MEM((const char*)key);
#endif
		MURMUR3_BODY((U32)(r0));
        MURMUR3_BODY((U32)(r0 >> 32));
    }

    if (m >= 4)
    {
        MURMUR3_BODY(*((U32*)key));
        key += 4;
    }

    uint32_t k1 = 0;
    switch (len & 3)
    {
    case 3: k1 ^= key[2] << 16;
    case 2: k1 ^= key[1] << 8;
    case 1: k1 ^= key[0];
        k1 *= MurmurHash3_x86_32_c1; 
        k1 = ROTL32(k1, 15); 
        k1 *= MurmurHash3_x86_32_c2; 
        h1 ^= k1;
    };
    
    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= MurmurHash3_x86_32_c4;
    h1 ^= h1 >> 13;
    h1 *= MurmurHash3_x86_32_c5;
    h1 ^= h1 >> 16;

    return h1;
}



#endif //RANDOM_HASH_MurMur3_32_h
