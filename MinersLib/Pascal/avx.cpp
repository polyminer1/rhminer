/**
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
/// @copyright Polyminer1, QualiaLibre

    
#include "precomp.h"
#include "MinersLib/Pascal/RandomHash.h"
#include "MinersLib/Pascal/RandomHash_core.h"

#if defined(RH_ENABLE_AVX) && 0

/*
    NOTE on AVX coding and why it's slower to use AVX instructions than SSE instructions. 
  
    In the case of RandomHash, the cpu gets throttled at the moment there is a non zero value, at the end of any ymmx registers.
    Rhminer could use some AVX instructions (see code in this file) to speed up some operations AND it could use the load/store to speed memory-ound operations (see code in this file).
    But doing this make the cpu being throttled 99.999% of the time because of the cpu ymmx being not zero all the time.
    AVX is not a fit for intensive work. Dont bother using it.
  
    The code here is just for studdy/test purose.
    Dont bother spending time on using AVX/AVX2.

    Polyminer1
*/

extern void RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41_2(U8* strideArray, U32 elementIdx, U8* strideArray2);
extern void RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41(U8* strideArray, U32 elementIdx);

uint32_t MurmurHash3_x86_32_Fast(const U8* key, int len)
{

    //const uint8_t * data = (const uint8_t*)key;
    RH_ASSERT((size_t(key) % 8) == 0);
    uint32_t h1 = 0;

    //----------
    // body
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

    //----------
    // tail / finish
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

    //----------
    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= MurmurHash3_x86_32_c4;
    h1 ^= h1 >> 13;
    h1 *= MurmurHash3_x86_32_c5;
    h1 ^= h1 >> 16;
    return h1;
}


void MurmurHash3_x86_32_Update_8(U64 chunk64, uint32_t len, MurmurHash3_x86_32_State* state)
{
    RH_ASSERT(len < S32_Max);
    RH_ASSERT(state->idx != 0xDEADBEEF)
        RH_ASSERT(len <= sizeof(U64));

    state->totalLen += len;
    uint32_t h1 = state->h1;
    int i = 0;

    //sonsume pending bytes
    if (state->idx)
    {
        while (len)
        {
            while (state->idx < 4 && len)       //TODO: optimiz - use switch case
            {
                U32 b = (U8)(chunk64 >> (i * 8)); //TODO: Manage endianness
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
        while (i < nblocks)     //TODO: optimiz - use switch case
        {
            U32 block = (U32)(chunk64 >> (i * 32));   //TODO: Manage endianness
            MURMUR3_BODY(block);
            i++;
            RH_ASSERT(i <= 2);
        }

        //save pending end bytes
        i = (nblocks * 4);
        while (i < (int)len)        //TODO: optimiz - use switch case
        {
            RH_ASSERT(state->idx < 4);
            U32 b = (U8)(chunk64 >> (i * 8)); //TODO: Manage endianness
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

//NOT WORKING, test purpose only
void Transfo0_2_16_AVX(U8* nextChunk, U32 size, U8* source)
{
    RHMINER_ASSERT(size <= 512);

    U32 rndState = MurmurHash3_x86_32_Fast(source, size);
    if (!rndState)
        rndState = 1;

#define RH_LD_LINE_256(r) r = _mm256_lddqu_si256((__m256i *)(source)); source += sizeof(__m256i);
#define _RH_LD_LINE_256(r) r = _mm256_lddqu_si256((__m256i *)(source)); 

    __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    switch (size / 32)
    {
    case 16:
    case 15: 
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); RH_LD_LINE_256(r9); RH_LD_LINE_256(r10); RH_LD_LINE_256(r11); RH_LD_LINE_256(r12); RH_LD_LINE_256(r13); RH_LD_LINE_256(r14); _RH_LD_LINE_256(r15);
        break;
    case 14:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); RH_LD_LINE_256(r9); RH_LD_LINE_256(r10); RH_LD_LINE_256(r11); RH_LD_LINE_256(r12); RH_LD_LINE_256(r13); _RH_LD_LINE_256(r14); 
        break;
    case 13:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); RH_LD_LINE_256(r9); RH_LD_LINE_256(r10); RH_LD_LINE_256(r11); RH_LD_LINE_256(r12); _RH_LD_LINE_256(r13);
        break;
    case 12:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); RH_LD_LINE_256(r9); RH_LD_LINE_256(r10); RH_LD_LINE_256(r11); _RH_LD_LINE_256(r12); 
        break;
    case 11:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); RH_LD_LINE_256(r9); RH_LD_LINE_256(r10); _RH_LD_LINE_256(r11); 
        break;
    case 10:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); RH_LD_LINE_256(r9); _RH_LD_LINE_256(r10);
        break;
    case 9:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); RH_LD_LINE_256(r8); _RH_LD_LINE_256(r9); 
        break;
    case 8:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); RH_LD_LINE_256(r7); _RH_LD_LINE_256(r8); 
        break;
    case 7:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); RH_LD_LINE_256(r6); _RH_LD_LINE_256(r7);
        break;
    case 6:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); RH_LD_LINE_256(r5); _RH_LD_LINE_256(r6); 
        break;
    case 5:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); RH_LD_LINE_256(r4); _RH_LD_LINE_256(r5); 
        break;
    case 4:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); RH_LD_LINE_256(r3); _RH_LD_LINE_256(r4); 
        break;
    case 3:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); RH_LD_LINE_256(r2); _RH_LD_LINE_256(r3); 
        break;
    case 2:
        RH_LD_LINE_256(r0); RH_LD_LINE_256(r1); _RH_LD_LINE_256(r2); 
        break;
    case 1:
        RH_LD_LINE_256(r0); _RH_LD_LINE_256(r1); 
        break;
    case 0:
        r0 = _mm256_lddqu_si256((__m256i *)(source));
        break;
    default: RHMINER_ASSERT(false);
    }

    U8* head = nextChunk;
    U8* end = head + size;
    //load work
    while (head < end)
    {
        uint32_t x = rndState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rndState = x;

        U32 d;
        //_mm256_extract_epi8

        // --- NOT TESTED ---
#define RH_GB256_4_AVX(r256, n)                                         \
        {                                                               \
            d = ((n) & 0x7)*8;                                          \
            switch((n)>>2)                                              \
            {                                                           \
                case 0:b = _mm256_extract_epi32(r256, 0)>>d; break;  \
                case 1:b = _mm256_extract_epi32(r256, 1)>>d; break;  \
                case 2:b = _mm256_extract_epi32(r256, 2)>>d; break;  \
                case 3:b = _mm256_extract_epi32(r256, 3)>>d; break;  \
                case 4:b = _mm256_extract_epi32(r256, 4)>>d; break;  \
                case 5:b = _mm256_extract_epi32(r256, 5)>>d; break;  \
                case 6:b = _mm256_extract_epi32(r256, 6)>>d; break;  \
                case 7:b = _mm256_extract_epi32(r256, 7)>>d; break;  \
                default:                                                \
                    RHMINER_ASSERT(false);                              \
            };                                                          \
            }

        U8 b;
        U32 val = x % size;
        U32 reg = val / 32;
        U32 n = val % 32;
        switch (reg)
        {
        case 15: RH_GB256_4_AVX(r15, n)  break;
        case 14: RH_GB256_4_AVX(r14, n)  break;
        case 13: RH_GB256_4_AVX(r13, n)  break;
        case 12: RH_GB256_4_AVX(r12, n)  break;
        case 11: RH_GB256_4_AVX(r11, n)  break;
        case 10: RH_GB256_4_AVX(r10, n)  break;
        case 9: RH_GB256_4_AVX(r9, n)  break;
        case 8: RH_GB256_4_AVX(r8, n)  break;
        case 7: RH_GB256_4_AVX(r7, n)  break;
        case 6: RH_GB256_4_AVX(r6, n)  break;
        case 5: RH_GB256_4_AVX(r5, n)  break;
        case 4: RH_GB256_4_AVX(r4, n)  break;
        case 3: RH_GB256_4_AVX(r3, n)  break;
        case 2: RH_GB256_4_AVX(r2, n)  break;
        case 1: RH_GB256_4_AVX(r1, n)  break;
        case 0: RH_GB256_4_AVX(r0, n)  break;
        default: RHMINER_ASSERT(false);
        }

        *head = b;
        head++;
    }
}


//much slower if used with sseoptimization 0
//NOT WORKING, test purpose only
void Transfo4_2_AVX(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ASSERT((size % 2) == 0);
    U32 halfSize = (size >> 1);

    __m128i const* left = (__m128i const*)outputPtr;
    __m128i const* right = (__m128i const*)(outputPtr + halfSize);
    __m256i* work = (__m256i*)nextChunk;
    __m256i v;
//  --- NOT TESTED, Bogus values  ---
    __m256i  ctrlMask = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);
    halfSize /= 32;
    while (halfSize)
    {
        //load v128 | v128
        v = _mm256_loadu2_m128i(left, right);
        right++;
        left++;

        //shuffle 256 bytes
        v = _mm256_shuffle_epi8(v, ctrlMask);
        //store work
        *work = v;
        work++;
        halfSize--;
    }
    /*
    RH_ASSERT((size % 2) == 0);
    U32 halfSize = size >> 1;

    U64 left = 0;
    U64 right = halfSize;
    U8* work = nextChunk;
    while (left < halfSize)
    {
        *work = outputPtr[right++];
        work++;
        *work = outputPtr[left++];
        work++;
    }
    */
}


void RH_STRIDE_ARRAY_UPDATE_MURMUR3_AVX2_2(U8* strideArray, U32 elementIdx, U8* strideArray2)
{
    RH_StridePtr lstridep = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstridep);

    if (size < sizeof(__m256i))
        return RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41_2(strideArray, elementIdx, strideArray2);

    U8* lstride = RH_STRIDE_GET_DATA(lstridep);

    MurmurHash3_x86_32_State* mm3_array2 = RH_StrideArrayStruct_GetAccum(strideArray2);
    MurmurHash3_x86_32_State* mm3_array1 = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array1->idx == 0);
    RH_ASSERT(mm3_array2->idx == 0);
    RH_ASSERT(mm3_array1->idx != 0xDEADBEEF);
    RH_ASSERT(mm3_array2->idx != 0xDEADBEEF);

    register U32 h1 = mm3_array1->h1;
    register U32 h2 = mm3_array2->h1;
    RH_ASSERT(((size_t)strideArray % 32) == 0);
    S32 n = (size / sizeof(__m256i)) * sizeof(__m256i);
    U32 m = size % sizeof(__m256i);
    U8* lstride_end = lstride + n;

    __m256i r0, r1;
    __m256i c1 = _mm256_set1_epi32(MurmurHash3_x86_32_c1);
    __m256i c2 = _mm256_set1_epi32(MurmurHash3_x86_32_c2);
    __m256i  perm_mask = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
    U32 r32;
    mm3_array1->totalLen += n;
    mm3_array2->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(__m256i*)(lstride);
        lstride += sizeof(__m256i);
        RH_PREFETCH_MEM((const char*)lstride);

        r0 = _mm256_mullo_epi32(r0, c1);    //avx2
        r1 = r0;
        r0 = _mm256_slli_epi32(r0, 15);    //avx2
        r1 = _mm256_srli_epi32(r1, 17);    //avx2
        r0 = _mm256_or_si256(r0, r1);       //avx2
        r0 = _mm256_mullo_epi32(r0, c2);    //avx2
        for (int i = 0; i < 8; i++)
        {
            r32 = _mm_cvtsi128_si32(_mm256_castsi256_si128(r0));
            //r32 = _mm256_cvtsi256_si32(r0);  //WTF. Slower than sse op code !?!?!
            RH_MURMUR3_BODY_2((U32)(r32), h1);
            RH_MURMUR3_BODY_2((U32)(r32), h2);
            r0 = _mm256_permutevar8x32_epi32(r0, perm_mask);    //avx2
        }
    }
    mm3_array1->h1 = h1;
    mm3_array2->h1 = h2;

    if (m)
    {
        U64 r0;
        S32 _m = (S32)m;
        while (_m > 0)  //TODO: optimiz - use switch case
        {
            r0 = *((U64 *)(lstride));

            U32 s = (_m >= sizeof(U64)) ? sizeof(U64) : (U32)_m;
            MurmurHash3_x86_32_Update_8(r0, s, mm3_array1);
            MurmurHash3_x86_32_Update_8(r0, s, mm3_array2);

            lstride += sizeof(U64);
            _m -= sizeof(U64);
        }
    }
}

void RH_STRIDE_ARRAY_UPDATE_MURMUR3_AVX2(U8* strideArray, U32 elementIdx)
{
    RH_StridePtr lstridep = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    U32 size = RH_STRIDE_GET_SIZE(lstridep);

    if (size < sizeof(__m256i))
        return RH_STRIDE_ARRAY_UPDATE_MURMUR3_SSE41(strideArray, elementIdx);

    U8* lstride = RH_STRIDE_GET_DATA(lstridep);

    MurmurHash3_x86_32_State* mm3_array = RH_StrideArrayStruct_GetAccum(strideArray);
    RH_ASSERT(mm3_array->idx != 0xDEADBEEF)
        RH_ASSERT(mm3_array->idx == 0);

    register U32 h1 = mm3_array->h1;
    RH_ASSERT(((size_t)strideArray % 32) == 0);
    S32 n = (size / sizeof(__m256i)) * sizeof(__m256i);
    U32 m = size % sizeof(__m256i);
    U8* lstride_end = lstride + n;
    __m256i r0, r1;
    __m256i c1 = _mm256_set1_epi32(MurmurHash3_x86_32_c1);
    __m256i c2 = _mm256_set1_epi32(MurmurHash3_x86_32_c2);
    __m256i perm_mask = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
    U32 r32;

    mm3_array->totalLen += n;
    while (lstride != lstride_end)
    {
        r0 = *(__m256i*)(lstride);
        lstride += sizeof(__m256i);
        RH_PREFETCH_MEM((const char*)lstride);

        r0 = _mm256_mullo_epi32(r0, c1);    //avx2
        r1 = r0;
        r0 = _mm256_slli_epi32(r0, 15);    //avx2
        r1 = _mm256_srli_epi32(r1, 17);    //avx2
        r0 = _mm256_or_si256(r0, r1);       //avx2
        r0 = _mm256_mullo_epi32(r0, c2);    //avx2
        for (int i = 0; i < 8; i++)
        {
            r32 = _mm_cvtsi128_si32(_mm256_castsi256_si128(r0));
            //r32 = _mm256_cvtsi256_si32(r0);  //WTF. Slower than sse op code !?!?!
            RH_MURMUR3_BODY_2((U32)(r32), h1);
            r0 = _mm256_permutevar8x32_epi32(r0, perm_mask);    //avx2
        }
    }

    mm3_array->h1 = h1;

    if (m)
    {
        U64 r0;
        S32 _m = (S32)m;
        while (_m > 0)  //TODO: optimiz - use switch case
        {
            r0 = *((U64 *)(lstride));

            U32 s = (_m >= sizeof(U64)) ? sizeof(U64) : (U32)_m;
            MurmurHash3_x86_32_Update_8(r0, s, mm3_array);

            lstride += sizeof(U64);
            _m -= sizeof(U64);
        }
    }
}

#endif //#ifdef RH_ENABLE_AVX
