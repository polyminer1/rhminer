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

#pragma once
#include "MinersLib/Pascal/RandomHash_def.h"

#ifdef RHMINER_PLATFORM_CPU

//IDEAL : src, dst and n are 32bytes aligned
inline void memcpy_uncached_load_sse41_SPILL(void *dest, const void *src, size_t n)
{
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;

    // align src to 128-bits
    if (s_int & 0xf) 
    {
        size_t nh = RH_Min(0x10 - (s_int & 0x0f), n);
        //memcpy(d, s, nh);
        *(U64*)d = *(U64*)s;
        if (nh > 8)
        {
            nh -= 8;
            d += 8; 
            d_int += 8;
            s += 8; 
            s_int += 8;
            n -= nh;
            *(U64*)d = *(U64*)s;
        }
        d += nh; 
        d_int += nh;
        s += nh; 
        s_int += nh;
        n -= nh;
    }

    if (d_int & 0xf)
    { 
        __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        while (n >= 8*sizeof(__m128i)) {
            r0 = RH_MM_LOAD128 ((__m128i *)(s));
            r1 = RH_MM_LOAD128 ((__m128i *)(s+1*sizeof(__m128i)));
            r2 = RH_MM_LOAD128 ((__m128i *)(s+2*sizeof(__m128i)));
            r3 = RH_MM_LOAD128 ((__m128i *)(s+3*sizeof(__m128i)));
            r4 = RH_MM_LOAD128 ((__m128i *)(s+4*sizeof(__m128i)));
            r5 = RH_MM_LOAD128 ((__m128i *)(s+5*sizeof(__m128i)));
            r6 = RH_MM_LOAD128 ((__m128i *)(s+6*sizeof(__m128i)));
            r7 = RH_MM_LOAD128 ((__m128i *)(s+7*sizeof(__m128i)));
            RH_MM_STORE128((__m128i *)(d), r0);
            RH_MM_STORE128((__m128i *)(d+1*sizeof(__m128i)), r1);
            RH_MM_STORE128((__m128i *)(d+2*sizeof(__m128i)), r2);
            RH_MM_STORE128((__m128i *)(d+3*sizeof(__m128i)), r3);
            RH_MM_STORE128((__m128i *)(d+4*sizeof(__m128i)), r4);
            RH_MM_STORE128((__m128i *)(d+5*sizeof(__m128i)), r5);
            RH_MM_STORE128((__m128i *)(d+6*sizeof(__m128i)), r6);
            RH_MM_STORE128((__m128i *)(d+7*sizeof(__m128i)), r7);
            s += 8*sizeof(__m128i);
            d += 8*sizeof(__m128i);
            n -= 8*sizeof(__m128i);
        }
        while (n >= sizeof(__m128i)) 
        {
            r0 = RH_MM_LOAD128 ((__m128i *)(s));
            RH_MM_STORE128((__m128i *)(d), r0);
            s += sizeof(__m128i);
            d += sizeof(__m128i);
            n -= sizeof(__m128i);
        }
    } 
    else 
    { 
        // or it IS aligned
        __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        while (n >= 8*sizeof(__m128i)) {
            r0 = RH_MM_LOAD128 ((__m128i *)(s));
            r1 = RH_MM_LOAD128 ((__m128i *)(s+1*sizeof(__m128i)));
            r2 = RH_MM_LOAD128 ((__m128i *)(s+2*sizeof(__m128i)));
            r3 = RH_MM_LOAD128 ((__m128i *)(s+3*sizeof(__m128i)));
            r4 = RH_MM_LOAD128 ((__m128i *)(s+4*sizeof(__m128i)));
            r5 = RH_MM_LOAD128 ((__m128i *)(s+5*sizeof(__m128i)));
            r6 = RH_MM_LOAD128 ((__m128i *)(s+6*sizeof(__m128i)));
            r7 = RH_MM_LOAD128 ((__m128i *)(s+7*sizeof(__m128i)));
            TH_MM_STREAM_STORE128((__m128i *)(d), r0);
            TH_MM_STREAM_STORE128((__m128i *)(d+1*sizeof(__m128i)), r1);
            TH_MM_STREAM_STORE128((__m128i *)(d+2*sizeof(__m128i)), r2);
            TH_MM_STREAM_STORE128((__m128i *)(d+3*sizeof(__m128i)), r3);
            TH_MM_STREAM_STORE128((__m128i *)(d+4*sizeof(__m128i)), r4);
            TH_MM_STREAM_STORE128((__m128i *)(d+5*sizeof(__m128i)), r5);
            TH_MM_STREAM_STORE128((__m128i *)(d+6*sizeof(__m128i)), r6);
            TH_MM_STREAM_STORE128((__m128i *)(d+7*sizeof(__m128i)), r7);
            s += 8*sizeof(__m128i);
            d += 8*sizeof(__m128i);
            n -= 8*sizeof(__m128i);
        }
        while (n >= sizeof(__m128i)) {
            r0 = RH_MM_LOAD128 ((__m128i *)(s));
            TH_MM_STREAM_STORE128((__m128i *)(d), r0);
            s += sizeof(__m128i);
            d += sizeof(__m128i);
            n -= sizeof(__m128i);
        }
    }

    if (n)
    {
        RH_ASSERT(n <= 16);
        *(U64*)d = *(U64*)s;
        if (n > 8)
        {
            d += 8;
            s += 8;
            *(U64*)d = *(U64*)s;
        }
    }

    // fencing because of NT stores
    // potential optimization: issue only when NT stores are actually emitted
    _mm_sfence();
}

#endif //CPU

#ifndef RANDOMHASH_CUDA

#define RH_INPLACE_MEMCPY_128(pDst, pSrc, byteCount)                    \
    {S32 n = RHMINER_CEIL(byteCount, sizeof(__m128i));                     \
    __m128i r0;                                                         \
    while (n >= sizeof(__m128i))                                        \
    {                                                                   \
        r0 = RH_MM_LOAD128 ((__m128i *)(pSrc));       \
        RH_MM_STORE128((__m128i *)(pDst), r0);        \
        pSrc += sizeof(__m128i);                                        \
        pDst += sizeof(__m128i);                                        \
        n -= sizeof(__m128i);                                           \
    }                                                                   \
    RH_MM_BARRIER();}

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_INPLACE_MEMCPY_128_A)(U8* pDst, U8* pSrc, U32 byteCount, MurmurHash3_x86_32_State* accum)
{
    RH_ASSERT(( (size_t)pDst % 8) == 0);
    RH_ASSERT(( (size_t)pSrc % 8) == 0);

    S32 n = (byteCount / sizeof(__m128i)) * sizeof(__m128i);
    U32 m = byteCount % sizeof(__m128i);
    //RH_MUR3_BACKUP_STATE(accum);
    __m128i r0;
    while (n > 0)
    {
        r0 = RH_MM_LOAD128((__m128i *)(pSrc ));
        MurmurHash3_x86_32_Update_16(r0, 16, accum);   //slightly faster on pc 
        //INPLACE_M_MurmurHash3_x86_32_Update_16(r0, 16); 
        RH_MM_STORE128((__m128i *)(pDst ), r0);
        pSrc += sizeof(__m128i);
        pDst += sizeof(__m128i);
        n -= sizeof(__m128i);
    }
    if (m)
    {
        r0 = RH_MM_LOAD128((__m128i *)(pSrc));
        RH_MM_STORE128((__m128i *)(pDst ), r0);
        MurmurHash3_x86_32_Update_16(r0, m, accum); //slightly faster on pc 
        //INPLACE_M_MurmurHash3_x86_32_Update_16(r0, m);  
    }
    RH_MM_BARRIER();
    //RH_MUR3_RESTORE_STATE(accum);
}

#else //!CPU

//TODO: Optmiz - Test memcpy
#define RH_INPLACE_MEMCPY_128(pDst, pSrc, byteCount)                    \
    {U8* end = pDst + byteCount;                                        \
    while(pDst < end)                                                   \
    {                                                                   \
        *(uint4 *)pDst = *(uint4 *)pSrc;                                \
        pDst += 16;                                                     \
        pSrc += 16;                                                     \
    }}


CUDA_DECL_DEVICE
void CUDA_SYM(RH_INPLACE_MEMCPY_128_A)(U8* pDst, U8* pSrc, U32 byteCount, MurmurHash3_x86_32_State* accum)
{
    S32 n = RHMINER_FLOOR(byteCount, sizeof(uint4));
    uint4 data;
    while (n >= sizeof(uint4))
    {
        data = *(uint4 *)pSrc;
        _CM(MurmurHash3_x86_32_Update_16)(data, 16, accum);
        *(uint4 *)pDst = data;
        pDst += sizeof(uint4);
        pSrc += sizeof(uint4);
        n -= sizeof(uint4);
    }
    data = *(uint4 *)pSrc;
    _CM(MurmurHash3_x86_32_Update_16)(data, byteCount % 16, accum);
    *(uint4 *)pDst = data;
}

#endif //CPU

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDE_MEMCPY_ALIGNED_SIZE128)(U8 *pDst, U8 *pSrc, size_t byteCount)
{
    RH_ASSERT(( (size_t)pDst % 8) == 0);
    RH_ASSERT(( (size_t)pSrc % 8) == 0);
    RH_ASSERT(( (size_t)pDst % 32) == 0);
    RH_ASSERT(( (size_t)pSrc % 32) == 0);

    RH_INPLACE_MEMCPY_128(pDst, pSrc, byteCount);
}

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(U8 *pDst, U8 *pSrc, size_t byteCount)
{
    RH_ASSERT(byteCount < RH_StrideSize - 256); 

#ifndef __CUDA_ARCH__
    size_t s = byteCount;
    memcpy_uncached_load_sse41_SPILL(pDst, pSrc, s); //NOT ALL ALIGNED !
    return;
#endif

    //CUDA and PC
    if (((size_t)pDst)%8 == 0 &&
        ((size_t)pSrc)%8 == 0)
    {
        U8* end = pDst + byteCount;
        while(pDst < end)
        {
            *(U64*)pDst = *(U64*)pSrc;
            pDst += 8;
            pSrc += 8;
        }
    }
    else
    {
        memcpy((void*)(pDst), (void*)(pSrc), byteCount); 
    }
}

#ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)(U8* strideArray, U32 elementIdx)
{
    RH_ASSERT(( (size_t)strideArray % 8) == 0);

    RH_StridePtr lstride = RH_STRIDEARRAY_GET(strideArray, elementIdx);
    S32 n = (RH_STRIDE_GET_SIZE(lstride) / sizeof(U64)) * sizeof(U64);
    U32 m = RH_STRIDE_GET_SIZE(lstride) % sizeof(U64);
    lstride = RH_STRIDE_GET_DATA(lstride);
    RH_StridePtr lstride_end = lstride + n;
    U64 r0;
    RH_MUR3_BACKUP_STATE(RH_StrideArrayStruct_GetAccum(strideArray));
    back_totalLen += n;
    register uint32_t h1 = back_h1;                                   
    while (lstride != lstride_end)
    {
        r0 = *(U64*)(lstride);
        RH_ASSERT(back_idx != 0xDEADBEEF)                        
        back_i = 0;                                              
        if (back_idx)                                            
        { 
            register U32 len = 8;                                          
            while(len)                                           
            {                                                    
                while (back_idx < 4 && len)                      
                {                                                
                    U32 b = (U8)(r0 >> (back_i*8));         
                    back_buf &= ~(0xFF << (back_idx*8));         
                    back_buf |= (b << (back_idx*8));             
                    back_idx++;                                  
                    len--;                                       
                    back_i++;                                    
                }                                                
                if (back_idx == 4)                               
                {                                                
                    MURMUR3_BODY(back_buf)                       
                    back_idx = 0;                                
                }                                                
            }                                                    
        }                                                        
        else                                                     
        {             
            U32 block;
            //1
            block = (U32)(r0);       
            MURMUR3_BODY(block);                             
            //2
            block = (U32)(r0 >> 32);       
            MURMUR3_BODY(block);                             
        }                                                        
                                                                     
        lstride += sizeof(U64);        
    }
    back_h1 = h1;                                            

    if (m)
    {
        r0 = *((U64 *)(lstride));
        INPLACE_M_MurmurHash3_x86_32_Update_8(r0, m); 
    }
    
    RH_MUR3_RESTORE_STATE(RH_StrideArrayStruct_GetAccum(strideArray));
}


CUDA_DECL_DEVICE
void CUDA_SYM(RH_STRIDEARRAY_PUSHBACK_MANY)(U8* strideArrayVar, U8* strideArrayVarSrc)
{
    RH_ASSERT(strideArrayVar != strideArrayVarSrc);
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVar) <= RH_StrideArrayCount);
    if (RH_STRIDEARRAY_GET_SIZE(strideArrayVar) == 0)
    {
        *RH_StrideArrayStruct_GetAccum(strideArrayVar) = *RH_StrideArrayStruct_GetAccum(strideArrayVarSrc);
        RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc)
        {
            RH_STRIDEARRAY_PUSHBACK_NO_ACCUM(strideArrayVar, strideItrator);
        }
        RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc)
    }
    else
    {
        RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc)
        {
            RH_STRIDEARRAY_PUSHBACK(strideArrayVar, strideItrator);
            _CM(RH_STRIDE_ARRAY_UPDATE_MURMUR3)((strideArrayVar), RH_STRIDEARRAY_GET_SIZE(strideArrayVar)-1);
        }
        RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc)
    }
}


#else

CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDEARRAY_PUSHBACK_MANY)(U8* strideArrayVar, U8* strideArrayVarSrc)
{                                                                               
    RH_ASSERT(strideArrayVar != strideArrayVarSrc);                             
    RH_ASSERT(RH_STRIDEARRAY_GET_SIZE(strideArrayVar) <= RH_StrideArrayCount);  
    RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc)                            
    {                                                                           
        RH_STRIDEARRAY_PUSHBACK(strideArrayVar, strideItrator);                 
    }                                                                           
    RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc)                              
} 
#endif


CUDA_DECL_DEVICE
inline void CUDA_SYM(RH_STRIDEARRAY_CLONE)(U8* strideArrayVar, U8* strideArrayVarSrc, RandomHash_State* state)
{                                                                               
    RH_StridePtrArray target = strideArrayVar;
    RH_STRIDEARRAY_RESET(target);
    RH_STRIDEARRAY_FOR_EACH_BEGIN(strideArrayVarSrc)
    {
        U32 _as = RH_STRIDEARRAY_GET_SIZE(target)++;
        RH_STRIDEARRAY_GET(target, _as) = _CM(RH_StrideArrayAllocOutput)(state);
        RH_StridePtr elem = RH_STRIDEARRAY_GET(target, _as);
        RH_STRIDE_COPY(elem, (strideItrator));
    }
    RH_STRIDEARRAY_FOR_EACH_END(strideArrayVarSrc)

#ifdef RH_ENABLE_OPTIM_STRIDE_ARRAY_MURMUR3
        *RH_StrideArrayStruct_GetAccum(target) = *RH_StrideArrayStruct_GetAccum(strideArrayVarSrc);
#endif
} 


//------------------------------------------------------------------------
//accum 8 code

#define RH_Accum_8() \
    U32 acc8_idx = 0;  \
    U64 acc8_buf = 0;  \

#define RH_Accum_8_Reset(p) \
    acc8_idx = 0;  \
    acc8_buf = 0;  \

#define RH_Accum_8_Add(chunk8, acc8_ptr)                    \
{                                                           \
    acc8_buf &= ~(0xFFLLU << (U64)(acc8_idx << 3));         \
    acc8_buf |= ((U64)(chunk8) << (U64)(acc8_idx << 3));    \
    acc8_idx++;                                             \
    if (acc8_idx == 8)                                      \
    {                                                       \
        *((U64*)acc8_ptr) = acc8_buf;                       \
        acc8_ptr += 8;                                      \
        acc8_idx = 0;                                       \
        acc8_buf = 0;                                       \
    }                                                       \
}


#define RH_Accum_8_Finish(acc8_ptr)      \
{                                       \
    RH_ASSERT(acc8_idx != 8);           \
    *((U64*)acc8_ptr) = acc8_buf;       \
}

extern bool g_isSSE41Supported;
inline void CUDA_SYM_DECL(Transfo0)(U8* nextChunk, U32 size, U8* workBytes)
{
    U32 rndState = _CM(MurmurHash3_x86_32_Fast)((const void *)nextChunk,size, 0);
    if (!rndState)
        rndState = 1;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(workBytes, nextChunk, size);

    //RH_ASSERT(((size_t)(nextChunk) % 8) == 0);
    RH_Accum_8();
    U8* end = nextChunk + size;
    //Align chunkPtr
    while(((size_t)nextChunk % 8) != 0 && (nextChunk < end))
    {
        uint32_t x = rndState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rndState = x;
        *nextChunk = workBytes[x % size];
        nextChunk++;
    }
    while(nextChunk < end)
    {
        uint32_t x = rndState;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        rndState = x;
        RH_Accum_8_Add(workBytes[x % size], nextChunk);
    }
    RH_Accum_8_Finish(nextChunk);
#else
#ifdef RHMINER_PLATFORM_CPU
    if (size <= 128 && (size_t(nextChunk)&0x0f)==0 && g_isSSE41Supported)
    {
        //load in mmx reg
        __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        switch(size/16)
        {
            case 8:
            case 7: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(nextChunk+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(nextChunk+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(nextChunk+4*sizeof(__m128i)));
                    r5 = RH_MM_LOAD128 ((__m128i *)(nextChunk+5*sizeof(__m128i)));
                    r6 = RH_MM_LOAD128 ((__m128i *)(nextChunk+6*sizeof(__m128i)));
                    r7 = RH_MM_LOAD128 ((__m128i *)(nextChunk+7*sizeof(__m128i)));
                    break;
            case 6: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(nextChunk+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(nextChunk+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(nextChunk+4*sizeof(__m128i)));
                    r5 = RH_MM_LOAD128 ((__m128i *)(nextChunk+5*sizeof(__m128i)));
                    r6 = RH_MM_LOAD128 ((__m128i *)(nextChunk+6*sizeof(__m128i)));
                    break;
            case 5: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(nextChunk+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(nextChunk+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(nextChunk+4*sizeof(__m128i)));
                    r5 = RH_MM_LOAD128 ((__m128i *)(nextChunk+5*sizeof(__m128i)));
                    break;
            case 4: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(nextChunk+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(nextChunk+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(nextChunk+4*sizeof(__m128i)));
                    break;
            case 3: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(nextChunk+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(nextChunk+3*sizeof(__m128i)));
                    break;
            case 2: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(nextChunk+2*sizeof(__m128i)));
                    break;
            case 1: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(nextChunk+1*sizeof(__m128i)));
                    break;
            case 0: r0 = RH_MM_LOAD128 ((__m128i *)(nextChunk+0*sizeof(__m128i)));
                    break;
            default: RHMINER_ASSERT(false);
        }

        U8* head = nextChunk;
        U8* end = head + size;
        //load work
        while(head < end)
        {
            uint32_t x = rndState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rndState = x;
            U32 d;
            #define RH_GB128(chunk128, n)                                         \
                {                                                               \
                    d = ((n) & 0x7)*8;                                          \
                    switch((n)>>2)                                              \
                    {                                                           \
                        case 0:b = _mm_extract_epi32_M(chunk128, 0)>>d; break;  \
                        case 1:b = _mm_extract_epi32_M(chunk128, 1)>>d; break;  \
                        case 2:b = _mm_extract_epi32_M(chunk128, 2)>>d; break;  \
                        case 3:b = _mm_extract_epi32_M(chunk128, 3)>>d; break;  \
                        default:                                                \
                            RHMINER_ASSERT(false);                              \
                    };                                                          \
                }

            U8 b;
            U32 val = x % size;
            U32 reg = val / 16;
            U32 n = val % 16;
            switch(reg)
            {
                case 7: RH_GB128(r7, n)  break;
                case 6: RH_GB128(r6, n)  break;
                case 5: RH_GB128(r5, n)  break;
                case 4: RH_GB128(r4, n)  break;
                case 3: RH_GB128(r3, n)  break;
                case 2: RH_GB128(r2, n)  break;
                case 1: RH_GB128(r1, n)  break;
                case 0: RH_GB128(r0, n)  break;
                default: RHMINER_ASSERT(false);
            }
        
            *head = b;
            head++;
        }
    }
    else
#endif //#ifdef RHMINER_PLATFORM_CPU
    {
        _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(&workBytes[0], nextChunk, size);
        U8* head = nextChunk;
        U8* end = head + size;
        while(head < end)
        {
            uint32_t x = rndState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rndState = x;
            *head = workBytes[x % size];
            head++;
        }
    }

#endif
}

inline void CUDA_SYM_DECL(Transfo0_2)(U8* nextChunk, U32 size, U8* source)
{
    U32 rndState = _CM(MurmurHash3_x86_32_Fast)((const void *)source,size, 0);
    if (!rndState)
        rndState = 1;
#ifdef RHMINER_PLATFORM_CPU
    if (size <= 128 && (size_t(source)&0x0f)==0 && g_isSSE41Supported)
    {
        //load in mmx reg
        __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        switch(size/16)
        {
            case 8:
            case 7: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                    r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                    r6 = RH_MM_LOAD128 ((__m128i *)(source+6*sizeof(__m128i)));
                    r7 = RH_MM_LOAD128 ((__m128i *)(source+7*sizeof(__m128i)));
                    break;
            case 6: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                    r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                    r6 = RH_MM_LOAD128 ((__m128i *)(source+6*sizeof(__m128i)));
                    break;
            case 5: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                    r5 = RH_MM_LOAD128 ((__m128i *)(source+5*sizeof(__m128i)));
                    break;
            case 4: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                    r4 = RH_MM_LOAD128 ((__m128i *)(source+4*sizeof(__m128i)));
                    break;
            case 3: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                    r3 = RH_MM_LOAD128 ((__m128i *)(source+3*sizeof(__m128i)));
                    break;
            case 2: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    r2 = RH_MM_LOAD128 ((__m128i *)(source+2*sizeof(__m128i)));
                    break;
            case 1: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    r1 = RH_MM_LOAD128 ((__m128i *)(source+1*sizeof(__m128i)));
                    break;
            case 0: r0 = RH_MM_LOAD128 ((__m128i *)(source+0*sizeof(__m128i)));
                    break;
            default: RHMINER_ASSERT(false);
        }

        U8* head = nextChunk;
        U8* end = head + size;
        //load work
        while(head < end)
        {
            uint32_t x = rndState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rndState = x;
            U32 d;
            #define RH_GB128(chunk128, n)                                         \
                {                                                               \
                    d = ((n) & 0x7)*8;                                          \
                    switch((n)>>2)                                              \
                    {                                                           \
                        case 0:b = _mm_extract_epi32_M(chunk128, 0)>>d; break;  \
                        case 1:b = _mm_extract_epi32_M(chunk128, 1)>>d; break;  \
                        case 2:b = _mm_extract_epi32_M(chunk128, 2)>>d; break;  \
                        case 3:b = _mm_extract_epi32_M(chunk128, 3)>>d; break;  \
                        default:                                                \
                            RHMINER_ASSERT(false);                              \
                    };                                                          \
                }

            U8 b;
            U32 val = x % size;
            U32 reg = val / 16;
            U32 n = val % 16;
            switch(reg)
            {
                case 7: RH_GB128(r7, n)  break;
                case 6: RH_GB128(r6, n)  break;
                case 5: RH_GB128(r5, n)  break;
                case 4: RH_GB128(r4, n)  break;
                case 3: RH_GB128(r3, n)  break;
                case 2: RH_GB128(r2, n)  break;
                case 1: RH_GB128(r1, n)  break;
                case 0: RH_GB128(r0, n)  break;
                default: RHMINER_ASSERT(false);
            }
        
            *head = b;
            head++;
        }
    }
    else
#endif //#ifdef RHMINER_PLATFORM_CPU
    {
        U8* head = nextChunk;
        U8* end = head + size;
        while(head < end)
        {
            uint32_t x = rndState;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rndState = x;
            *head = source[x % size];
            head++;
        }
    }
}

inline void CUDA_SYM_DECL(Transfo1)(U8* nextChunk, U32 size, U8* workBytes)
{
    U32 halfSize = size >> 1;
    U32 sizeIsOdd = size % 2;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(workBytes, nextChunk, halfSize);

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, nextChunk + halfSize + sizeIsOdd, halfSize);
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk + halfSize + sizeIsOdd, workBytes, halfSize);
    if (sizeIsOdd)
        nextChunk[halfSize] = workBytes[halfSize];
#else
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(&workBytes[0], nextChunk, halfSize);

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, nextChunk + halfSize + sizeIsOdd, halfSize);
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk + halfSize + sizeIsOdd, &workBytes[0], halfSize);
    if (sizeIsOdd)
        nextChunk[halfSize] = workBytes[halfSize];
#endif
}

inline void CUDA_SYM_DECL(Transfo1_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 halfSize = size >> 1;
    U32 sizeIsOdd = size % 2;

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, outputPtr + halfSize + sizeIsOdd, halfSize);
    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk + halfSize + sizeIsOdd, outputPtr, halfSize);
    if (sizeIsOdd)
        nextChunk[halfSize] = outputPtr[halfSize];

}

inline void CUDA_SYM_DECL(Transfo2)(U8* nextChunk, U32 size, U8* workBytes)
{
    U32 halfSize = size >> 1;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    
    U8* head = nextChunk;
    U8* end = head + halfSize;
    U8* tail = &nextChunk[size - 1];
    while(head < end)
    {
        U8 b = *head;
        *head = *tail;
        *tail = b;
        head++;
        tail--;
    }
#else
    
    U8 b;
    U8* head = nextChunk;
    U8* end = head + halfSize;
    U8* tail = &nextChunk[size - 1];
    while(head < end)
    {
        b = *head;
        *head = *tail;
        *tail = b;
        head++;
        tail--;
    }
#endif
}

inline void CUDA_SYM_DECL(Transfo2_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 halfSize = size >> 1;
    
    U8* srcHead = outputPtr;
    U8* srcEnd = srcHead + halfSize;
    U8* srcTail = &outputPtr[size - 1];
    U8* tail = &nextChunk[size - 1];
    U8* head = nextChunk;
    while(srcHead < srcEnd)
    {
        *head = *srcTail;
        *tail = *srcHead;
        head++;
        tail--;
        srcHead++;
        srcTail--;
    }
}



inline void CUDA_SYM_DECL(Transfo3)(U8* nextChunk, U32 size, U8* workBytes)
{
    U32 sizeIsOdd = size % 2;
    U32 halfSize = size >> 1;
    int left = 0;
    int right = (int)halfSize + sizeIsOdd;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    U8* accum = workBytes;
    RH_Accum_8();
    //RH_ASSERT(((size_t)(workBytes) % 8) == 0);
    while(left < halfSize)
    {
        RH_Accum_8_Add(nextChunk[left++], accum);
        RH_Accum_8_Add(nextChunk[right++], accum);
    }
    if (sizeIsOdd)
    {
        RH_Accum_8_Add(nextChunk[halfSize], accum);
    }
    RH_Accum_8_Finish(accum);

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, workBytes, size);

    RH_ASSERT(size < RH_StrideSize);                
#else
    U8* work = workBytes;
    while(left < halfSize)
    {
        *work = nextChunk[left++];
        work++;
        *work = nextChunk[right++];
        work++;
    }
    if (sizeIsOdd)
        *work = nextChunk[halfSize];

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, workBytes, size);

    RH_ASSERT(size < RH_StrideSize);                

#endif
}


inline void CUDA_SYM_DECL(Transfo3_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 sizeIsOdd = size % 2;
    U32 halfSize = size >> 1;
    int left = 0;
    int right = (int)halfSize + sizeIsOdd;
    U8* work = nextChunk;
    while(left < halfSize)
    {
        *work = outputPtr[left++];
        work++;
        *work = outputPtr[right++];
        work++;
    }
    if (sizeIsOdd)
        *work = outputPtr[halfSize];

    RH_ASSERT(size < RH_StrideSize);                
}

inline void CUDA_SYM_DECL(Transfo4)(U8* nextChunk, U32 size, U8* workBytes)
{
    RH_ALIGN(64) U32 sizeIsOdd = size % 2;
    RH_ALIGN(64) U32 halfSize = size >> 1;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    int left = 0;
    int right = halfSize + sizeIsOdd;
    U8* work = workBytes;
    //RH_ASSERT(((size_t)(workBytes) % 8) == 0);
    RH_Accum_8();
    while(left < halfSize)
    {
        RH_Accum_8_Add(nextChunk[right++], work);
        RH_Accum_8_Add(nextChunk[left++], work);
    }
    if (sizeIsOdd)
    {
        RH_Accum_8_Add(nextChunk[halfSize], work);
    }
    RH_Accum_8_Finish(work);

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, workBytes, size);

    RH_ASSERT(size < RH_StrideSize);
#else

    int left = 0;
    int right = halfSize + sizeIsOdd;
    U8* work = workBytes;
    while(left < halfSize)
    {
        *work = nextChunk[right++];
        work++;
        *work = nextChunk[left++];
        work++;
    }
    if (sizeIsOdd)
        *work = nextChunk[halfSize];

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(nextChunk, workBytes, size);

    RH_ASSERT(size < RH_StrideSize);
#endif
}

inline void CUDA_SYM_DECL(Transfo4_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    RH_ALIGN(64) U32 sizeIsOdd = size % 2;
    RH_ALIGN(64) U32 halfSize = size >> 1;

    int left = 0;
    int right = halfSize + sizeIsOdd;
    U8* work = nextChunk;
    while(left < halfSize)
    {
        *work = outputPtr[right++];
        work++;
        *work = outputPtr[left++];
        work++;
    }
    if (sizeIsOdd)
        *work = outputPtr[halfSize];
}


inline void CUDA_SYM_DECL(Transfo5)(U8* nextChunk, U32 size, U8* workBytes)
{
    U32 sizeIsOdd = size % 2;
    U32 halfSize = size >> 1;
    U32 i = 0;
    S32 itt = 0;
    S32 ritt = size-1;

    //workBytes = nextChunk;
    RH_ASSERT(size < RH_StrideSize);

    _CM(RH_STRIDE_MEMCPY_UNALIGNED_SIZE8)(workBytes, nextChunk, size);
    
    while(i < halfSize)
    {
        //first half
        nextChunk[i] = workBytes[itt] ^ workBytes[itt + 1];
        itt += 2;
        //second half
        nextChunk[i+halfSize + sizeIsOdd] = workBytes[i] ^ workBytes[ritt--];
        i++;
    }

    if (sizeIsOdd)
        nextChunk[halfSize] = workBytes[size-1];
}

inline void CUDA_SYM_DECL(Transfo5_2)(U8* nextChunk, U32 size, U8* outputPtr)
{
    U32 sizeIsOdd = size % 2;
    U32 halfSize = size >> 1;
    U32 i = 0;
    S32 itt = 0;
    S32 ritt = size-1;
    
    while(i < halfSize)
    {
        //first half
        nextChunk[i] = outputPtr[itt] ^ outputPtr[itt + 1];
        itt += 2;
        //second half
        nextChunk[i+halfSize + sizeIsOdd] = outputPtr[i] ^ outputPtr[ritt--];
        i++;
    }

    if (sizeIsOdd)
        nextChunk[halfSize] = outputPtr[size-1];
}

inline void CUDA_SYM_DECL(Transfo6)(U8* nextChunk, U32 size)
{
    RH_ALIGN(64) U32 i = 0;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    
    U8* work = nextChunk;
    //Align chunkPtr
    while(((size_t)work % 8) != 0 && (i < size))
    {
        *work = ROTL8(*work, size-i);
        i++;
        work++;
    }
    RH_Accum_8();
    while(i < size)
    {
        //nextChunk[i] = ROTL8(nextChunk[i], size-i);
        RH_Accum_8_Add((U8)ROTL8(nextChunk[i], size-i), work);
        i++;
    }
    RH_Accum_8_Finish(work);
#else

    /*
    while(i < size)
    {
        nextChunk[i] = ROTL8(nextChunk[i], size-i);
        i++;
    }
    */
    U8* end = nextChunk + (size >> 3)*8;
    while(((size_t)nextChunk % 8) != 0 && (i < size))
    {
        *nextChunk = ROTL8(*nextChunk, size-i);
        i++;
        nextChunk++;
    }
    while(nextChunk < end)
    {
        RH_ALIGN(64) U8 b;
        RH_ALIGN(64) U64 res = 0;
        RH_ALIGN(64) U64 buf = *(U64*)nextChunk;
        RH_ALIGN(64) U32 localSize = size - i;
        
        //UNROLL
        b = (U8)(buf);
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b));
        b = (U8)(buf >> (1<<3));
        b = ROTL8(b, localSize); localSize--; 
        res |= ((U64)(b) << (U64)(1 << 3));
        b = (U8)(buf >> (2<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(2 << 3));
        b = (U8)(buf >> (3<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(3 << 3));
        b = (U8)(buf >> (4<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(4 << 3));
        b = (U8)(buf >> (5<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(5 << 3));
        b = (U8)(buf >> (6<<3));
        b = ROTL8(b, localSize); localSize--; 
        res |= ((U64)(b) << (U64)(6 << 3));
        b = (U8)(buf >> (7<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(7 << 3));
        i += 8;
        *(U64*)nextChunk = res;
        nextChunk += 8;
    }
    while(i < size)
    {
        *nextChunk = ROTL8(*nextChunk, size-i);
        i++;
        nextChunk++;
    }
#endif
}


inline void CUDA_SYM_DECL(Transfo6_2)(U8* nextChunk, U32 size, U8* source)
{
    RH_ALIGN(64) U32 i = 0;
#ifdef __CUDA_ARCH__
    while(i < size)
    {
        nextChunk[i] = ROTL8(source[i], size-i);
        i++;
    }
#else
    U8* end = nextChunk + (size >> 3)*8;
    while(((size_t)source % 8) != 0 && (i < size))
    {
        *nextChunk = ROTL8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
    while(nextChunk < end)
    {
        U8 b;
        U64 res = 0;
        U64 buf = *(U64*)source;
        U32 localSize = size - i;
        
        //UNROLL
        b = (U8)(buf);
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b));
        b = (U8)(buf >> (1<<3));
        b = ROTL8(b, localSize); localSize--; 
        res |= ((U64)(b) << (U64)(1 << 3));
        b = (U8)(buf >> (2<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(2 << 3));
        b = (U8)(buf >> (3<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(3 << 3));
        b = (U8)(buf >> (4<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(4 << 3));
        b = (U8)(buf >> (5<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(5 << 3));
        b = (U8)(buf >> (6<<3));
        b = ROTL8(b, localSize); localSize--; 
        res |= ((U64)(b) << (U64)(6 << 3));
        b = (U8)(buf >> (7<<3));
        b = ROTL8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(7 << 3));
        i += 8;
        *(U64*)nextChunk = res;
        nextChunk += 8;
        source += 8;
    }
    while(i < size)
    {
        *nextChunk = ROTL8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
#endif //cudaarch
}

inline void CUDA_SYM_DECL(Transfo7)(U8* nextChunk, U32 size)
{
    RH_ALIGN(64) U32 i = 0;
#ifdef RH_ENABLE_OPTIM_EXPAND_ACCUM8
    U8* work = nextChunk;
    //Align chunkPtr
    while(((size_t)work % 8) != 0 && (i < size))
    {
        *work = ROTR8(*work, size-i);
        i++;
        work++;
    }
    RH_Accum_8();
    while(i < size)
    {
        //nextChunk[i] = ROTR8(nextChunk[i], size-i);
        RH_Accum_8_Add((U8)ROTR8(nextChunk[i], size-i), work);
        i++;
    }
    RH_Accum_8_Finish(work);
#else
    /*
    while(i < size)
    {
        nextChunk[i] = ROTR8(nextChunk[i], size-i);
        i++;
    }
    */
    U8* end = nextChunk + (size >> 3)*8;
    while(((size_t)nextChunk % 8) != 0 && (i < size))
    {
        *nextChunk = ROTR8(*nextChunk, size-i);
        i++;
        nextChunk++;
    }
    while(nextChunk < end)
    {
        RH_ALIGN(64) U8 b;
        RH_ALIGN(64) U64 res = 0;
        RH_ALIGN(64) U64 buf = *(U64*)nextChunk;
        RH_ALIGN(64) U32 localSize = size - i;

        //UNROLL
        b = (U8)(buf);
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b));
        b = (U8)(buf >> (1<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(1 << 3));
        b = (U8)(buf >> (2<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(2 << 3));
        b = (U8)(buf >> (3<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(3 << 3));
        b = (U8)(buf >> (4<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(4 << 3));
        b = (U8)(buf >> (5<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(5 << 3));
        b = (U8)(buf >> (6<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(6 << 3));
        b = (U8)(buf >> (7<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(7 << 3));
        i += 8;
        *(U64*)nextChunk = res;
        nextChunk += 8;
    }
    while(i < size)
    {
        *nextChunk = ROTR8(*nextChunk, size-i);
        i++;
        nextChunk++;
    }
#endif
}

inline void CUDA_SYM_DECL(Transfo7_2)(U8* nextChunk, U32 size, U8* source)
{
    RH_ALIGN(64) U32 i = 0;
#ifdef __CUDA_ARCH__
    while(i < size)
    {
        nextChunk[i] = ROTR8(source[i], size-i);
        i++;
    }
#else
    U8* end = nextChunk + (size >> 3)*8;
    while(((size_t)source % 8) != 0 && (i < size))
    {
        *nextChunk = ROTR8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
    while(nextChunk < end)
    {
        U8 b;
        U64 res = 0;
        U64 buf = *(U64*)source;
        U32 localSize = size - i;

        //UNROLL
        b = (U8)(buf);
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b));
        b = (U8)(buf >> (1<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(1 << 3));
        b = (U8)(buf >> (2<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(2 << 3));
        b = (U8)(buf >> (3<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(3 << 3));
        b = (U8)(buf >> (4<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(4 << 3));
        b = (U8)(buf >> (5<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(5 << 3));
        b = (U8)(buf >> (6<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(6 << 3));
        b = (U8)(buf >> (7<<3));
        b = ROTR8(b, localSize); localSize--;
        res |= ((U64)(b) << (U64)(7 << 3));
        i += 8;
        *(U64*)nextChunk = res;
        nextChunk += 8;
        source += 8;
    }
    while(i < size)
    {
        *nextChunk = ROTR8(*source, size-i);
        i++;
        nextChunk++;
        source++;
    }
#endif //#ifdef __CUDA_ARCH__
}

