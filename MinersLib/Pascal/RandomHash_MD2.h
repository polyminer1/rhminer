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


#include "RandomHash_core.h"

#ifndef RHMINER_PLATFORM_GPU
    #define UINT4 uint32_t
#elif defined(RHMINER_PLATFORM_CUDA)
    #define UINT4 uint32_t
#else
error
#endif


#define MD2_BLOCK_SIZE 16

RH_ALIGN(128) static const unsigned char md2_s[] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54,
    0xa1, 0xec, 0xf0, 0x06, 0x13, 0x62, 0xa7, 0x05, 0xf3, 0xc0, 0xc7,
    0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca, 0x1e,
    0x9b, 0x57, 0x3c, 0xfd, 0xd4, 0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18,
    0x8a, 0x17, 0xe5, 0x12, 0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde,
    0x49, 0xa0, 0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a, 0xa9, 0x68,
    0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89, 0x0b,
    0x22, 0x5f, 0x21, 0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27,
    0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97, 0x03, 0xff, 0x19, 0x30,
    0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56,
    0xaa, 0xc6, 0x4f, 0xb8, 0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76,
    0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1, 0x45, 0x9d, 0x70, 0x59,
    0x64, 0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8,
    0x02, 0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6, 0x1c, 0x46,
    0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f, 0x55, 0x47, 0xa3, 0x23, 0xdd,
    0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9, 0xce, 0xba, 0xc5, 0xea, 0x26,
    0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd,
    0xf4, 0x41, 0x81, 0x4d, 0x52, 0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1,
    0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a, 0x78,
    0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe,
    0x3b, 0x00, 0x1d, 0x39, 0xf2, 0xef, 0xb7, 0x0e, 0x66, 0x58, 0xd0,
    0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a, 0x31, 0x44,
    0x50, 0xb4, 0x8f, 0xed, 0x1f, 0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f,
    0x11, 0x83, 0x14
};


#if !defined(_WIN32_WINNT)
struct RH_ALIGN(128) md2 
{
    int L, f;  
    U8 pad[24];
    union
    {
        U8  c[MD2_BLOCK_SIZE];      
        U64 c64[2];
    } c;
    union
    {
        U8  x[MD2_BLOCK_SIZE * 3];  
        U64 x64[6];
        U32 x32[12];
    } x;

};


inline void md2_init(struct md2 *ctx)
{

    ctx->L = 0;
    ctx->f = 0;
    
    ctx->x.x64[0] = 0;
    ctx->x.x64[1] = 0;
    ctx->x.x64[2] = 0;
    ctx->x.x64[3] = 0;
    ctx->x.x64[4] = 0;
    ctx->x.x64[5] = 0;

    ctx->c.c64[0] = 0;
    ctx->c.c64[1] = 0;

}


inline void md2_append(struct md2 *ctx, const void *buf, size_t len)
{
    int j, k, t;
    const unsigned char *m;

    m = (const unsigned char*)buf;
    while (len) {
        for (; len && ctx->f < 16; len--, ctx->f++) {
            int b = *m++;
            ctx->x.x[ctx->f + 16] = b;
            ctx->x.x[ctx->f + 32] = b ^ ctx->x.x[ctx->f];
            ctx->L = ctx->c.c[ctx->f] ^= md2_s[b ^ ctx->L];
        }

        
        if (ctx->f == MD2_BLOCK_SIZE) {
            ctx->f = 0;
            t = 0;
            for (j = 0; j < 18; j++) {
                for (k = 0; k < 48; k++)
                    t = ctx->x.x[k] ^= md2_s[t];
                t = (t + j) % 256;
            }
        }
    }
}

#if 0
inline void md2_append_N(struct md2 *ctx,const char N, size_t len)
{
    int j, k, t;
    const unsigned char *m;

    while (len) {
        for (; len && ctx->f < 16; len--, ctx->f++) {
            ctx->x[ctx->f + 16] = N;
            ctx->x[ctx->f + 32] = N ^ ctx->x[ctx->f];
            ctx->L = ctx->c[ctx->f] ^= md2_s[N ^ ctx->L];
        }

        if (ctx->f == MD2_BLOCK_SIZE) {
            ctx->f = 0;
            t = 0;
            for (j = 0; j < 18; j++) {
                for (k = 0; k < 48; k++)
                    t = ctx->x[k] ^= md2_s[t];
                t = (t + j) % 256;
            }
        }
    }
}

inline void md2_finish(struct md2 *ctx, void *digest)
{
   int i, n;
    n = MD2_BLOCK_SIZE - ctx->f;
    md2_append_N(ctx, (char)n, n);
    md2_append(ctx, ctx->c, sizeof(ctx->c));
    memcpy(digest, ctx->x, 16);
}
#endif

void RandomHash_MD2(RH_StridePtr roundInput, RH_StridePtr output)
{
    U32 msgLen = RH_STRIDE_GET_SIZE(roundInput);
    U8* message = (U8*)RH_STRIDE_GET_DATA(roundInput);
    RH_STRIDE_SET_SIZE(output, 16);

    RH_ALIGN(128) U64 pad[2];
    RH_ALIGN(128) md2 ctx;
    md2_init(&ctx);

    if (msgLen == 32)
    {
        md2_append(&ctx, message, msgLen);
        pad[0] = 0x1010101010101010;
        pad[1] = 0x1010101010101010;
        md2_append(&ctx, (U8*)&pad[0], 16);
    }
    else if(msgLen == 100)
    {
        md2_append(&ctx, message, msgLen);
        pad[0] = 0x0c0c0c0c0c0c0c0c;
        pad[1] = 0x0c0c0c0c0c0c0c0c;
        md2_append(&ctx, (U8*)pad, 12);        
    }
    else
    {
        RH_ASSERT(false);
    }

    md2_append(&ctx, ctx.c.c, sizeof(ctx.c.c));
    U32* out = RH_STRIDE_GET_DATA(output);
    copy4(out, ctx.x.x32);
}

#else 

//Win32 ------------------------------------------------------------------------------------------------------------

struct md2 
{
    RH_ALIGN(128) int L, f;  
    RH_ALIGN(128) unsigned char c[MD2_BLOCK_SIZE];      
    RH_ALIGN(128) unsigned char x[MD2_BLOCK_SIZE * 3];  
};

inline void md2_init(struct md2 *ctx)
{
    ctx->L = 0;
    ctx->f = 0;
    __m128i xmm = _mm_set1_epi8((char)0); 
    RH_MM_STORE128(((__m128i *)ctx->x)+0, xmm); 
    RH_MM_STORE128(((__m128i *)ctx->c)+0, xmm); 
    RH_MM_STORE128(((__m128i *)ctx->c)+1, xmm); 
    RH_MM_STORE128(((__m128i *)ctx->c)+2, xmm);
}

inline void md2_append(struct md2 *ctx, U8 *buf, size_t len)
{
    int j, k, t, _f, _L;
    U8* _x;
    U8* _c;

    _c = ctx->c;
    _x = ctx->x;
    _f = ctx->f;
    _L = ctx->L;

    while (len) 
    {
        for (; len && _f < 16; len--, _f++) 
        {
            int b = *buf++;
            _x[_f + 16] = b;
            _x[_f + 32] = b ^ _x[_f];
            _c[_f     ] ^= md2_s[b ^ _L];
            _L = _c[_f];
        }

        if (_f == MD2_BLOCK_SIZE) 
        {
            U64 xcf, b;
            _f = 0;
            t = 0;
            for (j = 0; j < 18; j++)
            {
                for (k = 0; k < 48; k += 8)
                {
                    U8 tmp;
                    b = 0;

                    xcf = *(U64*)(void*)(_x + k);

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    t = tmp;

                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    t = tmp;

                    b = RH_swap_u64(b);
                    *(U64*)(void*)(_x + k) = b;
                }
                t = (t + j) % 256;
            }
        }
    }

    ctx->f = _f;
    ctx->L = _L;

}


inline void md2_append_16x(struct md2 *ctx, U8* buf, int len)
{
    int j, k, t, _f, _L;
    U8* _x;
    U8* _c;
    U64 b, xcf;

    _c = ctx->c;
    _x = ctx->x;
    _f = ctx->f;
    _L = ctx->L;    
    RH_ASSERT((_f%16) == 0); 
    while (len)
    {
        for (int j = 0; j < 2; j++)
        {
            b = *(U64*)(void*)buf;
            buf += 8;
            *(U64*)(_x + _f + 16) = b;
            xcf = *(U64*)(_x + _f);
            *(U64*)(_x + _f + 32) = b ^ xcf;

            xcf = *(U64*)(_c + _f); 
            U64 cf_md2sbl = 0;
            for (int i = 0; i < 8; i++)
            {
                U8 one_md2sbl = U8(xcf) ^ md2_s[U8(b) ^ _L];
                b >>= 8;
                xcf >>= 8;
                cf_md2sbl |= one_md2sbl;
                if (i < 7)
                    cf_md2sbl <<= 8;
                _L = one_md2sbl;
            }
            cf_md2sbl = RH_swap_u64(cf_md2sbl);
            *((U64*)(_c + _f)) = cf_md2sbl;
            _f += 8;
        }
        {
            _f = 0;
            t = 0;
            for (j = 0; j < 18; j++)
            {
                for (k = 0; k < 48; k += 8)
                {
                    U8 tmp;
                    b = 0;
                    xcf = *(U64*)(void*)(_x + k);
                    tmp = U8(xcf) ^ md2_s[t]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    b <<= 8;
                    xcf >>= 8;
                    tmp = U8(xcf) ^ md2_s[tmp]; 
                    b |= tmp;
                    t = tmp;

                    b = RH_swap_u64(b);
                    *(U64*)(void*)(_x + k) = b;
                }

                t = (t + j) % 256;
            }
        }
        len -= 16;
    }
    ctx->f = _f;
    ctx->L = _L;
}



inline void RandomHash_MD2(RH_StridePtr roundInput, RH_StridePtr output)
{
    U32 msgLen = RH_STRIDE_GET_SIZE(roundInput);
    U8* message = RH_STRIDE_GET_DATA8(roundInput);
    RH_STRIDE_SET_SIZE(output, 16);

    RH_ALIGN(128) U64 pad[2];
    RH_ALIGN(128) md2 ctx;
    md2_init(&ctx);

    if (msgLen == 32)
    {
        md2_append_16x(&ctx, message, msgLen);
        pad[0] = 0x1010101010101010;
        pad[1] = 0x1010101010101010;
        md2_append_16x(&ctx, (U8*)&pad[0], 16);
    }
    else if(msgLen == 100)
    {
        U32 end16 = (msgLen / 16) * 16;
        md2_append_16x(&ctx, message, end16);
        md2_append(&ctx, message+end16, msgLen-end16);

        pad[0] = 0x0c0c0c0c0c0c0c0c;
        pad[1] = 0x0c0c0c0c0c0c0c0c;
        md2_append(&ctx, (U8*)pad, 12);        
    }
    else
    {
        RH_ASSERT(false);
    }

    md2_append_16x(&ctx, ctx.c, sizeof(ctx.c));

    U64* out = RH_STRIDE_GET_DATA64(output);
    U64* c = (U64*)&ctx.x;
    out[0] = c[0];
    out[1] = c[1];
}
#endif //linux