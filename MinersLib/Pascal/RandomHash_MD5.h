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

#define MD5_BLOCKSIZE 64

#ifndef RHMINER_PLATFORM_GPU
    #define UINT4 uint32_t
#elif defined(RHMINER_PLATFORM_CUDA)
    #define UINT4 uint32_t
#else
error
#endif

#define MD5_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | (~z))) 


/* MD5_FF, MD5_GG, MD5_HH, and MD5_II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define MD5_FF(a, b, c, d, x, s, ac) \
  {(a) += MD5_F ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTL32 ((a), (s)); \
   (a) += (b); \
  }

#define MD5_GG(a, b, c, d, x, s, ac) \
  {(a) += MD5_G ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTL32 ((a), (s)); \
   (a) += (b); \
  }

#define MD5_HH(a, b, c, d, x, s, ac) \
  {(a) += MD5_H ((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTL32 ((a), (s)); \
   (a) += (b); \
  }

#define MD5_II(a, b, c, d, x, s, ac) \
  {(a) += MD5_I((b), (c), (d)) + (x) + (UINT4)(ac); \
   (a) = ROTL32 ((a), (s)); \
   (a) += (b); \
  }

PLATFORM_CONST uint32_t MD5_a0 = 0x67452301;
PLATFORM_CONST uint32_t MD5_b0 = 0xEFCDAB89;
PLATFORM_CONST uint32_t MD5_c0 = 0x98BADCFE;
PLATFORM_CONST uint32_t MD5_d0 = 0x10325476;



void md5(uint32_t *in, uint32_t *state) 
{
    uint32_t a, b, c, d;

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];

    /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22

    MD5_FF(a, b, c, d, in[0], S11, 3614090360); /* 1 */
    MD5_FF(d, a, b, c, in[1], S12, 3905402710); /* 2 */
    MD5_FF(c, d, a, b, in[2], S13, 606105819); /* 3 */
    MD5_FF(b, c, d, a, in[3], S14, 3250441966); /* 4 */
    MD5_FF(a, b, c, d, in[4], S11, 4118548399); /* 5 */
    MD5_FF(d, a, b, c, in[5], S12, 1200080426); /* 6 */
    MD5_FF(c, d, a, b, in[6], S13, 2821735955); /* 7 */
    MD5_FF(b, c, d, a, in[7], S14, 4249261313); /* 8 */
    MD5_FF(a, b, c, d, in[8], S11, 1770035416); /* 9 */
    MD5_FF(d, a, b, c, in[9], S12, 2336552879); /* 10 */
    MD5_FF(c, d, a, b, in[10], S13, 4294925233); /* 11 */
    MD5_FF(b, c, d, a, in[11], S14, 2304563134); /* 12 */
    MD5_FF(a, b, c, d, in[12], S11, 1804603682); /* 13 */
    MD5_FF(d, a, b, c, in[13], S12, 4254626195); /* 14 */
    MD5_FF(c, d, a, b, in[14], S13, 2792965006); /* 15 */
    MD5_FF(b, c, d, a, in[15], S14, 1236535329); /* 16 */

    /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20

    MD5_GG(a, b, c, d, in[1], S21, 4129170786); /* 17 */
    MD5_GG(d, a, b, c, in[6], S22, 3225465664); /* 18 */
    MD5_GG(c, d, a, b, in[11], S23, 643717713); /* 19 */
    MD5_GG(b, c, d, a, in[0], S24, 3921069994); /* 20 */
    MD5_GG(a, b, c, d, in[5], S21, 3593408605); /* 21 */
    MD5_GG(d, a, b, c, in[10], S22, 38016083); /* 22 */
    MD5_GG(c, d, a, b, in[15], S23, 3634488961); /* 23 */
    MD5_GG(b, c, d, a, in[4], S24, 3889429448); /* 24 */
    MD5_GG(a, b, c, d, in[9], S21, 568446438); /* 25 */
    MD5_GG(d, a, b, c, in[14], S22, 3275163606); /* 26 */
    MD5_GG(c, d, a, b, in[3], S23, 4107603335); /* 27 */
    MD5_GG(b, c, d, a, in[8], S24, 1163531501); /* 28 */
    MD5_GG(a, b, c, d, in[13], S21, 2850285829); /* 29 */
    MD5_GG(d, a, b, c, in[2], S22, 4243563512); /* 30 */
    MD5_GG(c, d, a, b, in[7], S23, 1735328473); /* 31 */
    MD5_GG(b, c, d, a, in[12], S24, 2368359562); /* 32 */

    /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23

    MD5_HH(a, b, c, d, in[5], S31, 4294588738); /* 33 */
    MD5_HH(d, a, b, c, in[8], S32, 2272392833); /* 34 */
    MD5_HH(c, d, a, b, in[11], S33, 1839030562); /* 35 */
    MD5_HH(b, c, d, a, in[14], S34, 4259657740); /* 36 */
    MD5_HH(a, b, c, d, in[1], S31, 2763975236); /* 37 */
    MD5_HH(d, a, b, c, in[4], S32, 1272893353); /* 38 */
    MD5_HH(c, d, a, b, in[7], S33, 4139469664); /* 39 */
    MD5_HH(b, c, d, a, in[10], S34, 3200236656); /* 40 */
    MD5_HH(a, b, c, d, in[13], S31, 681279174); /* 41 */
    MD5_HH(d, a, b, c, in[0], S32, 3936430074); /* 42 */
    MD5_HH(c, d, a, b, in[3], S33, 3572445317); /* 43 */
    MD5_HH(b, c, d, a, in[6], S34, 76029189); /* 44 */
    MD5_HH(a, b, c, d, in[9], S31, 3654602809); /* 45 */
    MD5_HH(d, a, b, c, in[12], S32, 3873151461); /* 46 */
    MD5_HH(c, d, a, b, in[15], S33, 530742520); /* 47 */
    MD5_HH(b, c, d, a, in[2], S34, 3299628645); /* 48 */

    /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
    MD5_II(a, b, c, d, in[0], S41, 4096336452); /* 49 */
    MD5_II(d, a, b, c, in[7], S42, 1126891415); /* 50 */
    MD5_II(c, d, a, b, in[14], S43, 2878612391); /* 51 */
    MD5_II(b, c, d, a, in[5], S44, 4237533241); /* 52 */
    MD5_II(a, b, c, d, in[12], S41, 1700485571); /* 53 */
    MD5_II(d, a, b, c, in[3], S42, 2399980690); /* 54 */
    MD5_II(c, d, a, b, in[10], S43, 4293915773); /* 55 */
    MD5_II(b, c, d, a, in[1], S44, 2240044497); /* 56 */
    MD5_II(a, b, c, d, in[8], S41, 1873313359); /* 57 */
    MD5_II(d, a, b, c, in[15], S42, 4264355552); /* 58 */
    MD5_II(c, d, a, b, in[6], S43, 2734768916); /* 59 */
    MD5_II(b, c, d, a, in[13], S44, 1309151649); /* 60 */
    MD5_II(a, b, c, d, in[4], S41, 4149444226); /* 61 */
    MD5_II(d, a, b, c, in[11], S42, 3174756917); /* 62 */
    MD5_II(c, d, a, b, in[2], S43, 718787259); /* 63 */
    MD5_II(b, c, d, a, in[9], S44, 3951481745); /* 64 */

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;

    return;
}
#undef S11
#undef S12
#undef S13
#undef S14
#undef S21
#undef S22
#undef S23
#undef S24
#undef S31
#undef S32
#undef S33
#undef S34
#undef S41
#undef S42
#undef S43
#undef S44


void RandomHash_MD5(RH_StridePtr roundInput, RH_StridePtr output)
{
    RH_ALIGN(64) uint32_t state[4] = { MD5_a0, MD5_b0, MD5_c0, MD5_d0 };
    RandomHash_MD_BASE_MAIN_LOOP(MD5_BLOCKSIZE, md5, uint64_t);

    //get the hash result
    U32* out = RH_STRIDE_GET_DATA(output);
    RH_STRIDE_SET_SIZE(output, 16);
    copy4(out, state);
}
