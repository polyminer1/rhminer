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

#define RIPEMD160_BLOCK_SIZE 64

inline U32 P1(U32 a, U32 b, U32 c)
{
    return (a & b) | (~a & c);
}

inline U32 P2(U32 a, U32 b, U32 c)
{
    return (a & b) | (a & c) | (b & c);;
}

inline U32 P3(U32 a, U32 b, U32 c)
{
    return a ^ b ^ c;
}

inline void RipemdRoundFunction(uint32_t* data, uint32_t* state)
{
    U32  a, b, c, d, aa, bb, cc, dd;

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    aa = a;
    bb = b;
    cc = c;
    dd = d;

    a = ROTL32(P1(b, c, d) + a + data[0], 11);
    d = ROTL32(P1(a, b, c) + d + data[1], 14);
    c = ROTL32(P1(d, a, b) + c + data[2], 15);
    b = ROTL32(P1(c, d, a) + b + data[3], 12);
    a = ROTL32(P1(b, c, d) + a + data[4], 5);
    d = ROTL32(P1(a, b, c) + d + data[5], 8);
    c = ROTL32(P1(d, a, b) + c + data[6], 7);
    b = ROTL32(P1(c, d, a) + b + data[7], 9);
    a = ROTL32(P1(b, c, d) + a + data[8], 11);
    d = ROTL32(P1(a, b, c) + d + data[9], 13);
    c = ROTL32(P1(d, a, b) + c + data[10], 14);
    b = ROTL32(P1(c, d, a) + b + data[11], 15);
    a = ROTL32(P1(b, c, d) + a + data[12], 6);
    d = ROTL32(P1(a, b, c) + d + data[13], 7);
    c = ROTL32(P1(d, a, b) + c + data[14], 9);
    b = ROTL32(P1(c, d, a) + b + data[15], 8);

    a = ROTL32(P2(b, c, d) + a + data[7] + RH_RIPEMD_C2, 7);
    d = ROTL32(P2(a, b, c) + d + data[4] + RH_RIPEMD_C2, 6);
    c = ROTL32(P2(d, a, b) + c + data[13] + RH_RIPEMD_C2, 8);
    b = ROTL32(P2(c, d, a) + b + data[1] + RH_RIPEMD_C2, 13);
    a = ROTL32(P2(b, c, d) + a + data[10] + RH_RIPEMD_C2, 11);
    d = ROTL32(P2(a, b, c) + d + data[6] + RH_RIPEMD_C2, 9);
    c = ROTL32(P2(d, a, b) + c + data[15] + RH_RIPEMD_C2, 7);
    b = ROTL32(P2(c, d, a) + b + data[3] + RH_RIPEMD_C2, 15);
    a = ROTL32(P2(b, c, d) + a + data[12] + RH_RIPEMD_C2, 7);
    d = ROTL32(P2(a, b, c) + d + data[0] + RH_RIPEMD_C2, 12);
    c = ROTL32(P2(d, a, b) + c + data[9] + RH_RIPEMD_C2, 15);
    b = ROTL32(P2(c, d, a) + b + data[5] + RH_RIPEMD_C2, 9);
    a = ROTL32(P2(b, c, d) + a + data[14] + RH_RIPEMD_C2, 7);
    d = ROTL32(P2(a, b, c) + d + data[2] + RH_RIPEMD_C2, 11);
    c = ROTL32(P2(d, a, b) + c + data[11] + RH_RIPEMD_C2, 13);
    b = ROTL32(P2(c, d, a) + b + data[8] + RH_RIPEMD_C2, 12);

    a = ROTL32(P3(b, c, d) + a + data[3] + RH_RIPEMD_C4, 11);
    d = ROTL32(P3(a, b, c) + d + data[10] + RH_RIPEMD_C4, 13);
    c = ROTL32(P3(d, a, b) + c + data[2] + RH_RIPEMD_C4, 14);
    b = ROTL32(P3(c, d, a) + b + data[4] + RH_RIPEMD_C4, 7);
    a = ROTL32(P3(b, c, d) + a + data[9] + RH_RIPEMD_C4, 14);
    d = ROTL32(P3(a, b, c) + d + data[15] + RH_RIPEMD_C4, 9);
    c = ROTL32(P3(d, a, b) + c + data[8] + RH_RIPEMD_C4, 13);
    b = ROTL32(P3(c, d, a) + b + data[1] + RH_RIPEMD_C4, 15);
    a = ROTL32(P3(b, c, d) + a + data[14] + RH_RIPEMD_C4, 6);
    d = ROTL32(P3(a, b, c) + d + data[7] + RH_RIPEMD_C4, 8);
    c = ROTL32(P3(d, a, b) + c + data[0] + RH_RIPEMD_C4, 13);
    b = ROTL32(P3(c, d, a) + b + data[6] + RH_RIPEMD_C4, 6);
    a = ROTL32(P3(b, c, d) + a + data[11] + RH_RIPEMD_C4, 12);
    d = ROTL32(P3(a, b, c) + d + data[13] + RH_RIPEMD_C4, 5);
    c = ROTL32(P3(d, a, b) + c + data[5] + RH_RIPEMD_C4, 7);
    b = ROTL32(P3(c, d, a) + b + data[12] + RH_RIPEMD_C4, 5);

    aa = ROTL32(P1(bb, cc, dd) + aa + data[0] + RH_RIPEMD_C1, 11);
    dd = ROTL32(P1(aa, bb, cc) + dd + data[1] + RH_RIPEMD_C1, 14);
    cc = ROTL32(P1(dd, aa, bb) + cc + data[2] + RH_RIPEMD_C1, 15);
    bb = ROTL32(P1(cc, dd, aa) + bb + data[3] + RH_RIPEMD_C1, 12);
    aa = ROTL32(P1(bb, cc, dd) + aa + data[4] + RH_RIPEMD_C1, 5);
    dd = ROTL32(P1(aa, bb, cc) + dd + data[5] + RH_RIPEMD_C1, 8);
    cc = ROTL32(P1(dd, aa, bb) + cc + data[6] + RH_RIPEMD_C1, 7);
    bb = ROTL32(P1(cc, dd, aa) + bb + data[7] + RH_RIPEMD_C1, 9);
    aa = ROTL32(P1(bb, cc, dd) + aa + data[8] + RH_RIPEMD_C1, 11);
    dd = ROTL32(P1(aa, bb, cc) + dd + data[9] + RH_RIPEMD_C1, 13);
    cc = ROTL32(P1(dd, aa, bb) + cc + data[10] + RH_RIPEMD_C1, 14);
    bb = ROTL32(P1(cc, dd, aa) + bb + data[11] + RH_RIPEMD_C1, 15);
    aa = ROTL32(P1(bb, cc, dd) + aa + data[12] + RH_RIPEMD_C1, 6);
    dd = ROTL32(P1(aa, bb, cc) + dd + data[13] + RH_RIPEMD_C1, 7);
    cc = ROTL32(P1(dd, aa, bb) + cc + data[14] + RH_RIPEMD_C1, 9);
    bb = ROTL32(P1(cc, dd, aa) + bb + data[15] + RH_RIPEMD_C1, 8);

    aa = ROTL32(P2(bb, cc, dd) + aa + data[7], 7);
    dd = ROTL32(P2(aa, bb, cc) + dd + data[4], 6);
    cc = ROTL32(P2(dd, aa, bb) + cc + data[13], 8);
    bb = ROTL32(P2(cc, dd, aa) + bb + data[1], 13);
    aa = ROTL32(P2(bb, cc, dd) + aa + data[10], 11);
    dd = ROTL32(P2(aa, bb, cc) + dd + data[6], 9);
    cc = ROTL32(P2(dd, aa, bb) + cc + data[15], 7);
    bb = ROTL32(P2(cc, dd, aa) + bb + data[3], 15);
    aa = ROTL32(P2(bb, cc, dd) + aa + data[12], 7);
    dd = ROTL32(P2(aa, bb, cc) + dd + data[0], 12);
    cc = ROTL32(P2(dd, aa, bb) + cc + data[9], 15);
    bb = ROTL32(P2(cc, dd, aa) + bb + data[5], 9);
    aa = ROTL32(P2(bb, cc, dd) + aa + data[14], 7);
    dd = ROTL32(P2(aa, bb, cc) + dd + data[2], 11);
    cc = ROTL32(P2(dd, aa, bb) + cc + data[11], 13);
    bb = ROTL32(P2(cc, dd, aa) + bb + data[8], 12);

    aa = ROTL32(P3(bb, cc, dd) + aa + data[3] + RH_RIPEMD_C3, 11);
    dd = ROTL32(P3(aa, bb, cc) + dd + data[10] + RH_RIPEMD_C3, 13);
    cc = ROTL32(P3(dd, aa, bb) + cc + data[2] + RH_RIPEMD_C3, 14);
    bb = ROTL32(P3(cc, dd, aa) + bb + data[4] + RH_RIPEMD_C3, 7);
    aa = ROTL32(P3(bb, cc, dd) + aa + data[9] + RH_RIPEMD_C3, 14);
    dd = ROTL32(P3(aa, bb, cc) + dd + data[15] + RH_RIPEMD_C3, 9);
    cc = ROTL32(P3(dd, aa, bb) + cc + data[8] + RH_RIPEMD_C3, 13);
    bb = ROTL32(P3(cc, dd, aa) + bb + data[1] + RH_RIPEMD_C3, 15);
    aa = ROTL32(P3(bb, cc, dd) + aa + data[14] + RH_RIPEMD_C3, 6);
    dd = ROTL32(P3(aa, bb, cc) + dd + data[7] + RH_RIPEMD_C3, 8);
    cc = ROTL32(P3(dd, aa, bb) + cc + data[0] + RH_RIPEMD_C3, 13);
    bb = ROTL32(P3(cc, dd, aa) + bb + data[6] + RH_RIPEMD_C3, 6);
    aa = ROTL32(P3(bb, cc, dd) + aa + data[11] + RH_RIPEMD_C3, 12);
    dd = ROTL32(P3(aa, bb, cc) + dd + data[13] + RH_RIPEMD_C3, 5);
    cc = ROTL32(P3(dd, aa, bb) + cc + data[5] + RH_RIPEMD_C3, 7);
    bb = ROTL32(P3(cc, dd, aa) + bb + data[12] + RH_RIPEMD_C3, 5);

    cc = cc + state[0] + b;
    state[0] = state[1] + c + dd;
    state[1] = state[2] + d + aa;
    state[2] = state[3] + a + bb;
    state[3] = cc;
}


void RandomHash_RIPEMD(RH_StridePtr roundInput, RH_StridePtr output)
{
    RH_ALIGN(64) uint32_t state[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    RandomHash_MD_BASE_MAIN_LOOP(64, RipemdRoundFunction, uint64_t);

    U32* out = RH_STRIDE_GET_DATA(output);
    RH_STRIDE_SET_SIZE(output, 16);
    copy4(out, state);

}
