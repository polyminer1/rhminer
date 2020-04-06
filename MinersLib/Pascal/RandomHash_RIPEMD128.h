/**
 *
 * Copyright 2018 Polyminer1 <https://github.com/polyminer1>
 *
 * To the extent possible under law, the author(s) have dedicated all copyright
 * ^ related ^ neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along with
 * this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

///
/// @file
/// @copyright Polyminer1, QualiaLibre

#include "RandomHash_core.h"


inline void Ripemd128RoundFunction(uint32_t* data, uint32_t* state)
{
    U32 a, b, c, d, aa, bb, cc, dd;
        
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  aa = a;
  bb = b;
  cc = c;
  dd = d;

  a = a + (data[0] + (b ^ c ^ d));
  a = ROTL32(a, 11);
  d = d + (data[1] + (a ^ b ^ c));
  d = ROTL32(d, 14);
  c = c + (data[2] + (d ^ a ^ b));
  c = ROTL32(c, 15);
  b = b + (data[3] + (c ^ d ^ a));
  b = ROTL32(b, 12);
  a = a + (data[4] + (b ^ c ^ d));
  a = ROTL32(a, 5);
  d = d + (data[5] + (a ^ b ^ c));
  d = ROTL32(d, 8);
  c = c + (data[6] + (d ^ a ^ b));
  c = ROTL32(c, 7);
  b = b + (data[7] + (c ^ d ^ a));
  b = ROTL32(b, 9);
  a = a + (data[8] + (b ^ c ^ d));
  a = ROTL32(a, 11);
  d = d + (data[9] + (a ^ b ^ c));
  d = ROTL32(d, 13);
  c = c + (data[10] + (d ^ a ^ b));
  c = ROTL32(c, 14);
  b = b + (data[11] + (c ^ d ^ a));
  b = ROTL32(b, 15);
  a = a + (data[12] + (b ^ c ^ d));
  a = ROTL32(a, 6);
  d = d + (data[13] + (a ^ b ^ c));
  d = ROTL32(d, 7);
  c = c + (data[14] + (d ^ a ^ b));
  c = ROTL32(c, 9);
  b = b + (data[15] + (c ^ d ^ a));
  b = ROTL32(b, 8);

  a = a + (data[7] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
  a = ROTL32(a, 7);
  d = d + (data[4] + RH_RIPEMD_C2 + ((a & b) | (~a & c)));
  d = ROTL32(d, 6);
  c = c + (data[13] + RH_RIPEMD_C2 + ((d & a) | (~d & b)));
  c = ROTL32(c, 8);
  b = b + (data[1] + RH_RIPEMD_C2 + ((c & d) | (~c & a)));
  b = ROTL32(b, 13);
  a = a + (data[10] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
  a = ROTL32(a, 11);
  d = d + (data[6] + RH_RIPEMD_C2 + ((a & b) | (~a & c)));
  d = ROTL32(d, 9);
  c = c + (data[15] + RH_RIPEMD_C2 + ((d & a) | (~d & b)));
  c = ROTL32(c, 7);
  b = b + (data[3] + RH_RIPEMD_C2 + ((c & d) | (~c & a)));
  b = ROTL32(b, 15);
  a = a + (data[12] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
  a = ROTL32(a, 7);
  d = d + (data[0] + RH_RIPEMD_C2 + ((a & b) | (~a & c)));
  d = ROTL32(d, 12);
  c = c + (data[9] + RH_RIPEMD_C2 + ((d & a) | (~d & b)));
  c = ROTL32(c, 15);
  b = b + (data[5] + RH_RIPEMD_C2 + ((c & d) | (~c & a)));
  b = ROTL32(b, 9);
  a = a + (data[2] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
  a = ROTL32(a, 11);
  d = d + (data[14] + RH_RIPEMD_C2 + ((a & b) | (~a & c)));
  d = ROTL32(d, 7);
  c = c + (data[11] + RH_RIPEMD_C2 + ((d & a) | (~d & b)));
  c = ROTL32(c, 13);
  b = b + (data[8] + RH_RIPEMD_C2 + ((c & d) | (~c & a)));
  b = ROTL32(b, 12);

  a = a + (data[3] + RH_RIPEMD_C4 + ((b | ~c) ^ d));
  a = ROTL32(a, 11);
  d = d + (data[10] + RH_RIPEMD_C4 + ((a | ~b) ^ c));
  d = ROTL32(d, 13);
  c = c + (data[14] + RH_RIPEMD_C4 + ((d | ~a) ^ b));
  c = ROTL32(c, 6);
  b = b + (data[4] + RH_RIPEMD_C4 + ((c | ~d) ^ a));
  b = ROTL32(b, 7);
  a = a + (data[9] + RH_RIPEMD_C4 + ((b | ~c) ^ d));
  a = ROTL32(a, 14);
  d = d + (data[15] + RH_RIPEMD_C4 + ((a | ~b) ^ c));
  d = ROTL32(d, 9);
  c = c + (data[8] + RH_RIPEMD_C4 + ((d | ~a) ^ b));
  c = ROTL32(c, 13);
  b = b + (data[1] + RH_RIPEMD_C4 + ((c | ~d) ^ a));
  b = ROTL32(b, 15);
  a = a + (data[2] + RH_RIPEMD_C4 + ((b | ~c) ^ d));
  a = ROTL32(a, 14);
  d = d + (data[7] + RH_RIPEMD_C4 + ((a | ~b) ^ c));
  d = ROTL32(d, 8);
  c = c + (data[0] + RH_RIPEMD_C4 + ((d | ~a) ^ b));
  c = ROTL32(c, 13);
  b = b + (data[6] + RH_RIPEMD_C4 + ((c | ~d) ^ a));
  b = ROTL32(b, 6);
  a = a + (data[13] + RH_RIPEMD_C4 + ((b | ~c) ^ d));
  a = ROTL32(a, 5);
  d = d + (data[11] + RH_RIPEMD_C4 + ((a | ~b) ^ c));
  d = ROTL32(d, 12);
  c = c + (data[5] + RH_RIPEMD_C4 + ((d | ~a) ^ b));
  c = ROTL32(c, 7);
  b = b + (data[12] + RH_RIPEMD_C4 + ((c | ~d) ^ a));
  b = ROTL32(b, 5);

  a = a + (data[1] + RH_RIPEMD_C6 + ((b & d) | (c & ~d)));
  a = ROTL32(a, 11);
  d = d + (data[9] + RH_RIPEMD_C6 + ((a & c) | (b & ~c)));
  d = ROTL32(d, 12);
  c = c + (data[11] + RH_RIPEMD_C6 + ((d & b) | (a & ~b)));
  c = ROTL32(c, 14);
  b = b + (data[10] + RH_RIPEMD_C6 + ((c & a) | (d & ~a)));
  b = ROTL32(b, 15);
  a = a + (data[0] + RH_RIPEMD_C6 + ((b & d) | (c & ~d)));
  a = ROTL32(a, 14);
  d = d + (data[8] + RH_RIPEMD_C6 + ((a & c) | (b & ~c)));
  d = ROTL32(d, 15);
  c = c + (data[12] + RH_RIPEMD_C6 + ((d & b) | (a & ~b)));
  c = ROTL32(c, 9);
  b = b + (data[4] + RH_RIPEMD_C6 + ((c & a) | (d & ~a)));
  b = ROTL32(b, 8);
  a = a + (data[13] + RH_RIPEMD_C6 + ((b & d) | (c & ~d)));
  a = ROTL32(a, 9);
  d = d + (data[3] + RH_RIPEMD_C6 + ((a & c) | (b & ~c)));
  d = ROTL32(d, 14);
  c = c + (data[7] + RH_RIPEMD_C6 + ((d & b) | (a & ~b)));
  c = ROTL32(c, 5);
  b = b + (data[15] + RH_RIPEMD_C6 + ((c & a) | (d & ~a)));
  b = ROTL32(b, 6);
  a = a + (data[14] + RH_RIPEMD_C6 + ((b & d) | (c & ~d)));
  a = ROTL32(a, 8);
  d = d + (data[5] + RH_RIPEMD_C6 + ((a & c) | (b & ~c)));
  d = ROTL32(d, 6);
  c = c + (data[6] + RH_RIPEMD_C6 + ((d & b) | (a & ~b)));
  c = ROTL32(c, 5);
  b = b + (data[2] + RH_RIPEMD_C6 + ((c & a) | (d & ~a)));
  b = ROTL32(b, 12);

  aa = aa + (data[5] + RH_RIPEMD_C1 + ((bb & dd) | (cc & ~dd)));
  aa = ROTL32(aa, 8);
  dd = dd + (data[14] + RH_RIPEMD_C1 + ((aa & cc) | (bb & ~cc)));
  dd = ROTL32(dd, 9);
  cc = cc + (data[7] + RH_RIPEMD_C1 + ((dd & bb) | (aa & ~bb)));
  cc = ROTL32(cc, 9);
  bb = bb + (data[0] + RH_RIPEMD_C1 + ((cc & aa) | (dd & ~aa)));
  bb = ROTL32(bb, 11);
  aa = aa + (data[9] + RH_RIPEMD_C1 + ((bb & dd) | (cc & ~dd)));
  aa = ROTL32(aa, 13);
  dd = dd + (data[2] + RH_RIPEMD_C1 + ((aa & cc) | (bb & ~cc)));
  dd = ROTL32(dd, 15);
  cc = cc + (data[11] + RH_RIPEMD_C1 + ((dd & bb) | (aa & ~bb)));
  cc = ROTL32(cc, 15);
  bb = bb + (data[4] + RH_RIPEMD_C1 + ((cc & aa) | (dd & ~aa)));
  bb = ROTL32(bb, 5);
  aa = aa + (data[13] + RH_RIPEMD_C1 + ((bb & dd) | (cc & ~dd)));
  aa = ROTL32(aa, 7);
  dd = dd + (data[6] + RH_RIPEMD_C1 + ((aa & cc) | (bb & ~cc)));
  dd = ROTL32(dd, 7);
  cc = cc + (data[15] + RH_RIPEMD_C1 + ((dd & bb) | (aa & ~bb)));
  cc = ROTL32(cc, 8);
  bb = bb + (data[8] + RH_RIPEMD_C1 + ((cc & aa) | (dd & ~aa)));
  bb = ROTL32(bb, 11);
  aa = aa + (data[1] + RH_RIPEMD_C1 + ((bb & dd) | (cc & ~dd)));
  aa = ROTL32(aa, 14);
  dd = dd + (data[10] + RH_RIPEMD_C1 + ((aa & cc) | (bb & ~cc)));
  dd = ROTL32(dd, 14);
  cc = cc + (data[3] + RH_RIPEMD_C1 + ((dd & bb) | (aa & ~bb)));
  cc = ROTL32(cc, 12);
  bb = bb + (data[12] + RH_RIPEMD_C1 + ((cc & aa) | (dd & ~aa)));
  bb = ROTL32(bb, 6);

  aa = aa + (data[6] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
  aa = ROTL32(aa, 9);
  dd = dd + (data[11] + RH_RIPEMD_C3 + ((aa | ~bb) ^ cc));
  dd = ROTL32(dd, 13);
  cc = cc + (data[3] + RH_RIPEMD_C3 + ((dd | ~aa) ^ bb));
  cc = ROTL32(cc, 15);
  bb = bb + (data[7] + RH_RIPEMD_C3 + ((cc | ~dd) ^ aa));
  bb = ROTL32(bb, 7);
  aa = aa + (data[0] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
  aa = ROTL32(aa, 12);
  dd = dd + (data[13] + RH_RIPEMD_C3 + ((aa | ~bb) ^ cc));
  dd = ROTL32(dd, 8);
  cc = cc + (data[5] + RH_RIPEMD_C3 + ((dd | ~aa) ^ bb));
  cc = ROTL32(cc, 9);
  bb = bb + (data[10] + RH_RIPEMD_C3 + ((cc | ~dd) ^ aa));
  bb = ROTL32(bb, 11);
  aa = aa + (data[14] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
  aa = ROTL32(aa, 7);
  dd = dd + (data[15] + RH_RIPEMD_C3 + ((aa | ~bb) ^ cc));
  dd = ROTL32(dd, 7);
  cc = cc + (data[8] + RH_RIPEMD_C3 + ((dd | ~aa) ^ bb));
  cc = ROTL32(cc, 12);
  bb = bb + (data[12] + RH_RIPEMD_C3 + ((cc | ~dd) ^ aa));
  bb = ROTL32(bb, 7);
  aa = aa + (data[4] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
  aa = ROTL32(aa, 6);
  dd = dd + (data[9] + RH_RIPEMD_C3 + ((aa | ~bb) ^ cc));
  dd = ROTL32(dd, 15);
  cc = cc + (data[1] + RH_RIPEMD_C3 + ((dd | ~aa) ^ bb));
  cc = ROTL32(cc, 13);
  bb = bb + (data[2] + RH_RIPEMD_C3 + ((cc | ~dd) ^ aa));
  bb = ROTL32(bb, 11);

  aa = aa + (data[15] + RH_RIPEMD_C5 + ((bb & cc) | (~bb & dd)));
  aa = ROTL32(aa, 9);
  dd = dd + (data[5] + RH_RIPEMD_C5 + ((aa & bb) | (~aa & cc)));
  dd = ROTL32(dd, 7);
  cc = cc + (data[1] + RH_RIPEMD_C5 + ((dd & aa) | (~dd & bb)));
  cc = ROTL32(cc, 15);
  bb = bb + (data[3] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & aa)));
  bb = ROTL32(bb, 11);
  aa = aa + (data[7] + RH_RIPEMD_C5 + ((bb & cc) | (~bb & dd)));
  aa = ROTL32(aa, 8);
  dd = dd + (data[14] + RH_RIPEMD_C5 + ((aa & bb) | (~aa & cc)));
  dd = ROTL32(dd, 6);
  cc = cc + (data[6] + RH_RIPEMD_C5 + ((dd & aa) | (~dd & bb)));
  cc = ROTL32(cc, 6);
  bb = bb + (data[9] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & aa)));
  bb = ROTL32(bb, 14);
  aa = aa + (data[11] + RH_RIPEMD_C5 + ((bb & cc) | (~bb & dd)));
  aa = ROTL32(aa, 12);
  dd = dd + (data[8] + RH_RIPEMD_C5 + ((aa & bb) | (~aa & cc)));
  dd = ROTL32(dd, 13);
  cc = cc + (data[12] + RH_RIPEMD_C5 + ((dd & aa) | (~dd & bb)));
  cc = ROTL32(cc, 5);
  bb = bb + (data[2] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & aa)));
  bb = ROTL32(bb, 14);
  aa = aa + (data[10] + RH_RIPEMD_C5 + ((bb & cc) | (~bb & dd)));
  aa = ROTL32(aa, 13);
  dd = dd + (data[0] + RH_RIPEMD_C5 + ((aa & bb) | (~aa & cc)));
  dd = ROTL32(dd, 13);
  cc = cc + (data[4] + RH_RIPEMD_C5 + ((dd & aa) | (~dd & bb)));
  cc = ROTL32(cc, 7);
  bb = bb + (data[13] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & aa)));
  bb = ROTL32(bb, 5);

  aa = aa + (data[8] + (bb ^ cc ^ dd));
  aa = ROTL32(aa, 15);
  dd = dd + (data[6] + (aa ^ bb ^ cc));
  dd = ROTL32(dd, 5);
  cc = cc + (data[4] + (dd ^ aa ^ bb));
  cc = ROTL32(cc, 8);
  bb = bb + (data[1] + (cc ^ dd ^ aa));
  bb = ROTL32(bb, 11);
  aa = aa + (data[3] + (bb ^ cc ^ dd));
  aa = ROTL32(aa, 14);
  dd = dd + (data[11] + (aa ^ bb ^ cc));
  dd = ROTL32(dd, 14);
  cc = cc + (data[15] + (dd ^ aa ^ bb));
  cc = ROTL32(cc, 6);
  bb = bb + (data[0] + (cc ^ dd ^ aa));
  bb = ROTL32(bb, 14);
  aa = aa + (data[5] + (bb ^ cc ^ dd));
  aa = ROTL32(aa, 6);
  dd = dd + (data[12] + (aa ^ bb ^ cc));
  dd = ROTL32(dd, 9);
  cc = cc + (data[2] + (dd ^ aa ^ bb));
  cc = ROTL32(cc, 12);
  bb = bb + (data[13] + (cc ^ dd ^ aa));
  bb = ROTL32(bb, 9);
  aa = aa + (data[9] + (bb ^ cc ^ dd));
  aa = ROTL32(aa, 12);
  dd = dd + (data[7] + (aa ^ bb ^ cc));
  dd = ROTL32(dd, 5);
  cc = cc + (data[10] + (dd ^ aa ^ bb));
  cc = ROTL32(cc, 15);
  bb = bb + (data[14] + (cc ^ dd ^ aa));
  bb = ROTL32(bb, 8);

  dd = dd + c + state[1];
  state[1] = state[2] + d + aa;
  state[2] = state[3] + a + bb;
  state[3] = state[0] + b + cc;
  state[0] = dd;
}


void RandomHash_RIPEMD128(RH_StridePtr roundInput, RH_StridePtr output)
{
    RH_ALIGN(64) uint32_t state[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    RandomHash_MD_BASE_MAIN_LOOP(64, Ripemd128RoundFunction, uint64_t);
    
    //get the hash result
    U32* out = RH_STRIDE_GET_DATA(output);
    RH_STRIDE_SET_SIZE(output, 16);
    copy4(out, state);    
} 
