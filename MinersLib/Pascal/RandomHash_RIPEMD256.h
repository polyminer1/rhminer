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


//#include "RandomHash_core.h"

#define RIPEMD160_BLOCK_SIZE 64

inline void Ripemd256RoundFunction(uint32_t* data, uint32_t* state)
{
	uint32_t a, b, c, d, aa, bb, cc, dd;

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	aa = state[4];
	bb = state[5];
	cc = state[6];
	dd = state[7];

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

	aa = aa + (data[7] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
	aa = ROTL32(aa, 7);
	d = d + (data[4] + RH_RIPEMD_C2 + ((aa & b) | (~aa & c)));
	d = ROTL32(d, 6);
	c = c + (data[13] + RH_RIPEMD_C2 + ((d & aa) | (~d & b)));
	c = ROTL32(c, 8);
	b = b + (data[1] + RH_RIPEMD_C2 + ((c & d) | (~c & aa)));
	b = ROTL32(b, 13);
	aa = aa + (data[10] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
	aa = ROTL32(aa, 11);
	d = d + (data[6] + RH_RIPEMD_C2 + ((aa & b) | (~aa & c)));
	d = ROTL32(d, 9);
	c = c + (data[15] + RH_RIPEMD_C2 + ((d & aa) | (~d & b)));
	c = ROTL32(c, 7);
	b = b + (data[3] + RH_RIPEMD_C2 + ((c & d) | (~c & aa)));
	b = ROTL32(b, 15);
	aa = aa + (data[12] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
	aa = ROTL32(aa, 7);
	d = d + (data[0] + RH_RIPEMD_C2 + ((aa & b) | (~aa & c)));
	d = ROTL32(d, 12);
	c = c + (data[9] + RH_RIPEMD_C2 + ((d & aa) | (~d & b)));
	c = ROTL32(c, 15);
	b = b + (data[5] + RH_RIPEMD_C2 + ((c & d) | (~c & aa)));
	b = ROTL32(b, 9);
	aa = aa + (data[2] + RH_RIPEMD_C2 + ((b & c) | (~b & d)));
	aa = ROTL32(aa, 11);
	d = d + (data[14] + RH_RIPEMD_C2 + ((aa & b) | (~aa & c)));
	d = ROTL32(d, 7);
	c = c + (data[11] + RH_RIPEMD_C2 + ((d & aa) | (~d & b)));
	c = ROTL32(c, 13);
	b = b + (data[8] + RH_RIPEMD_C2 + ((c & d) | (~c & aa)));
	b = ROTL32(b, 12);

	a = a + (data[6] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
	a = ROTL32(a, 9);
	dd = dd + (data[11] + RH_RIPEMD_C3 + ((a | ~bb) ^ cc));
	dd = ROTL32(dd, 13);
	cc = cc + (data[3] + RH_RIPEMD_C3 + ((dd | ~a) ^ bb));
	cc = ROTL32(cc, 15);
	bb = bb + (data[7] + RH_RIPEMD_C3 + ((cc | ~dd) ^ a));
	bb = ROTL32(bb, 7);
	a = a + (data[0] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
	a = ROTL32(a, 12);
	dd = dd + (data[13] + RH_RIPEMD_C3 + ((a | ~bb) ^ cc));
	dd = ROTL32(dd, 8);
	cc = cc + (data[5] + RH_RIPEMD_C3 + ((dd | ~a) ^ bb));
	cc = ROTL32(cc, 9);
	bb = bb + (data[10] + RH_RIPEMD_C3 + ((cc | ~dd) ^ a));
	bb = ROTL32(bb, 11);
	a = a + (data[14] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
	a = ROTL32(a, 7);
	dd = dd + (data[15] + RH_RIPEMD_C3 + ((a | ~bb) ^ cc));
	dd = ROTL32(dd, 7);
	cc = cc + (data[8] + RH_RIPEMD_C3 + ((dd | ~a) ^ bb));
	cc = ROTL32(cc, 12);
	bb = bb + (data[12] + RH_RIPEMD_C3 + ((cc | ~dd) ^ a));
	bb = ROTL32(bb, 7);
	a = a + (data[4] + RH_RIPEMD_C3 + ((bb | ~cc) ^ dd));
	a = ROTL32(a, 6);
	dd = dd + (data[9] + RH_RIPEMD_C3 + ((a | ~bb) ^ cc));
	dd = ROTL32(dd, 15);
	cc = cc + (data[1] + RH_RIPEMD_C3 + ((dd | ~a) ^ bb));
	cc = ROTL32(cc, 13);
	bb = bb + (data[2] + RH_RIPEMD_C3 + ((cc | ~dd) ^ a));
	bb = ROTL32(bb, 11);

	aa = aa + (data[3] + RH_RIPEMD_C4 + ((bb | ~c) ^ d));
	aa = ROTL32(aa, 11);
	d = d + (data[10] + RH_RIPEMD_C4 + ((aa | ~bb) ^ c));
	d = ROTL32(d, 13);
	c = c + (data[14] + RH_RIPEMD_C4 + ((d | ~aa) ^ bb));
	c = ROTL32(c, 6);
	bb = bb + (data[4] + RH_RIPEMD_C4 + ((c | ~d) ^ aa));
	bb = ROTL32(bb, 7);
	aa = aa + (data[9] + RH_RIPEMD_C4 + ((bb | ~c) ^ d));
	aa = ROTL32(aa, 14);
	d = d + (data[15] + RH_RIPEMD_C4 + ((aa | ~bb) ^ c));
	d = ROTL32(d, 9);
	c = c + (data[8] + RH_RIPEMD_C4 + ((d | ~aa) ^ bb));
	c = ROTL32(c, 13);
	bb = bb + (data[1] + RH_RIPEMD_C4 + ((c | ~d) ^ aa));
	bb = ROTL32(bb, 15);
	aa = aa + (data[2] + RH_RIPEMD_C4 + ((bb | ~c) ^ d));
	aa = ROTL32(aa, 14);
	d = d + (data[7] + RH_RIPEMD_C4 + ((aa | ~bb) ^ c));
	d = ROTL32(d, 8);
	c = c + (data[0] + RH_RIPEMD_C4 + ((d | ~aa) ^ bb));
	c = ROTL32(c, 13);
	bb = bb + (data[6] + RH_RIPEMD_C4 + ((c | ~d) ^ aa));
	bb = ROTL32(bb, 6);
	aa = aa + (data[13] + RH_RIPEMD_C4 + ((bb | ~c) ^ d));
	aa = ROTL32(aa, 5);
	d = d + (data[11] + RH_RIPEMD_C4 + ((aa | ~bb) ^ c));
	d = ROTL32(d, 12);
	c = c + (data[5] + RH_RIPEMD_C4 + ((d | ~aa) ^ bb));
	c = ROTL32(c, 7);
	bb = bb + (data[12] + RH_RIPEMD_C4 + ((c | ~d) ^ aa));
	bb = ROTL32(bb, 5);

	a = a + (data[15] + RH_RIPEMD_C5 + ((b & cc) | (~b & dd)));
	a = ROTL32(a, 9);
	dd = dd + (data[5] + RH_RIPEMD_C5 + ((a & b) | (~a & cc)));
	dd = ROTL32(dd, 7);
	cc = cc + (data[1] + RH_RIPEMD_C5 + ((dd & a) | (~dd & b)));
	cc = ROTL32(cc, 15);
	b = b + (data[3] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & a)));
	b = ROTL32(b, 11);
	a = a + (data[7] + RH_RIPEMD_C5 + ((b & cc) | (~b & dd)));
	a = ROTL32(a, 8);
	dd = dd + (data[14] + RH_RIPEMD_C5 + ((a & b) | (~a & cc)));
	dd = ROTL32(dd, 6);
	cc = cc + (data[6] + RH_RIPEMD_C5 + ((dd & a) | (~dd & b)));
	cc = ROTL32(cc, 6);
	b = b + (data[9] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & a)));
	b = ROTL32(b, 14);
	a = a + (data[11] + RH_RIPEMD_C5 + ((b & cc) | (~b & dd)));
	a = ROTL32(a, 12);
	dd = dd + (data[8] + RH_RIPEMD_C5 + ((a & b) | (~a & cc)));
	dd = ROTL32(dd, 13);
	cc = cc + (data[12] + RH_RIPEMD_C5 + ((dd & a) | (~dd & b)));
	cc = ROTL32(cc, 5);
	b = b + (data[2] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & a)));
	b = ROTL32(b, 14);
	a = a + (data[10] + RH_RIPEMD_C5 + ((b & cc) | (~b & dd)));
	a = ROTL32(a, 13);
	dd = dd + (data[0] + RH_RIPEMD_C5 + ((a & b) | (~a & cc)));
	dd = ROTL32(dd, 13);
	cc = cc + (data[4] + RH_RIPEMD_C5 + ((dd & a) | (~dd & b)));
	cc = ROTL32(cc, 7);
	b = b + (data[13] + RH_RIPEMD_C5 + ((cc & dd) | (~cc & a)));
	b = ROTL32(b, 5);

	aa = aa + (data[1] + RH_RIPEMD_C6 + ((bb & d) | (cc & ~d)));
	aa = ROTL32(aa, 11);
	d = d + (data[9] + RH_RIPEMD_C6 + ((aa & cc) | (bb & ~cc)));
	d = ROTL32(d, 12);
	cc = cc + (data[11] + RH_RIPEMD_C6 + ((d & bb) | (aa & ~bb)));
	cc = ROTL32(cc, 14);
	bb = bb + (data[10] + RH_RIPEMD_C6 + ((cc & aa) | (d & ~aa)));
	bb = ROTL32(bb, 15);
	aa = aa + (data[0] + RH_RIPEMD_C6 + ((bb & d) | (cc & ~d)));
	aa = ROTL32(aa, 14);
	d = d + (data[8] + RH_RIPEMD_C6 + ((aa & cc) | (bb & ~cc)));
	d = ROTL32(d, 15);
	cc = cc + (data[12] + RH_RIPEMD_C6 + ((d & bb) | (aa & ~bb)));
	cc = ROTL32(cc, 9);
	bb = bb + (data[4] + RH_RIPEMD_C6 + ((cc & aa) | (d & ~aa)));
	bb = ROTL32(bb, 8);
	aa = aa + (data[13] + RH_RIPEMD_C6 + ((bb & d) | (cc & ~d)));
	aa = ROTL32(aa, 9);
	d = d + (data[3] + RH_RIPEMD_C6 + ((aa & cc) | (bb & ~cc)));
	d = ROTL32(d, 14);
	cc = cc + (data[7] + RH_RIPEMD_C6 + ((d & bb) | (aa & ~bb)));
	cc = ROTL32(cc, 5);
	bb = bb + (data[15] + RH_RIPEMD_C6 + ((cc & aa) | (d & ~aa)));
	bb = ROTL32(bb, 6);
	aa = aa + (data[14] + RH_RIPEMD_C6 + ((bb & d) | (cc & ~d)));
	aa = ROTL32(aa, 8);
	d = d + (data[5] + RH_RIPEMD_C6 + ((aa & cc) | (bb & ~cc)));
	d = ROTL32(d, 6);
	cc = cc + (data[6] + RH_RIPEMD_C6 + ((d & bb) | (aa & ~bb)));
	cc = ROTL32(cc, 5);
	bb = bb + (data[2] + RH_RIPEMD_C6 + ((cc & aa) | (d & ~aa)));
	bb = ROTL32(bb, 12);

	a = a + (data[8] + (b ^ c ^ dd));
	a = ROTL32(a, 15);
	dd = dd + (data[6] + (a ^ b ^ c));
	dd = ROTL32(dd, 5);
	c = c + (data[4] + (dd ^ a ^ b));
	c = ROTL32(c, 8);
	b = b + (data[1] + (c ^ dd ^ a));
	b = ROTL32(b, 11);
	a = a + (data[3] + (b ^ c ^ dd));
	a = ROTL32(a, 14);
	dd = dd + (data[11] + (a ^ b ^ c));
	dd = ROTL32(dd, 14);
	c = c + (data[15] + (dd ^ a ^ b));
	c = ROTL32(c, 6);
	b = b + (data[0] + (c ^ dd ^ a));
	b = ROTL32(b, 14);
	a = a + (data[5] + (b ^ c ^ dd));
	a = ROTL32(a, 6);
	dd = dd + (data[12] + (a ^ b ^ c));
	dd = ROTL32(dd, 9);
	c = c + (data[2] + (dd ^ a ^ b));
	c = ROTL32(c, 12);
	b = b + (data[13] + (c ^ dd ^ a));
	b = ROTL32(b, 9);
	a = a + (data[9] + (b ^ c ^ dd));
	a = ROTL32(a, 12);
	dd = dd + (data[7] + (a ^ b ^ c));
	dd = ROTL32(dd, 5);
	c = c + (data[10] + (dd ^ a ^ b));
	c = ROTL32(c, 15);
	b = b + (data[14] + (c ^ dd ^ a));
	b = ROTL32(b, 8);

	state[0]  = state[0] + aa;
	state[1]  = state[1] + bb;
	state[2]  = state[2] + cc;
	state[3]  = state[3] + dd;
	state[4]  = state[4] + a;
	state[5]  = state[5] + b;
	state[6]  = state[6] + c;
	state[7]  = state[7] + d;
}


void RandomHash_RIPEMD256(RH_StridePtr roundInput, RH_StridePtr output)
{
    RH_ALIGN(64) uint32_t state[8] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0x76543210,0xFEDCBA98,0x89ABCDEF,0x01234567 };
    RandomHash_MD_BASE_MAIN_LOOP(RIPEMD160_BLOCK_SIZE, Ripemd256RoundFunction, uint64_t);
    
    //get the hash result
    U32* out = RH_STRIDE_GET_DATA(output);
    RH_STRIDE_SET_SIZE(output, 8 * 4);
    copy8(out, state);
}
