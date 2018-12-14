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

#define SPH_SIZE_haval256_5   256

#define HAVAL_RSTATE \
		s0 = sc.s0; \
		s1 = sc.s1; \
		s2 = sc.s2; \
		s3 = sc.s3; \
		s4 = sc.s4; \
		s5 = sc.s5; \
		s6 = sc.s6; \
		s7 = sc.s7; \

#define HAVAL_WSTATE \
		sc.s0 = s0; \
		sc.s1 = s1; \
		sc.s2 = s2; \
		sc.s3 = s3; \
		sc.s4 = s4; \
		sc.s5 = s5; \
		sc.s6 = s6; \
		sc.s7 = s7; \

#define HAVAL_SAVE_STATE \
		u0 = s0; \
		u1 = s1; \
		u2 = s2; \
		u3 = s3; \
		u4 = s4; \
		u5 = s5; \
		u6 = s6; \
		u7 = s7; \

#define HAVAL_UPDATE_STATE \
		s0 = (s0 + u0); \
		s1 = (s1 + u1); \
		s2 = (s2 + u2); \
		s3 = (s3 + u3); \
		s4 = (s4 + u4); \
		s5 = (s5 + u5); \
		s6 = (s6 + u6); \
		s7 = (s7 + u7); \


#define F1(x6, x5, x4, x3, x2, x1, x0) (((x1) & ((x0) ^ (x4))) ^ ((x2) & (x5)) ^ ((x3) & (x6)) ^ (x0))
#define F2(x6, x5, x4, x3, x2, x1, x0) (((x2) & (((x1) & ~(x3)) ^ ((x4) & (x5)) ^ (x6) ^ (x0))) ^ ((x4) & ((x1) ^ (x5))) ^ ((x3 & (x5)) ^ (x0)))
#define F3(x6, x5, x4, x3, x2, x1, x0) (((x3) & (((x1) & (x2)) ^ (x6) ^ (x0))) ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ (x0))
#define F4(x6, x5, x4, x3, x2, x1, x0) (((x3) & (((x1) & (x2)) ^ ((x4) | (x6)) ^ (x5))) ^ ((x4) & ((~(x2) & (x5)) ^ (x1) ^ (x6) ^ (x0))) ^ ((x2) & (x6)) ^ (x0))
#define F5(x6, x5, x4, x3, x2, x1, x0) (((x0) & ~(((x1) & (x2) & (x3)) ^ (x5))) ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ ((x3) & (x6)))


#define FP3_1(x6, x5, x4, x3, x2, x1, x0) F1(x1, x0, x3, x5, x6, x2, x4)
#define FP3_2(x6, x5, x4, x3, x2, x1, x0) F2(x4, x2, x1, x0, x5, x3, x6)
#define FP3_3(x6, x5, x4, x3, x2, x1, x0) F3(x6, x1, x2, x3, x4, x5, x0)

#define FP4_1(x6, x5, x4, x3, x2, x1, x0) F1(x2, x6, x1, x4, x5, x3, x0)
#define FP4_2(x6, x5, x4, x3, x2, x1, x0) F2(x3, x5, x2, x0, x1, x6, x4)
#define FP4_3(x6, x5, x4, x3, x2, x1, x0) F3(x1, x4, x3, x6, x0, x2, x5)
#define FP4_4(x6, x5, x4, x3, x2, x1, x0) F4(x6, x4, x0, x5, x2, x1, x3)

#define FP5_1(x6, x5, x4, x3, x2, x1, x0) F1(x3, x4, x1, x0, x5, x2, x6)
#define FP5_2(x6, x5, x4, x3, x2, x1, x0) F2(x6, x2, x1, x0, x3, x4, x5)
#define FP5_3(x6, x5, x4, x3, x2, x1, x0) F3(x2, x6, x0, x4, x3, x1, x5)
#define FP5_4(x6, x5, x4, x3, x2, x1, x0) F4(x1, x5, x3, x2, x0, x4, x6)
#define FP5_5(x6, x5, x4, x3, x2, x1, x0) F5(x2, x5, x0, x6, x4, x3, x1)


#define HAVAL_STEP(n, p, x7, x6, x5, x4, x3, x2, x1, x0, w, c)  { \
		uint32_t t = FP ## n ## _ ## p(x6, x5, x4, x3, x2, x1, x0); \
        (x7) = ROTR32(t, 7) + ROTR32((x7), 11) + (w) + (c); \
	}

#define HAVAL_INW(i)   *(uint32_t*)(load_ptr + 4 * (i))

#define HAVAL_PASS1(n) \
   HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 0), (0x00000000)); \
   HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 1), (0x00000000)); \
   HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW( 2), (0x00000000)); \
   HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW( 3), (0x00000000)); \
   HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 4), (0x00000000)); \
   HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW( 5), (0x00000000)); \
   HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW( 6), (0x00000000)); \
   HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW( 7), (0x00000000)); \
 \
   HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 8), (0x00000000)); \
   HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 9), (0x00000000)); \
   HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(10), (0x00000000)); \
   HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(11), (0x00000000)); \
   HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(12), (0x00000000)); \
   HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(13), (0x00000000)); \
   HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(14), (0x00000000)); \
   HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(15), (0x00000000)); \
 \
   HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(16), (0x00000000)); \
   HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(17), (0x00000000)); \
   HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(18), (0x00000000)); \
   HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(19), (0x00000000)); \
   HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(20), (0x00000000)); \
   HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(21), (0x00000000)); \
   HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(22), (0x00000000)); \
   HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(23), (0x00000000)); \
 \
   HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(24), (0x00000000)); \
   HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(25), (0x00000000)); \
   HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(26), (0x00000000)); \
   HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(27), (0x00000000)); \
   HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(28), (0x00000000)); \
   HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(29), (0x00000000)); \
   HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(30), (0x00000000)); \
   HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(31), (0x00000000)); \

#define HAVAL_PASS2(n) \
   HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 5), (0x452821E6)); \
   HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(14), (0x38D01377)); \
   HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(26), (0xBE5466CF)); \
   HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(18), (0x34E90C6C)); \
   HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(11), (0xC0AC29B7)); \
   HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(28), (0xC97C50DD)); \
   HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW( 7), (0x3F84D5B5)); \
   HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(16), (0xB5470917)); \
 \
   HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 0), (0x9216D5D9)); \
   HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(23), (0x8979FB1B)); \
   HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(20), (0xD1310BA6)); \
   HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(22), (0x98DFB5AC)); \
   HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 1), (0x2FFD72DB)); \
   HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(10), (0xD01ADFB7)); \
   HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW( 4), (0xB8E1AFED)); \
   HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW( 8), (0x6A267E96)); \
 \
   HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(30), (0xBA7C9045)); \
   HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 3), (0xF12C7F99)); \
   HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(21), (0x24A19947)); \
   HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW( 9), (0xB3916CF7)); \
   HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(17), (0x0801F2E2)); \
   HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(24), (0x858EFC16)); \
   HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(29), (0x636920D8)); \
   HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW( 6), (0x71574E69)); \
 \
   HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(19), (0xA458FEA3)); \
   HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(12), (0xF4933D7E)); \
   HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(15), (0x0D95748F)); \
   HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(13), (0x728EB658)); \
   HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 2), (0x718BCD58)); \
   HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(25), (0x82154AEE)); \
   HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(31), (0x7B54A41D)); \
   HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(27), (0xC25A59B5)); \


#define HAVAL_PASS3(n) \
   HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(19), (0x9C30D539)); \
   HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 9), (0x2AF26013)); \
   HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW( 4), (0xC5D1B023)); \
   HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(20), (0x286085F0)); \
   HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(28), (0xCA417918)); \
   HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(17), (0xB8DB38EF)); \
   HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW( 8), (0x8E79DCB0)); \
   HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(22), (0x603A180E)); \
 \
   HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(29), (0x6C9E0E8B)); \
   HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(14), (0xB01E8A3E)); \
   HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(25), (0xD71577C1)); \
   HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(12), (0xBD314B27)); \
   HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(24), (0x78AF2FDA)); \
   HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(30), (0x55605C60)); \
   HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(16), (0xE65525F3)); \
   HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(26), (0xAA55AB94)); \
 \
   HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(31), (0x57489862)); \
   HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(15), (0x63E81440)); \
   HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW( 7), (0x55CA396A)); \
   HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW( 3), (0x2AAB10B6)); \
   HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 1), (0xB4CC5C34)); \
   HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW( 0), (0x1141E8CE)); \
   HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(18), (0xA15486AF)); \
   HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(27), (0x7C72E993)); \
 \
   HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(13), (0xB3EE1411)); \
   HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 6), (0x636FBC2A)); \
   HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(21), (0x2BA9C55D)); \
   HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(10), (0x741831F6)); \
   HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(23), (0xCE5C3E16)); \
   HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(11), (0x9B87931E)); \
   HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW( 5), (0xAFD6BA33)); \
   HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW( 2), (0x6C24CF5C)); \

#define HAVAL_PASS4(n) \
   HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(24), (0x7A325381)); \
   HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 4), (0x28958677)); \
   HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW( 0), (0x3B8F4898)); \
   HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(14), (0x6B4BB9AF)); \
   HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 2), (0xC4BFE81B)); \
   HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW( 7), (0x66282193)); \
   HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(28), (0x61D809CC)); \
   HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(23), (0xFB21A991)); \
 \
   HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(26), (0x487CAC60)); \
   HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 6), (0x5DEC8032)); \
   HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(30), (0xEF845D5D)); \
   HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(20), (0xE98575B1)); \
   HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(18), (0xDC262302)); \
   HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(25), (0xEB651B88)); \
   HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(19), (0x23893E81)); \
   HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW( 3), (0xD396ACC5)); \
 \
   HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(22), (0x0F6D6FF3)); \
   HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(11), (0x83F44239)); \
   HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(31), (0x2E0B4482)); \
   HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(21), (0xA4842004)); \
   HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 8), (0x69C8F04A)); \
   HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(27), (0x9E1F9B5E)); \
   HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(12), (0x21C66842)); \
   HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW( 9), (0xF6E96C9A)); \
 \
   HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 1), (0x670C9C61)); \
   HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(29), (0xABD388F0)); \
   HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW( 5), (0x6A51A0D2)); \
   HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(15), (0xD8542F68)); \
   HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(17), (0x960FA728)); \
   HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(10), (0xAB5133A3)); \
   HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(16), (0x6EEF0B6C)); \
   HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(13), (0x137A3BE4)); \

#define HAVAL_PASS5(n) \
   HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(27), (0xBA3BF050)); \
   HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 3), (0x7EFB2A98)); \
   HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(21), (0xA1F1651D)); \
   HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(26), (0x39AF0176)); \
   HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(17), (0x66CA593E)); \
   HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW(11), (0x82430E88)); \
   HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(20), (0x8CEE8619)); \
   HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(29), (0x456F9FB4)); \
 \
   HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW(19), (0x7D84A5C3)); \
   HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 0), (0x3B8B5EBE)); \
   HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(12), (0xE06F75D8)); \
   HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW( 7), (0x85C12073)); \
   HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(13), (0x401A449F)); \
   HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW( 8), (0x56C16AA6)); \
   HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(31), (0x4ED3AA62)); \
   HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(10), (0x363F7706)); \
 \
   HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 5), (0x1BFEDF72)); \
   HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW( 9), (0x429B023D)); \
   HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(14), (0x37D0D724)); \
   HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(30), (0xD00A1248)); \
   HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW(18), (0xDB0FEAD3)); \
   HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW( 6), (0x49F1C09B)); \
   HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(28), (0x075372C9)); \
   HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(24), (0x80991B7B)); \
 \
   HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL_INW( 2), (0x25D479D8)); \
   HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL_INW(23), (0xF6E8DEF7)); \
   HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL_INW(16), (0xE3FE501A)); \
   HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL_INW(22), (0xB6794C3B)); \
   HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL_INW( 4), (0x976CE0BD)); \
   HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL_INW( 1), (0x04C006BA)); \
   HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL_INW(25), (0xC1A94FB6)); \
   HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL_INW(15), (0x409F60C4)); \

struct RH_ALIGN(64) RH_sph_haval_context {
	unsigned char buf[128];    /* first field, for alignment */
	uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
	unsigned passes;
	uint64_t count;
};


void CUDA_SYM_DECL(RandomHash_Haval_5_256)(RH_StridePtr roundInput, RH_StridePtr output)
{
    RH_ALIGN(64) RH_sph_haval_context sc;
    //init
    const unsigned passes = 5;
    const unsigned olen = 8;
	sc.s0 = (0x243F6A88);
	sc.s1 = (0x85A308D3);
	sc.s2 = (0x13198A2E);
	sc.s3 = (0x03707344);
	sc.s4 = (0xA4093822);
	sc.s5 = (0x299F31D0);
	sc.s6 = (0x082EFA98);
	sc.s7 = (0xEC4E6C89);
	sc.passes = passes;
	sc.count = 0;

    //haval _short
	unsigned current;
    uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
	uint32_t u0, u1, u2, u3, u4, u5, u6, u7;

    const unsigned char *data = RH_STRIDE_GET_DATA(roundInput);
    size_t len = RH_STRIDE_GET_SIZE(roundInput);

    uint32_t orig_len = (U32)len;
    //haval()
	HAVAL_RSTATE;
	while (len >= 128U) {
        const unsigned char *const load_ptr = (const unsigned char *)(data);
		HAVAL_SAVE_STATE;
		HAVAL_PASS1(5);
		HAVAL_PASS2(5);
		HAVAL_PASS3(5);
		HAVAL_PASS4(5);
		HAVAL_PASS5(5);
		HAVAL_UPDATE_STATE;

		data = (const unsigned char *)data + 128U;
		len -= 128U;
	}
	HAVAL_WSTATE;
	if (len > 0)
		memcpy(sc.buf, data, len);

	sc.count += (uint64_t)orig_len;

    //haval_close
	current = (unsigned)sc.count & 127U;
	sc.buf[current ++] = (0x01 << 0) | ((0 & 0xFF) >> (8 - 0));
	HAVAL_RSTATE;
	if (current > 118U) {
		memset(sc.buf + current, 0, 128U - current);
        const unsigned char *const load_ptr = (const unsigned char *)(sc.buf);
		HAVAL_SAVE_STATE;
		HAVAL_PASS1(5);
		HAVAL_PASS2(5);
		HAVAL_PASS3(5);
		HAVAL_PASS4(5);
		HAVAL_PASS5(5);
		HAVAL_UPDATE_STATE;
		current = 0;
	}
	memset(sc.buf + current, 0, 118U - current);
	sc.buf[118] = 0x01 | (5 << 3);
	sc.buf[119] = olen << 3;
	*(uint64_t*)(sc.buf + 120) = RHMINER_T64(sc.count << 3);
    const unsigned char *const load_ptr = (const unsigned char *)(sc.buf);
	HAVAL_SAVE_STATE;
	HAVAL_PASS1(5);
	HAVAL_PASS2(5);
	HAVAL_PASS3(5);
	HAVAL_PASS4(5);
	HAVAL_PASS5(5);

	//haval_out(sc, dst);    
    RH_STRIDE_SET_SIZE(output, 32);
    uint32_t* buf = (uint32_t*)RH_STRIDE_GET_DATA(output);
	buf[0] = (s0 + u0);
	buf[1] = (s1 + u1);
	buf[2] = (s2 + u2);
	buf[3] = (s3 + u3);
	buf[4] = (s4 + u4);
	buf[5] = (s5 + u5);
	buf[6] = (s6 + u6);
	buf[7] = (s7 + u7);
}
