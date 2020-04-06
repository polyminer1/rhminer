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

#define HAVAL5_256_HAVAL_RSTATE \
		s0 = sc.s0; \
		s1 = sc.s1; \
		s2 = sc.s2; \
		s3 = sc.s3; \
		s4 = sc.s4; \
		s5 = sc.s5; \
		s6 = sc.s6; \
		s7 = sc.s7; \

#define HAVAL5_256_HAVAL_WSTATE \
		sc.s0 = s0; \
		sc.s1 = s1; \
		sc.s2 = s2; \
		sc.s3 = s3; \
		sc.s4 = s4; \
		sc.s5 = s5; \
		sc.s6 = s6; \
		sc.s7 = s7; \

#define HAVAL5_256_HAVAL_SAVE_STATE \
		u0 = s0; \
		u1 = s1; \
		u2 = s2; \
		u3 = s3; \
		u4 = s4; \
		u5 = s5; \
		u6 = s6; \
		u7 = s7; \

#define HAVAL5_256_HAVAL_UPDATE_STATE \
		s0 = (s0 + u0); \
		s1 = (s1 + u1); \
		s2 = (s2 + u2); \
		s3 = (s3 + u3); \
		s4 = (s4 + u4); \
		s5 = (s5 + u5); \
		s6 = (s6 + u6); \
		s7 = (s7 + u7); \


#define HAVAL5_256_F1(x6, x5, x4, x3, x2, x1, x0) (((x1) & ((x0) ^ (x4))) ^ ((x2) & (x5)) ^ ((x3) & (x6)) ^ (x0))
#define HAVAL5_256_F2(x6, x5, x4, x3, x2, x1, x0) (((x2) & (((x1) & ~(x3)) ^ ((x4) & (x5)) ^ (x6) ^ (x0))) ^ ((x4) & ((x1) ^ (x5))) ^ ((x3 & (x5)) ^ (x0)))
#define HAVAL5_256_F3(x6, x5, x4, x3, x2, x1, x0) (((x3) & (((x1) & (x2)) ^ (x6) ^ (x0))) ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ (x0))
#define HAVAL5_256_F4(x6, x5, x4, x3, x2, x1, x0) (((x3) & (((x1) & (x2)) ^ ((x4) | (x6)) ^ (x5))) ^ ((x4) & ((~(x2) & (x5)) ^ (x1) ^ (x6) ^ (x0))) ^ ((x2) & (x6)) ^ (x0))
#define HAVAL5_256_F5(x6, x5, x4, x3, x2, x1, x0) (((x0) & ~(((x1) & (x2) & (x3)) ^ (x5))) ^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ ((x3) & (x6)))


#define HAVAL5_256_FP3_1(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F1(x1, x0, x3, x5, x6, x2, x4)
#define HAVAL5_256_FP3_2(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F2(x4, x2, x1, x0, x5, x3, x6)
#define HAVAL5_256_FP3_3(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F3(x6, x1, x2, x3, x4, x5, x0)

#define HAVAL5_256_FP4_1(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F1(x2, x6, x1, x4, x5, x3, x0)
#define HAVAL5_256_FP4_2(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F2(x3, x5, x2, x0, x1, x6, x4)
#define HAVAL5_256_FP4_3(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F3(x1, x4, x3, x6, x0, x2, x5)
#define HAVAL5_256_FP4_4(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F4(x6, x4, x0, x5, x2, x1, x3)

#define HAVAL5_256_FP5_1(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F1(x3, x4, x1, x0, x5, x2, x6)
#define HAVAL5_256_FP5_2(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F2(x6, x2, x1, x0, x3, x4, x5)
#define HAVAL5_256_FP5_3(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F3(x2, x6, x0, x4, x3, x1, x5)
#define HAVAL5_256_FP5_4(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F4(x1, x5, x3, x2, x0, x4, x6)
#define HAVAL5_256_FP5_5(x6, x5, x4, x3, x2, x1, x0) HAVAL5_256_F5(x2, x5, x0, x6, x4, x3, x1)


#define HAVAL5_256_HAVAL_STEP(n, p, x7, x6, x5, x4, x3, x2, x1, x0, w, c)  { \
		uint32_t t = HAVAL5_256_FP ## n ## _ ## p(x6, x5, x4, x3, x2, x1, x0); \
        (x7) = ROTR32(t, 7) + ROTR32((x7), 11) + (w) + (c); \
	}

#define HAVAL5_256_HAVAL_INW(i)   *(uint32_t*)(load_ptr + 4 * (i))

#define HAVAL5_256_HAVAL_PASS1(n) \
   HAVAL5_256_HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 0), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 1), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW( 2), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW( 3), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 4), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW( 5), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW( 6), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW( 7), (0x00000000)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 8), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 9), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(10), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(11), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(12), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(13), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(14), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(15), (0x00000000)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(16), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(17), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(18), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(19), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(20), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(21), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(22), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(23), (0x00000000)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(24), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(25), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(26), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(27), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(28), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(29), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(30), (0x00000000)); \
   HAVAL5_256_HAVAL_STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(31), (0x00000000)); \

#define HAVAL5_256_HAVAL_PASS2(n) \
   HAVAL5_256_HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 5), (0x452821E6)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(14), (0x38D01377)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(26), (0xBE5466CF)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(18), (0x34E90C6C)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(11), (0xC0AC29B7)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(28), (0xC97C50DD)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW( 7), (0x3F84D5B5)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(16), (0xB5470917)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 0), (0x9216D5D9)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(23), (0x8979FB1B)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(20), (0xD1310BA6)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(22), (0x98DFB5AC)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 1), (0x2FFD72DB)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(10), (0xD01ADFB7)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW( 4), (0xB8E1AFED)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW( 8), (0x6A267E96)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(30), (0xBA7C9045)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 3), (0xF12C7F99)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(21), (0x24A19947)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW( 9), (0xB3916CF7)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(17), (0x0801F2E2)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(24), (0x858EFC16)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(29), (0x636920D8)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW( 6), (0x71574E69)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(19), (0xA458FEA3)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(12), (0xF4933D7E)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(15), (0x0D95748F)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(13), (0x728EB658)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 2), (0x718BCD58)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(25), (0x82154AEE)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(31), (0x7B54A41D)); \
   HAVAL5_256_HAVAL_STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(27), (0xC25A59B5)); \


#define HAVAL5_256_HAVAL_PASS3(n) \
   HAVAL5_256_HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(19), (0x9C30D539)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 9), (0x2AF26013)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW( 4), (0xC5D1B023)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(20), (0x286085F0)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(28), (0xCA417918)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(17), (0xB8DB38EF)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW( 8), (0x8E79DCB0)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(22), (0x603A180E)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(29), (0x6C9E0E8B)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(14), (0xB01E8A3E)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(25), (0xD71577C1)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(12), (0xBD314B27)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(24), (0x78AF2FDA)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(30), (0x55605C60)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(16), (0xE65525F3)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(26), (0xAA55AB94)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(31), (0x57489862)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(15), (0x63E81440)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW( 7), (0x55CA396A)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW( 3), (0x2AAB10B6)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 1), (0xB4CC5C34)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW( 0), (0x1141E8CE)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(18), (0xA15486AF)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(27), (0x7C72E993)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(13), (0xB3EE1411)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 6), (0x636FBC2A)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(21), (0x2BA9C55D)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(10), (0x741831F6)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(23), (0xCE5C3E16)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(11), (0x9B87931E)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW( 5), (0xAFD6BA33)); \
   HAVAL5_256_HAVAL_STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW( 2), (0x6C24CF5C)); \

#define HAVAL5_256_HAVAL_PASS4(n) \
   HAVAL5_256_HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(24), (0x7A325381)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 4), (0x28958677)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW( 0), (0x3B8F4898)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(14), (0x6B4BB9AF)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 2), (0xC4BFE81B)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW( 7), (0x66282193)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(28), (0x61D809CC)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(23), (0xFB21A991)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(26), (0x487CAC60)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 6), (0x5DEC8032)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(30), (0xEF845D5D)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(20), (0xE98575B1)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(18), (0xDC262302)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(25), (0xEB651B88)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(19), (0x23893E81)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW( 3), (0xD396ACC5)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(22), (0x0F6D6FF3)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(11), (0x83F44239)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(31), (0x2E0B4482)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(21), (0xA4842004)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 8), (0x69C8F04A)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(27), (0x9E1F9B5E)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(12), (0x21C66842)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW( 9), (0xF6E96C9A)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 1), (0x670C9C61)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(29), (0xABD388F0)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW( 5), (0x6A51A0D2)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(15), (0xD8542F68)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(17), (0x960FA728)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(10), (0xAB5133A3)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(16), (0x6EEF0B6C)); \
   HAVAL5_256_HAVAL_STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(13), (0x137A3BE4)); \

#define HAVAL_PASS5(n) \
   HAVAL5_256_HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(27), (0xBA3BF050)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 3), (0x7EFB2A98)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(21), (0xA1F1651D)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(26), (0x39AF0176)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(17), (0x66CA593E)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW(11), (0x82430E88)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(20), (0x8CEE8619)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(29), (0x456F9FB4)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW(19), (0x7D84A5C3)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 0), (0x3B8B5EBE)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(12), (0xE06F75D8)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW( 7), (0x85C12073)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(13), (0x401A449F)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW( 8), (0x56C16AA6)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(31), (0x4ED3AA62)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(10), (0x363F7706)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 5), (0x1BFEDF72)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW( 9), (0x429B023D)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(14), (0x37D0D724)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(30), (0xD00A1248)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW(18), (0xDB0FEAD3)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW( 6), (0x49F1C09B)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(28), (0x075372C9)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(24), (0x80991B7B)); \
 \
   HAVAL5_256_HAVAL_STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, HAVAL5_256_HAVAL_INW( 2), (0x25D479D8)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, HAVAL5_256_HAVAL_INW(23), (0xF6E8DEF7)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, HAVAL5_256_HAVAL_INW(16), (0xE3FE501A)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, HAVAL5_256_HAVAL_INW(22), (0xB6794C3B)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, HAVAL5_256_HAVAL_INW( 4), (0x976CE0BD)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, HAVAL5_256_HAVAL_INW( 1), (0x04C006BA)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, HAVAL5_256_HAVAL_INW(25), (0xC1A94FB6)); \
   HAVAL5_256_HAVAL_STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, HAVAL5_256_HAVAL_INW(15), (0x409F60C4)); \


static inline U32 mix128(U32 a0, U32 a1, U32 a2, U32 a3, int n)
{
    U32 tmp;

    tmp = (a0 & U32(0x000000FF))
        | (a1 & U32(0x0000FF00))
        | (a2 & U32(0x00FF0000))
        | (a3 & U32(0xFF000000));
    if (n > 0)
        tmp = ROTL32(tmp, n);
    return tmp;
}

static inline U32 mix160_0(U32 x5, U32 x6, U32 x7)
{
    U32 tmp;

    tmp = (x5 & U32(0x01F80000))
        | (x6 & U32(0xFE000000))
        | (x7 & U32(0x0000003F));
    return ROTL32(tmp, 13);
}

static inline U32 mix160_1(U32 x5, U32 x6, U32 x7)
{
    U32 tmp;

    tmp = (x5 & U32(0xFE000000))
        | (x6 & U32(0x0000003F))
        | (x7 & U32(0x00000FC0));
    return ROTL32(tmp, 7);
}

static inline U32 mix160_2(U32 x5, U32 x6, U32 x7)
{
    U32 tmp;

    tmp = (x5 & U32(0x0000003F))
        | (x6 & U32(0x00000FC0))
        | (x7 & U32(0x0007F000));
    return tmp;
}

static inline U32 mix160_3(U32 x5, U32 x6, U32 x7)
{
    U32 tmp;

    tmp = (x5 & U32(0x00000FC0))
        | (x6 & U32(0x0007F000))
        | (x7 & U32(0x01F80000));
    return tmp >> 6;
}

static inline U32 mix160_4(U32 x5, U32 x6, U32 x7)
{
    U32 tmp;

    tmp = (x5 & U32(0x0007F000))
        | (x6 & U32(0x01F80000))
        | (x7 & U32(0xFE000000));
    return tmp >> 12;
}

static inline U32 mix192_0(U32 x6, U32 x7)
{
    U32 tmp;

    tmp = (x6 & U32(0xFC000000)) | (x7 & U32(0x0000001F));
    return ROTL32(tmp, 6);
}

static inline U32 mix192_1(U32 x6, U32 x7)
{
    return (x6 & U32(0x0000001F)) | (x7 & U32(0x000003E0));
}

static inline U32 mix192_2(U32 x6, U32 x7)
{
    return ((x6 & U32(0x000003E0)) | (x7 & U32(0x0000FC00))) >> 5;
}

static inline U32 mix192_3(U32 x6, U32 x7)
{
    return ((x6 & U32(0x0000FC00)) | (x7 & U32(0x001F0000))) >> 10;
}

static inline U32 mix192_4(U32 x6, U32 x7)
{
    return ((x6 & U32(0x001F0000)) | (x7 & U32(0x03E00000))) >> 16;
}

static inline U32 mix192_5(U32 x6, U32 x7)
{
    return ((x6 & U32(0x03E00000)) | (x7 & U32(0xFC000000))) >> 21;
}

struct RH_ALIGN(64) RH_sph_haval_context {
	unsigned char buf[128];    /* first field, for alignment */
	uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
	unsigned passes;
	uint64_t count;
};


void RandomHash_Haval_5_256(RH_StridePtr roundInput, RH_StridePtr output, U32 bitSize)
{
    RH_ALIGN(64) RH_sph_haval_context sc;
    //init
    const unsigned passes = 5;
    const unsigned olen = bitSize >> 5;
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

    const unsigned char *data = RH_STRIDE_GET_DATA8(roundInput);
    size_t len = RH_STRIDE_GET_SIZE(roundInput);

    uint32_t orig_len = (U32)len;
    //haval()
	HAVAL5_256_HAVAL_RSTATE;
	while (len >= 128U) {
        const unsigned char *const load_ptr = (const unsigned char *)(data);
		HAVAL5_256_HAVAL_SAVE_STATE;
		HAVAL5_256_HAVAL_PASS1(5);
		HAVAL5_256_HAVAL_PASS2(5);
		HAVAL5_256_HAVAL_PASS3(5);
		HAVAL5_256_HAVAL_PASS4(5);
		HAVAL_PASS5(5);
		HAVAL5_256_HAVAL_UPDATE_STATE;

		data = (const unsigned char *)data + 128U;
		len -= 128U;
	}
	HAVAL5_256_HAVAL_WSTATE;
	if (len > 0)
		memcpy(sc.buf, data, len);

	sc.count += (uint64_t)orig_len;

    //haval_close
	current = (unsigned)sc.count & 127U;
	sc.buf[current ++] = (0x01 << 0) | ((0 & 0xFF) >> (8 - 0));
	HAVAL5_256_HAVAL_RSTATE;
	if (current > 118U) {
		memset(sc.buf + current, 0, 128U - current);
        const unsigned char *const load_ptr = (const unsigned char *)(sc.buf);
		HAVAL5_256_HAVAL_SAVE_STATE;
		HAVAL5_256_HAVAL_PASS1(5);
		HAVAL5_256_HAVAL_PASS2(5);
		HAVAL5_256_HAVAL_PASS3(5);
		HAVAL5_256_HAVAL_PASS4(5);
		HAVAL_PASS5(5);
		HAVAL5_256_HAVAL_UPDATE_STATE;
		current = 0;
	}
	memset(sc.buf + current, 0, 118U - current);
	sc.buf[118] = 0x01 | (5 << 3);
	sc.buf[119] = olen << 3;
	*(uint64_t*)(sc.buf + 120) = RHMINER_T64(sc.count << 3);
    const unsigned char *const load_ptr = (const unsigned char *)(sc.buf);
	HAVAL5_256_HAVAL_SAVE_STATE;
	HAVAL5_256_HAVAL_PASS1(5);
	HAVAL5_256_HAVAL_PASS2(5);
	HAVAL5_256_HAVAL_PASS3(5);
	HAVAL5_256_HAVAL_PASS4(5);
	HAVAL_PASS5(5);


    //---------------------
    s0 = (s0 + u0);
    s1 = (s1 + u1);
    s2 = (s2 + u2);
    s3 = (s3 + u3);
    s4 = (s4 + u4);
    s5 = (s5 + u5);
    s6 = (s6 + u6);
    s7 = (s7 + u7);
	//haval_out(sc, dst);    
    RH_STRIDE_SET_SIZE(output, bitSize >> 3);
    uint32_t* buf = RH_STRIDE_GET_DATA(output);
    switch (olen) 
    {
    case 4:
        buf[0] = U32(s0 + mix128(s7, s4, s5, s6, 24));
        buf[1] = U32(s1 + mix128(s6, s7, s4, s5, 16));
        buf[2] = U32(s2 + mix128(s5, s6, s7, s4, 8));
        buf[3] = U32(s3 + mix128(s4, s5, s6, s7, 0));
        break;
    case 5:
        buf[0] = U32(s0 + mix160_0(s5, s6, s7));
        buf[1] = U32(s1 + mix160_1(s5, s6, s7));
        buf[2] = U32(s2 + mix160_2(s5, s6, s7));
        buf[3] = U32(s3 + mix160_3(s5, s6, s7));
        buf[4] = U32(s4 + mix160_4(s5, s6, s7));
        break;
    case 6:
        buf[0] = U32(s0 + mix192_0(s6, s7));
        buf[1] = U32(s1 + mix192_1(s6, s7));
        buf[2] = U32(s2 + mix192_2(s6, s7));
        buf[3] = U32(s3 + mix192_3(s6, s7));
        buf[4] = U32(s4 + mix192_4(s6, s7));
        buf[5] = U32(s5 + mix192_5(s6, s7));
        break;
    case 7:
        buf[0] = U32(s0 + ((s7 >> 27) & 0x1F));
        buf[1] = U32(s1 + ((s7 >> 22) & 0x1F));
        buf[2] = U32(s2 + ((s7 >> 18) & 0x0F));
        buf[3] = U32(s3 + ((s7 >> 13) & 0x1F));
        buf[4] = U32(s4 + ((s7 >> 9) & 0x0F));
        buf[5] = U32(s5 + ((s7 >> 4) & 0x1F));
        buf[6] = U32(s6 + ((s7) & 0x0F));
        break;
    case 8:
        /*
        sph_enc32le(buf, s0);
        sph_enc32le(buf + 4, s1);
        sph_enc32le(buf + 8, s2);
        sph_enc32le(buf + 12, s3);
        sph_enc32le(buf + 16, s4);
        sph_enc32le(buf + 20, s5);
        sph_enc32le(buf + 24, s6);
        sph_enc32le(buf + 28, s7);*/
        buf[0] = s0;
        buf[1] = s1;
        buf[2] = s2;
        buf[3] = s3;
        buf[4] = s4;
        buf[5] = s5;
        buf[6] = s6;
        buf[7] = s7;
        break;
    }
}
