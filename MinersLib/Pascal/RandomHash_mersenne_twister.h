/* 
 * The Mersenne Twister pseudo-random number generator (PRNG)
 *
 * This is an implementation of fast PRNG called MT19937, meaning it has a
 * period of 2^19937-1, which is a Mersenne prime.
 *
 * This PRNG is fast and suitable for non-cryptographic code.  For instance, it
 * would be perfect for Monte Carlo simulations, etc.
 *
 * For all the details on this algorithm, see the original paper:
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/ARTICLES/mt.pdf
 *
 * Written by Christian Stigen Larsen
 * Distributed under the modified BSD license.
 * 2015-02-17, 2017-12-06
 */

// Better on older Intel Core i7, but worse on newer Intel Xeon CPUs (undefine
// it on those).
//#define MT_UNROLL_MORE

/*
 * We have an array of 624 32-bit values, and there are 31 unused bits, so we
 * have a seed value of 624*32-31 = 19937 bits.
 */
#ifndef RANDOM_HASH_mersenne_twister_H
#define RANDOM_HASH_mersenne_twister_H

#if defined(_MSC_VER) || defined(__GNUC__)
    #include <stdlib.h>
#endif

#define MERSENNE_TWISTER_SIZE 624


struct mersenne_twister_state 
{
  uint32_t MT[MERSENNE_TWISTER_SIZE];
  uint32_t MT_TEMPERED[MERSENNE_TWISTER_SIZE];
  size_t index = MERSENNE_TWISTER_SIZE;
  U32    seed = 0;
};


#define  MERSENNE_TWISTER_PERIOD    397
#define  MERSENNE_TWISTER_DIFF      (MERSENNE_TWISTER_SIZE - MERSENNE_TWISTER_PERIOD)
#define  MERSENNE_TWISTER_MAGIC     0x9908b0df

#define M32(x) (0x80000000 & (x)) 
#define L31(x) (0x7FFFFFFF & (x)) 
#define M32_64pak(x) (0x80000000 & (U32)(x)) 
#define L31_64pak(x) (0x7FFFFFFF & (U32)(x>>32)) 


#define UNROLL(expr) \
  y = M32(state->MT[i]) | L31(state->MT[i+1]); \
  state->MT[i] = state->MT[expr] ^ (y >> 1) ^ (((int32_t(y) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); \
  ++i;



inline void merssen_twister_seed(uint32_t value, mersenne_twister_state* state)
{
    /*
     * The equation below is a linear congruential generator (LCG), one of the
     * oldest known pseudo-random number generator algorithms, in the form
     * X_(n+1) = = (a*X_n + c) (mod m).
     *
     * We've implicitly got m=32 (mask + word size of 32 bits), so there is no
     * need to explicitly use modulus.
     *
     * What is interesting is the multiplier a.  The one we have below is
     * 0x6c07865 --- 1812433253 in decimal, and is called the Borosh-Niederreiter
     * multiplier for modulus 2^32.
     *
     * It is mentioned in passing in Knuth's THE ART OF COMPUTER
     * PROGRAMMING, Volume 2, page 106, Table 1, line 13.  LCGs are
     * treated in the same book, pp. 10-26
     *
     * You can read the original paper by Borosh and Niederreiter as well.  It's
     * called OPTIMAL MULTIPLIERS FOR PSEUDO-RANDOM NUMBER GENERATION BY THE
     * LINEAR CONGRUENTIAL METHOD (1983) at
     * http://www.springerlink.com/content/n7765ku70w8857l7/
     *
     * You can read about LCGs at:
     * http://en.wikipedia.org/wiki/Linear_congruential_generator
     *
     * From that page, it says: "A common Mersenne twister implementation,
     * interestingly enough, uses an LCG to generate seed data.",
     *
     * Since we're using 32-bits data types for our MT array, we can skip the
     * masking with 0xFFFFFFFF below.
     */

    state->MT[0] = value;
    state->index = MERSENNE_TWISTER_SIZE;

    for (uint_fast32_t i = 1; i < MERSENNE_TWISTER_SIZE; ++i)
        state->MT[i] = 0x6c078965 * (state->MT[i - 1] ^ state->MT[i - 1] >> 30) + i;
}


inline uint32_t merssen_twister_rand(mersenne_twister_state* state)
{
    if (state->index == MERSENNE_TWISTER_SIZE) 
    {
      /*
       * For performance reasons, we've unrolled the loop three times, thus
       * mitigating the need for any modulus operations. Anyway, it seems this
       * trick is old hat: http://www.quadibloc.com/crypto/co4814.htm
       */

      size_t i = 0;
      uint32_t y;

      // i = [0 ... 226]
      while ( i < MERSENNE_TWISTER_DIFF ) {
        /*
         * We're doing 226 = 113*2, an even number of steps, so we can safely
         * unroll one more step here for speed:
         */
        UNROLL(i+MERSENNE_TWISTER_PERIOD);

    #ifdef MT_UNROLL_MORE
        UNROLL(i+MERSENNE_TWISTER_PERIOD);
    #endif
      }

      // i = [227 ... 622]
      while ( i < MERSENNE_TWISTER_SIZE -1 ) {
        /*
         * 623-227 = 396 = 2*2*3*3*11, so we can unroll this loop in any number
         * that evenly divides 396 (2, 4, 6, etc). Here we'll unroll 11 times.
         */
        UNROLL(i-MERSENNE_TWISTER_DIFF);

    #ifdef MT_UNROLL_MORE
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
        UNROLL(i-MERSENNE_TWISTER_DIFF);
    #endif
      }

      {
        // i = 623, last step rolls over
        y = M32(state->MT[MERSENNE_TWISTER_SIZE-1]) | L31(state->MT[0]);
        state->MT[MERSENNE_TWISTER_SIZE-1] = state->MT[MERSENNE_TWISTER_PERIOD-1] ^ (y >> 1) ^ (((int32_t(y) << 31) >>
              31) & MERSENNE_TWISTER_MAGIC);
      }
      // Temper all numbers in a batch
      for (size_t i = 0; i < MERSENNE_TWISTER_SIZE; ++i) {
        y = state->MT[i];
        y ^= y >> 11;
        y ^= y << 7  & 0x9d2c5680;
        y ^= y << 15 & 0xefc60000;
        y ^= y >> 18;
        state->MT_TEMPERED[i] = y;
      }

      state->index = 0;
    }
        
    return state->MT_TEMPERED[state->index++];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define RH_MERSSEN_TWISTER_MERGE { \
            while(mt != end) \
            { \
                U32 y = *mt; \
                y ^= y >> 11; \
                y ^= y << 7 & 0x9d2c5680; \
                y ^= y << 15 & 0xefc60000; \
                y ^= y >> 18; \
                *out = y; \
                out++; \
                mt++; \
            }}
            
#define RH_MERSSEN_TWISTER_MERGE_SSE { \
            __m128i y,r0, r1; \
            __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);   \
            __m128i c2 = _mm_cvtsi32_si128(0xefc60000); \
            c1 = _mm_shuffle_epi32(c1, 0); \
            c2 = _mm_shuffle_epi32(c2, 0); \
            while(mt != end) \
            {  \
                y = RH_MM_LOAD128((__m128i*)mt); \
                mt += sizeof(__m128i)/4; \
                r0 = _mm_srli_epi32(y, 11); \
                y = _mm_xor_si128(y, r0); \
                r0 = _mm_slli_epi32(y, 7); \
                r1 = _mm_and_si128(r0, c1); \
                y = _mm_xor_si128(y, r1); \
                r0 = _mm_slli_epi32(y, 15); \
                r1 = _mm_and_si128(r0, c2); \
                y = _mm_xor_si128(y, r1); \
                r0 = _mm_srli_epi32(y, 18); \
                y = _mm_xor_si128(y, r0); \
                RH_MM_STORE128((__m128i*)out, y); \
                out += sizeof(__m128i)/4; \
            }}


#define RH_MERSSEN_TWISTER_MERGE_SSE3 RH_MERSSEN_TWISTER_MERGE 

extern bool g_isSSE3Supported;
extern bool g_isSSE4Supported;
extern bool g_isAVX2Supported;

inline void merssen_twister_seed_fast(uint32_t value, mersenne_twister_state* state)
{
    state->index = MERSENNE_TWISTER_SIZE;
    U32* mt = state->MT;
    U32* end = mt + MERSENNE_TWISTER_SIZE;

    U32 pval = (0x6c078965 * (value ^ value >> 30) + 1);
    U64 val64 = ((U64)pval) << 32 | value;
    *(U64*)mt = val64;
    mt += 2;

    U32 i = 2;
    while (mt != end)
    {
        val64 = 0x6c078965 * (pval ^ pval >> 30) + i;
        pval = (U32)val64;
        i++;
        pval = 0x6c078965 * (pval ^ pval >> 30) + i;
        val64 |= ((U64)pval) << 32;
        *(U64*)mt = val64;
        mt += 2;
        i++;
    }
}


inline uint32_t merssen_twister_rand_fast(mersenne_twister_state* state)
{
    if (state->index == MERSENNE_TWISTER_SIZE)
    {
        size_t i = 0;
        uint32_t y;
        U32* mt;
        U32* end;

        while (i < MERSENNE_TWISTER_DIFF)
        {
            y = M32(state->MT[i]) | L31(state->MT[i + 1]);
            state->MT[i] = state->MT[(i + MERSENNE_TWISTER_PERIOD)] ^ (y >> 1) ^ (((int32_t(y) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);
            ++i;
        }
        while (i < MERSENNE_TWISTER_SIZE - 1)
        {
            y = M32(state->MT[i]) | L31(state->MT[i + 1]);
            state->MT[i] = state->MT[(i - MERSENNE_TWISTER_DIFF)] ^ (y >> 1) ^ (((int32_t(y) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);
            ++i;
        }
        
        {
            y = M32(state->MT[MERSENNE_TWISTER_SIZE - 1]) | L31(state->MT[0]);
            state->MT[MERSENNE_TWISTER_SIZE - 1] = state->MT[MERSENNE_TWISTER_PERIOD - 1] ^ (y >> 1) ^
                (((int32_t(y) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);
        }

        mt = state->MT; 
        end = mt + MERSENNE_TWISTER_SIZE;
        U32* out = state->MT_TEMPERED;
        state->index = 0;

#if defined(RHMINER_ENABLE_SSE4)
    #if defined(RHMINER_COND_SSE4)
        if (g_isSSE4Supported || g_isSSE3Supported)
            RH_MERSSEN_TWISTER_MERGE_SSE
        else
            RH_MERSSEN_TWISTER_MERGE
    #else
        RH_MERSSEN_TWISTER_MERGE_SSE
    #endif 
#else
        if (g_isSSE3Supported)
            RH_MERSSEN_TWISTER_MERGE_SSE
        else
            RH_MERSSEN_TWISTER_MERGE
#endif    
        
    }
    return state->MT_TEMPERED[state->index++];
}

#define merssen_twister_rand_fast_partial merssen_twister_rand_fast_partial_slow
#define merssen_twister_rand_fast_partial_204(S) merssen_twister_rand_fast_partial(S, 204)

inline uint32_t merssen_twister_rand_fast_partial_slow(mersenne_twister_state* state, const int MaxPrecalc)
{
    if (state->index == MERSENNE_TWISTER_SIZE)
    {
        const U32 A = RHMINER_CEIL(MaxPrecalc, 4);

        {
            const U32 Pe = RHMINER_CEIL(MaxPrecalc + 1 + MERSENNE_TWISTER_PERIOD, 4);
            const U32 Pb = MERSENNE_TWISTER_PERIOD;
            U32* mt = state->MT;
            
#if defined(RH2_ENABLE_MERSSEN_INTERLEAVE) && !defined(RHMINER_NO_SSE4)

            U32 lval =  state->seed;
            U32 i = 1;
            mt = state->MT;
            mt[0] = lval;
            mt += 2;
            while (i <= A)
            {
                lval =  0x6c078965 * (lval ^ lval >> 30) + i;
                *mt = lval;
                i++;
                mt += 2;
            }
            mt = state->MT + 1;
            while (i < Pb)
            {
                lval = 0x6c078965 * (lval ^ lval >> 30) + i++;
            }
            while (i <= Pe)
            {
                lval = 0x6c078965 * (lval ^ lval >> 30) + i++;
                *mt = lval;
                mt += 2;
            }

            i = 0;
            mt = state->MT;
            RH_PREFETCH_MEM(mt+4);
            while (i < A)
            {
                lval = M32(*mt);
                mt += 2;
                lval |= L31(*mt);
                mt--;
                lval = *mt ^ (lval >> 1) ^ (((int32_t(lval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

                lval ^= lval >> 11; 
                lval ^= lval << 7 & 0x9d2c5680; 
                lval ^= lval << 15 & 0xefc60000; 
                lval ^= lval >> 18; 

                mt--;
                *mt = lval;
                mt+=2;
                ++i;
            }

            state->index = 0;            


#else 
            U32 lval =  state->seed;
            U32 i = 1;
            *mt = lval;
            mt++;
            while (i <= A)
            {
                lval =  0x6c078965 * (lval ^ lval >> 30) + i;
                *mt = lval;
                i++;
                mt++;
            }

            while (i < Pb)
                lval = 0x6c078965 * (lval ^ lval >> 30) + i++;

            U32 fval = lval;
            mt = state->MT;
            U32* end = mt + A;
            while (mt != end)
            {
                lval = M32(*mt);
                mt++;
                lval |= L31(*mt);
                fval = 0x6c078965 * (fval ^ fval >> 30) + i++; 
                lval = fval ^ (lval >> 1) ^ (((int32_t(lval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

                mt--;
                *mt = lval;
                mt++;
            }
            state->index = 0;            
            mt = state->MT;
            {
                __m128i y, r0, r1;
                __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
                __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
                c1 = _mm_shuffle_epi32(c1, 0);
                c2 = _mm_shuffle_epi32(c2, 0);
                while (mt != end)
                {
                    y = RH_MM_LOAD128((__m128i*)mt);
                    r0 = _mm_srli_epi32(y, 11);
                    y = _mm_xor_si128(y, r0);
                    r0 = _mm_slli_epi32(y, 7);
                    r1 = _mm_and_si128(r0, c1);
                    y = _mm_xor_si128(y, r1);
                    r0 = _mm_slli_epi32(y, 15);
                    r1 = _mm_and_si128(r0, c2);
                    y = _mm_xor_si128(y, r1);
                    r0 = _mm_srli_epi32(y, 18);
                    y = _mm_xor_si128(y, r0);
                    RH_MM_STORE128((__m128i*)mt, y);
                    mt += 4;
                }
            }

#endif 
        }

    }
#ifdef _DEBUG
    else
    {
        const U32 A = RHMINER_CEIL(MaxPrecalc, 4) * 2;
        const size_t C = state->index;
        RHMINER_ASSERT(C < A);
    }
#endif

    U32 res = state->MT[state->index];

#if defined(RH2_ENABLE_MERSSEN_INTERLEAVE) && !defined(RHMINER_NO_SSE4)
    state->index += 2;
#else
    state->index++;
#endif
    return res;
}

#define RH_MT_ST(r, fr, idx1, idx2) \
    pval = _mm_extract_epi32_M(r, idx1); \
    pval2 = _mm_extract_epi32_M(r, idx2); \
    pvaln = _mm_extract_epi32_M(fr, idx1); \
    pval = M32(pval) | L31(pval2); \
    pval = pvaln ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); \
    r1 = _mm_insert_epi32_M(r1, pval, idx1); \

#define RH_MT_ST_N(r,rn, fr ) \
    pval = _mm_extract_epi32_M(r, 3); \
    pval2 = _mm_extract_epi32_M(rn, 0); \
    pvaln = _mm_extract_epi32_M(fr, 3); \
    pval = M32(pval) | L31(pval2); \
    pval = pvaln ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); \
    r1 = _mm_insert_epi32_M(r1, pval, 3); \

#if defined(RHMINER_ENABLE_SSE4)

#if 0
inline uint32_t merssen_twister_rand_fast_partial_SSE4(mersenne_twister_state* state,  U32 MaxPrecalc)
{

    if (state->index == MERSENNE_TWISTER_SIZE)
    {
            U32* mt = state->MT;

            RH_ASSERT((MaxPrecalc % 4) == 0);
            U32 pval = state->seed;
            U32 pval1, pval2, pvaln, i;
            __m128i f1,r1,r2;
            __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
            __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
            c1 = _mm_shuffle_epi32(c1, 0);
            c2 = _mm_shuffle_epi32(c2, 0);
            MaxPrecalc += 8;
            for (i = 1; i < MaxPrecalc; )
            {
                r1 = _mm_cvtsi32_si128(pval);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                r1 = _mm_insert_epi32_M(r1, pval, 1);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                r1 = _mm_insert_epi32_M(r1, pval, 2);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                r1 = _mm_insert_epi32_M(r1, pval, 3);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++; 
                
                RH_MM_STORE128((__m128i*)mt, r1);
                mt += 4;
            }
            while (i < MERSENNE_TWISTER_PERIOD)
            {
                pval =  0x6c078965 * (pval ^ pval >> 30) + i++;
            }        
            

#define RH_MT_ST_P(r, fr, idx1, idx2) \
    pval1 = _mm_extract_epi32_M(r, idx1); \
    pval2 = _mm_extract_epi32_M(r, idx2); \
    pvaln = _mm_extract_epi32_M(fr, idx1); \
    pval1 = M32(pval1) | L31(pval2); \
    pval1 = pvaln ^ (pval1 >> 1) ^ (((int32_t(pval1) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); \
    r1 = _mm_insert_epi32_M(r1, pval1, idx1); \

#define RH_MT_ST_P_N(r,rn, fr ) \
    pval1 = _mm_extract_epi32_M(r, 3); \
    pval2 = _mm_extract_epi32_M(rn, 0); \
    pvaln = _mm_extract_epi32_M(fr, 3); \
    pval1 = M32(pval1) | L31(pval2); \
    pval1 = pvaln ^ (pval1 >> 1) ^ (((int32_t(pval1) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); \
    r1 = _mm_insert_epi32_M(r1, pval1, 3); \


            MaxPrecalc += MERSENNE_TWISTER_PERIOD-8;
            mt = state->MT;
            r1 = RH_MM_LOAD128((__m128i*)mt);
            for (; i < MaxPrecalc; )
            {
                
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                f1 = _mm_cvtsi32_si128(pval);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                f1 = _mm_insert_epi32_M(f1, pval, 1);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                f1 = _mm_insert_epi32_M(f1, pval, 2);
                pval = 0x6c078965 * (pval ^ pval >> 30) + i++;
                f1 = _mm_insert_epi32_M(f1, pval, 3);


                RH_MT_ST_P(r1, f1, 0, 1);
                RH_MT_ST_P(r1, f1, 1, 2);
                RH_MT_ST_P(r1, f1, 2, 3);

                r2 = RH_MM_LOAD128((__m128i*)(mt + 4));
                RH_MT_ST_P_N(r1, r2, f1);

                f1 = _mm_srli_epi32(r1, 11);
                r1 = _mm_xor_si128(r1, f1);
                f1 = _mm_slli_epi32(r1, 7);
                f1 = _mm_and_si128(f1, c1);
                r1 = _mm_xor_si128(r1, f1);
                f1 = _mm_slli_epi32(r1, 15);
                f1 = _mm_and_si128(f1, c2);
                r1 = _mm_xor_si128(r1, f1);
                f1 = _mm_srli_epi32(r1, 18);
                r1 = _mm_xor_si128(r1, f1);

                RH_MM_STORE128((__m128i*)mt, r1);
                mt += 4;
                r1 = r2;
            }

            state->index = 0;
    }
#ifdef _DEBUG
    else
    {
        const U32 A = RHMINER_CEIL(MaxPrecalc, 4) * 2;
        const size_t C = state->index;
        RHMINER_ASSERT(C < A);
    }
#endif

    U32 res = state->MT[state->index];
    state->index++;
    return res;
}
#endif //0

#define merssen_twister_rand_fast_partial_12(S) merssen_twister_rand_fast_partial_12_SSE4(S)

inline uint32_t merssen_twister_rand_fast_partial_12_SSE4(mersenne_twister_state* state)
{
    if (state->index == MERSENNE_TWISTER_SIZE)
    {
#ifdef RH2_ENABLE_MERSSEN_12_SSE4

        __m128i f1,r1,r2,r3;
        __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
        __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
        c1 = _mm_shuffle_epi32(c1, 0);
        c2 = _mm_shuffle_epi32(c2, 0);
        U32  pval, pval2, pvaln, ip1_l, f_last;
        pval = state->seed;
        
        r1 =  _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 1; 
        r1 = _mm_insert_epi32_M(r1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 2;
        r1 = _mm_insert_epi32_M(r1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 3;
        r1 = _mm_insert_epi32_M(r1, pval, 3);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 4; 

        r2 = _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 5;
        r2 = _mm_insert_epi32_M(r2, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 6;
        r2 = _mm_insert_epi32_M(r2, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 7;
        r2 = _mm_insert_epi32_M(r2, pval, 3);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 8; 

        r3 = _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval^ pval>> 30) + 9;
        r3 = _mm_insert_epi32_M(r3, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 10;
        r3 = _mm_insert_epi32_M(r3, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 11;
        r3 = _mm_insert_epi32_M(r3, pval, 3);
        ip1_l = 0x6c078965 * (pval ^ pval >> 30) + 12; 

        pval2 = 13;
        pval = ip1_l;
        while (pval2 <= MERSENNE_TWISTER_PERIOD)
        {
            pval =  0x6c078965 * (pval ^ pval >> 30) + pval2++;
        }        

        f1 = _mm_cvtsi32_si128(pval);        
        pval = 0x6c078965 * (pval ^ pval >> 30) + 398;
        f1 = _mm_insert_epi32_M(f1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 399;
        f1 = _mm_insert_epi32_M(f1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 400;
        f1 = _mm_insert_epi32_M(f1, pval, 3);
        f_last = pval;

        RH_MT_ST(r1, f1, 0, 1);
        RH_MT_ST(r1, f1, 1, 2);
        RH_MT_ST(r1, f1, 2, 3);
        RH_MT_ST_N(r1, r2, f1);

       
        {
            f1 = _mm_srli_epi32(r1, 11);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 7);
            f1 = _mm_and_si128(f1, c1);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 15);
            f1 = _mm_and_si128(f1, c2);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_srli_epi32(r1, 18);
            r1 = _mm_xor_si128(r1, f1);
            RH_MM_STORE128((__m128i*)state->MT + 0, r1);
        }

        pval = 0x6c078965 * (f_last ^ f_last >> 30) + 401;
        f1 = _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 402;
        f1 = _mm_insert_epi32_M(f1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 403;
        f1 = _mm_insert_epi32_M(f1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 404;
        f1 = _mm_insert_epi32_M(f1, pval, 3);
        f_last = pval;

        RH_MT_ST(r2, f1, 0, 1);
        RH_MT_ST(r2, f1, 1, 2);
        RH_MT_ST(r2, f1, 2, 3);
        RH_MT_ST_N(r2, r3, f1);

        {
            f1 = _mm_srli_epi32(r1, 11);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 7);
            f1 = _mm_and_si128(f1, c1);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 15);
            f1 = _mm_and_si128(f1, c2);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_srli_epi32(r1, 18);
            r1 = _mm_xor_si128(r1, f1);
            RH_MM_STORE128((__m128i*)&(state->MT[4]), r1);
        }

        pval = 0x6c078965 * (f_last ^ f_last >> 30) + 405; 
        f1 = _mm_insert_epi32_M(f1, pval, 0);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 406;
        f1 = _mm_insert_epi32_M(f1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 407;
        f1 = _mm_insert_epi32_M(f1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 408;
        f1 = _mm_insert_epi32_M(f1, pval, 3);

        RH_MT_ST(r3, f1, 0, 1);
        RH_MT_ST(r3, f1, 1, 2);
        RH_MT_ST(r3, f1, 2, 3);
        pval = _mm_extract_epi32_M(r3, 3);
        pval2 = ip1_l; 
        pvaln = _mm_extract_epi32_M(f1, 3); 
        pval = M32(pval) | L31(pval2); 
        pval = pvaln ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); 
        r1 = _mm_insert_epi32_M(r1, pval, 3); 

        {
            f1 = _mm_srli_epi32(r1, 11);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 7);
            f1 = _mm_and_si128(f1, c1);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_slli_epi32(r1, 15);
            f1 = _mm_and_si128(f1, c2);
            r1 = _mm_xor_si128(r1, f1);
            f1 = _mm_srli_epi32(r1, 18);
            r1 = _mm_xor_si128(r1, f1);
            RH_MM_STORE128((__m128i*)&(state->MT[8]), r1);
        }
#else //RH2_ENABLE_MERSSEN_12_SSE4
            const U32 A = 12;
            const U32 Pe = A + MERSENNE_TWISTER_PERIOD;
            const U32 Pb = MERSENNE_TWISTER_PERIOD;

            U32 lval =  state->seed;
            U32 i = 1;
            U32* mt = state->MT;
            *mt = lval;
            mt++;
            while (i <= A)
            {
                lval =  0x6c078965 * (lval ^ lval >> 30) + i;
                *mt = lval;
                i++;
                mt++;
            }

            while (i < Pb)
                lval = 0x6c078965 * (lval ^ lval >> 30) + i++;

            U32 fval = lval;
            mt = state->MT;
            U32* end = mt + A;
            while (mt != end)
            {
                lval = M32(*mt);
                mt++;
                lval |= L31(*mt);
                fval = 0x6c078965 * (fval ^ fval >> 30) + i++; 
                lval = fval ^ (lval >> 1) ^ (((int32_t(lval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC);

                mt--;
                *mt = lval;
                mt++;
            }
            state->index = 0;            
            mt = state->MT;
            {
                __m128i y, r0, r1;
                __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
                __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
                c1 = _mm_shuffle_epi32(c1, 0);
                c2 = _mm_shuffle_epi32(c2, 0);
                while (mt != end)
                {
                    y = RH_MM_LOAD128((__m128i*)mt);
                    r0 = _mm_srli_epi32(y, 11);
                    y = _mm_xor_si128(y, r0);
                    r0 = _mm_slli_epi32(y, 7);
                    r1 = _mm_and_si128(r0, c1);
                    y = _mm_xor_si128(y, r1);
                    r0 = _mm_slli_epi32(y, 15);
                    r1 = _mm_and_si128(r0, c2);
                    y = _mm_xor_si128(y, r1);
                    r0 = _mm_srli_epi32(y, 18);
                    y = _mm_xor_si128(y, r0);
                    RH_MM_STORE128((__m128i*)mt, y);
                    mt += 4;
                }
            }
#endif 
        state->index = 0;
    }
#ifdef _DEBUG
    else
    {
        const U32 A = 12;
        const U32 C = (U32)state->index;
        RHMINER_ASSERT(C < A);
    }
#endif

    return state->MT[state->index++];
}



#define merssen_twister_rand_fast_partial_4(S)  merssen_twister_rand_fast_partial_4_SSE4(S)

inline uint32_t merssen_twister_rand_fast_partial_4_SSE4(mersenne_twister_state* state)
{
    if (state->index == MERSENNE_TWISTER_SIZE)
    {
        __m128i f1,r1;
         __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
         __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
        c1 = _mm_shuffle_epi32(c1, 0);
        c2 = _mm_shuffle_epi32(c2, 0);
        U32  pval, pval2, pvaln, ip1_l;
        pval = state->seed;
        
        r1 =  _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 1;
        r1 = _mm_insert_epi32_M(r1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 2;
        r1 = _mm_insert_epi32_M(r1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 3;
        r1 = _mm_insert_epi32_M(r1, pval, 3);
        ip1_l = 0x6c078965 * (pval ^ pval >> 30) + 4; 

        pval2 = 5;
        pval = ip1_l;
        while (pval2 <= MERSENNE_TWISTER_PERIOD)
        {
            pval = 0x6c078965 * (pval ^ pval >> 30) + pval2++;
        }

        f1 = _mm_cvtsi32_si128(pval);        
        pval = 0x6c078965 * (pval ^ pval >> 30) + 398;
        f1 = _mm_insert_epi32_M(f1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 399;
        f1 = _mm_insert_epi32_M(f1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 400;
        f1 = _mm_insert_epi32_M(f1, pval, 3);

        RH_MT_ST(r1, f1, 0, 1);
        RH_MT_ST(r1, f1, 1, 2);
        RH_MT_ST(r1, f1, 2, 3);
        pval = _mm_extract_epi32_M(r1, 3);
        pval2 = ip1_l; 
        pvaln = _mm_extract_epi32_M(f1, 3); 
        pval = M32(pval) | L31(pval2); 
        pval = pvaln ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); 
        r1 = _mm_insert_epi32_M(r1, pval, 3); 

        f1 = _mm_srli_epi32(r1, 11); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_slli_epi32(r1, 7);
        f1 = _mm_and_si128(f1, c1); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_slli_epi32(r1, 15); 
        f1 = _mm_and_si128(f1, c2); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_srli_epi32(r1, 18); 
        r1 = _mm_xor_si128(r1, f1); 
        RH_MM_STORE128((__m128i*)&(state->MT[0]), r1); 

        state->index = 0;
    }
    
    return state->MT[state->index++];
}


inline uint32_t merssen_twister_rand_fast_partial_8_SSE4(mersenne_twister_state* state)
{
    if (state->index == MERSENNE_TWISTER_SIZE)
    {
        __m128i f1,f2,r1,r2;
         __m128i c1 = _mm_cvtsi32_si128(0x9d2c5680);
         __m128i c2 = _mm_cvtsi32_si128(0xefc60000);
        c1 = _mm_shuffle_epi32(c1, 0);
        c2 = _mm_shuffle_epi32(c2, 0);
        U32  pval, pval2, pvaln, ip1_l;
        pval = state->seed;
        
        r1 =  _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 1;
        r1 = _mm_insert_epi32_M(r1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 2;
        r1 = _mm_insert_epi32_M(r1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 3;
        r1 = _mm_insert_epi32_M(r1, pval, 3);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 4; 

        r2 = _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 5;
        r2 = _mm_insert_epi32_M(r2, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 6;
        r2 = _mm_insert_epi32_M(r2, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 7;
        r2 = _mm_insert_epi32_M(r2, pval, 3);
        ip1_l = 0x6c078965 * (pval ^ pval >> 30) + 8; 

        pval2 = 9;
        pval = ip1_l;
        while (pval2 <= MERSENNE_TWISTER_PERIOD)
        {
            pval = 0x6c078965 * (pval ^ pval >> 30) + pval2++;
        }

        f1 = _mm_cvtsi32_si128(pval);        
        pval = 0x6c078965 * (pval ^ pval >> 30) + 398;
        f1 = _mm_insert_epi32_M(f1, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 399;
        f1 = _mm_insert_epi32_M(f1, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 400;
        f1 = _mm_insert_epi32_M(f1, pval, 3);

        pval = 0x6c078965 * (pval ^ pval >> 30) + 401;
        f2 = _mm_cvtsi32_si128(pval);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 402;
        f2 = _mm_insert_epi32_M(f2, pval, 1);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 403;
        f2 = _mm_insert_epi32_M(f2, pval, 2);
        pval = 0x6c078965 * (pval ^ pval >> 30) + 404;
        f2 = _mm_insert_epi32_M(f2, pval, 3);


        RH_MT_ST(r1, f1, 0, 1);
        RH_MT_ST(r1, f1, 1, 2);
        RH_MT_ST(r1, f1, 2, 3);
        RH_MT_ST_N(r1, r2, f1);
        f1 = _mm_srli_epi32(r1, 11); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_slli_epi32(r1, 7);
        f1 = _mm_and_si128(f1, c1); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_slli_epi32(r1, 15); 
        f1 = _mm_and_si128(f1, c2); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_srli_epi32(r1, 18); 
        r1 = _mm_xor_si128(r1, f1); 
        RH_MM_STORE128((__m128i*)state->MT+0, r1); 

        RH_MT_ST(r2, f2, 0, 1);
        RH_MT_ST(r2, f2, 1, 2);
        RH_MT_ST(r2, f2, 2, 3);
        pval = _mm_extract_epi32_M(r2, 3);
        pval2 = ip1_l; 
        pvaln = _mm_extract_epi32_M(f2, 3); 
        pval = M32(pval) | L31(pval2); 
        pval = pvaln ^ (pval >> 1) ^ (((int32_t(pval) << 31) >> 31) & MERSENNE_TWISTER_MAGIC); 
        r1 = _mm_insert_epi32_M(r1, pval, 3); 

        f1 = _mm_srli_epi32(r1, 11); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_slli_epi32(r1, 7);
        f1 = _mm_and_si128(f1, c1); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_slli_epi32(r1, 15); 
        f1 = _mm_and_si128(f1, c2); 
        r1 = _mm_xor_si128(r1, f1); 
        f1 = _mm_srli_epi32(r1, 18); 
        r1 = _mm_xor_si128(r1, f1); 
        RH_MM_STORE128((__m128i*)&(state->MT[4]), r1); 
        
        state->index = 0;
    }
#ifdef _DEBUG
    else
    {
        const U32 A = 8;
        const U32 C = (U32)state->index;
        RHMINER_ASSERT(C < A);
    }
#endif
    
    return state->MT[state->index++];
}

#else //#if defined(RHMINER_ENABLE_SSE4)

#define merssen_twister_rand_fast_partial_12(S) merssen_twister_rand_fast_partial(S, 12)
#define merssen_twister_rand_fast_partial_4(S)  merssen_twister_rand_fast_partial(S, 8)
#define merssen_twister_rand_fast_partial_204(S) merssen_twister_rand_fast_partial(S, 204)
#define merssen_twister_rand_fast_partial merssen_twister_rand_fast_partial_slow


#endif //#if defined(RHMINER_ENABLE_SSE4)

inline void merssen_twister_seed_fast_partial(uint32_t value, mersenne_twister_state* state, const int MaxPrecalc)
{
    RH_ASSERT(MaxPrecalc != MERSENNE_TWISTER_SIZE)
    state->index = MERSENNE_TWISTER_SIZE;
    state->seed = value;
}



#endif //#define RANDOM_HASH_mersenne_twister_H