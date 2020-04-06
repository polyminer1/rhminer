#include "precomp.h"

#ifndef _int128_c_
#define _int128_c_

#include "int128_o.h"

#include <stdlib.h>
#include <stdio.h>

// Assignment and Assignment-Conversion operators

s128_o::s128_o(int x)
{
    this->low = x;

    if (x < 0) {
        this->high = -1;
    }
    else {
        this->high = 0;
    }
}

s128_o::s128_o(__int64 x)
{
    this->low = x;

    if (x < 0) {
        this->high = -1;
    }
    else {
        this->high = 0;
    }
}

s128_o::s128_o(double x)
{
    unsigned __int64 t, m, h, l;

    if (x < -1.7014118346046e38) {
        // overflow negative
        this->high = HIGHBIT;
        this->low = 0;
    }
    else if (x < -9.2233720368547e18) {
        // take the 54 mantissa bits and shift into position
        t = *((unsigned __int64 *)&x);
        m = (t & BITS51_0) | BIT_52;
        t = (t & BITS62_0) >> 52;
        // if x is 1.5 * 2^1, t will be 1024
        // if x is 1.5 * 2^52, t will be 1024+51 = 1075
        t = t - 1075;
        if (t > 64) {
            l = 0;
            h = m << (t - 64);
        }
        else {
            l = m << t;
            h = m >> (64 - t);
        }
        this->low = ~l;
        this->high = ~h;
    }
    else if (x < 9.2233720368547e18) {
        // it will fit in a unsigned __int64
        this->low = ((unsigned __int64)x);
        this->high = ((x<0) ? -1 : 0);
    }
    else if (x < 1.7014118346046e38) {
        // take the 54 mantissa bits and shift into position
        t = *((unsigned __int64 *)&x);
        m = (t & BITS51_0) | BIT_52;
        t = (t & BITS62_0) >> 52;
        // if x is 1.5 * 2^1, t will be 1024
        // if x is 1.5 * 2^52, t will be 1024+51 = 1075
        t = t - 1075;
        if (t > 64) {
            this->low = 0;
            this->high = m << (t - 64);
        }
        else {
            this->low = m << t;
            this->high = m >> (64 - t);
        }
    }
    else {
        // overflow positive
        this->high = BITS62_0;
        this->low = ALLBITS;
    }
}

// Infix addition

s128_o operator + (const s128_o & lhs, const s128_o & rhs)
{
    s128_o res;

    res.high = lhs.high + rhs.high;
    res.low = lhs.low + rhs.low;
    if (res.low < rhs.low) {
        (res.high)++;
    }

    return res;
}

s128_o operator + (const s128_o & lhs, __int64 rhs)
{
    s128_o res;

    if (rhs < 0) {
        res.high = lhs.high - 1;
        res.low = lhs.low + rhs;
        if (res.low < (unsigned __int64)rhs) {
            (res.high)++;
        }
    }
    else {
        res.high = lhs.high;
        res.low = lhs.low + rhs;
        if (res.low < (unsigned __int64)rhs) {
            (res.high)++;
        }
    }

    return res;
}

s128_o operator + (const s128_o & lhs, unsigned __int64 rhs)
{
    s128_o res;

    res.high = lhs.high;
    res.low = lhs.low + rhs;
    if (res.low < rhs) {
        (res.high)++;
    }

    return res;
}

s128_o operator + (const s128_o & lhs, int rhs)
{
    s128_o res;

    if (rhs < 0) {
        res.high = lhs.high - 1;
        res.low = lhs.low + rhs;
        if (res.low < rhs) {
            (res.high)++;
        }
    }
    else {
        res.high = lhs.high;
        res.low = lhs.low + rhs;
        if (res.low < rhs) {
            (res.high)++;
        }
    }

    return res;
}

s128_o operator + (__int64 lhs, const s128_o & rhs)
{
    s128_o res;
    s128_o a;

    a.low = lhs;
    a.high = (lhs < 0) ? -1 : 0;

    res.high = a.high + rhs.high;
    res.low = a.low + rhs.low;
    if (res.low < rhs.low) {
        (res.high)++;
    }

    return res;
}

// Infix subtraction

s128_o operator - (const s128_o & lhs, const s128_o & rhs)
{
    s128_o res;

    res.high = lhs.high - rhs.high;
    res.low = lhs.low - rhs.low;
    if (res.low > lhs.low) {
        // borrow
        (res.high)--;
    }

    return res;
}

s128_o operator - (const s128_o & lhs, int rhs)
{
    s128_o res;
    s128_o b;

    b.low = rhs;
    b.high = (rhs < 0) ? -1 : 0;

    res.high = lhs.high - b.high;
    res.low = lhs.low - b.low;
    if (res.low > lhs.low) {
        (res.high)--;
    }

    return res;
}

// Unary minus

s128_o operator - (const s128_o & x)
{
    s128_o res;

    res.high = ~(x.high);
    res.low = ~(x.low);
    res.low += 1;
    if (res.low == 0) {
        res.high += 1;
    }

    return res;
}

// Support routines for multiply, divide and modulo

s128 s128_shr(s128_o x)
{
    s128_o rv;

    //  printf("%.16llX %016llX  >> ", x.high, x.low);
    rv.low = (x.low >> 1) | (x.high << 63);
    rv.high = x.high >> 1;

    //  printf("%.16llX %016llX\n", rv.high, rv.low);

    return rv;
}

s128 s128_shl(s128_o x)
{
    s128_o rv;

    rv.high = (x.high << 1) | (x.low >> 63);
    rv.low = x.low << 1;

    return rv;
}

// Multiplication

// old: 28.7  new: 6.3

/* Enable both of these defines to get error-checking */
#define MULT1_OLD 0
#define MULT1_NEW 1

static int m1err = 0;

s128_o mult1(s128_o xi, s128_o yi)
{
#if MULT1_OLD
    s128_o x, y, rv1;
    int s;
#endif
#if MULT1_NEW
    s128_o rv2;
    unsigned __int64 acc, ac2, carry, o1, o2;
    unsigned __int64 a, b, c, d, e, f, g, h;
#endif

#if MULT1_OLD
    s = 1;
    x = xi; y = yi;
    if (x < ((s128_o)0)) {
        s = -s;
        x = -x;
    }
    if (y < ((s128_o)0)) {
        s = -s;
        y = -y;
    }

    rv1 = 0;
    while (y != ((s128_o)0)) {
        if (y.low & 1) {
            rv1 = rv1 + x;
        }
        x = s128_shl(x);
        y = s128_shr(y);
    }

    if (s < 0) {
        rv1 = -rv1;
    }
#endif

#if MULT1_NEW

    /*            x      a  b  c  d
    y      e  f  g  h
    ---------------
    ah bh ch dh
    bg cg dg
    cf df
    de
    --------------------------
    -o2-- -o1--
    */

    d = xi.low & LO_WORD;
    c = (xi.low & HI_WORD) >> 32LL;
    b = xi.high & LO_WORD;
    a = (xi.high & HI_WORD) >> 32LL;

    h = yi.low & LO_WORD;
    g = (yi.low & HI_WORD) >> 32LL;
    f = yi.high & LO_WORD;
    e = (yi.high & HI_WORD) >> 32LL;

    acc = d * h;
    o1 = acc & LO_WORD;
    acc >>= 32LL;
    carry = 0;
    ac2 = acc + c * h; if (ac2 < acc) { carry++; }
    acc = ac2 + d * g; if (acc < ac2) { carry++; }
    rv2.low = o1 | (acc << 32LL);
    ac2 = (acc >> 32LL) | (carry << 32LL); carry = 0;

    acc = ac2 + b * h; if (acc < ac2) { carry++; }
    ac2 = acc + c * g; if (ac2 < acc) { carry++; }
    acc = ac2 + d * f; if (acc < ac2) { carry++; }
    o2 = acc & LO_WORD;
    ac2 = (acc >> 32LL) | (carry << 32LL);

    acc = ac2 + a * h;
    ac2 = acc + b * g;
    acc = ac2 + c * f;
    ac2 = acc + d * e;
    rv2.high = (ac2 << 32LL) | o2;

#if MULT1_OLD
    if ((rv1.high != rv2.high) || (rv1.low != rv2.low)) {
        printf("m1 err1 %16llX %016llX * %16llX %016llX\n",
            xi.high, xi.low, yi.high, yi.low);
        printf("abcd   %8llX %08llX %08llX %08llX\n", a, b, c, d);
        printf("efgh   %8llX %08llX %08llX %08llX\n", e, f, g, h);
        printf("    -> %16llX %016llX\n", rv2.high, rv2.low);
        m1err++;
        if (m1err > 10) {
            POLY_EXIT_APP(-1);
        }
    }
#endif

#endif

#if MULT1_NEW
    return rv2;
#else
    return rv1;
#endif
}

s128_o operator * (const s128_o & lhs, const s128_o & rhs)
{
    s128_o rv;

    rv = mult1(lhs, rhs);

    return rv;
}

s128_o operator * (const s128_o & lhs, unsigned __int64 rhs)
{
    s128_o t;
    s128_o rv;

    t = rhs;
    rv = mult1(lhs, rhs);

    return rv;
}

s128_o operator * (const s128_o & lhs, int rhs)
{
    s128_o t;
    s128_o rv;

    t = rhs;
    rv = mult1(lhs, rhs);

    return rv;
}

// Division

s128_o div1(s128_o x, s128_o d, s128_o *r)
{
    int s;
    s128_o d1, p2, rv;

    //printf("divide %.16llX %016llX / %.16llX %016llX\n", x.high, x.low, d.high, d.low);

    /* check for divide by zero */
    if ((d.low == 0) && (d.high == 0)) {
        rv.low = x.low / d.low; /* This will cause runtime error */
    }

    s = 1;
    if (x < ((s128_o)0)) {
        // notice that MININT will be unchanged, this is used below.
        s = -s;
        x = -x;
    }
    if (d < ((s128_o)0)) {
        s = -s;
        d = -d;
    }

    if (d == ((s128_o)1)) {
        /* This includes the overflow case MININT/-1 */
        rv = x;
        x = 0;
    }
    else if (x < ((s128_o)d)) {
        /* x < d, so quotient is 0 and x is remainder */
        rv = 0;
    }
    else {
        rv = 0;

        /* calculate biggest power of 2 times d that's <= x */
        p2 = 1; d1 = d;
        x = x - d1;
        while (x >= d1) {
            x = x - d1;
            d1 = d1 + d1;
            p2 = p2 + p2;
        }
        x = x + d1;

        while (p2 != ((s128_o)0)) {
            //printf("x %.16llX %016llX d1 %.16llX %016llX\n", x.high, x.low, d1.high, d1.low);
            if (x >= d1) {
                x = x - d1;
                rv = rv + p2;
                //printf("`.. %.16llX %016llX\n", rv.high, rv.low);
            }
            p2 = s128_shr(p2);
            d1 = s128_shr(d1);
        }

        /* whatever is left in x is the remainder */
    }

    /* Put sign in result */
    if (s < 0) {
        rv = -rv;
    }

    /* return remainder if they asked for it */
    if (r) {
        *r = x;
    }

    return rv;
}

s128_o operator / (const s128_o & x, const s128_o & d)
{
    s128_o rv;

    rv = div1(x, d, 0);

    return rv;
}

s128_o operator / (const s128_o & x, unsigned __int64 d)
{
    s128_o t;
    s128_o rv;

    t.high = 0; t.low = d;
    rv = div1(x, t, 0);

    return rv;
}

// Comparison operators

int operator < (const s128_o & lhs, const s128_o & rhs)
{
    if (lhs.high < rhs.high)
        return 1;
    if (rhs.high < lhs.high)
        return 0;
    // high components are equal
    if (lhs.low < rhs.low)
        return 1;
    return 0;
}

int operator < (const s128_o & lhs, int rhs)
{
    s128_o r;

    r = rhs;

    if (lhs.high < r.high)
        return 1;
    if (r.high < lhs.high)
        return 0;
    // high components are equal
    if (lhs.low < r.low)
        return 1;
    return 0;
}

int operator <= (const s128_o & lhs, const s128_o & rhs)
{
    if ((lhs.high == rhs.high) && (lhs.low == rhs.low))
        return 1;

    if (lhs.high < rhs.high)
        return 1;
    if (rhs.high < lhs.high)
        return 0;
    // high components are equal
    if (lhs.low < rhs.low)
        return 1;
    return 0;
}

int operator <= (const s128_o & lhs, int rhs)
{
    s128_o t;

    t = rhs;

    if ((lhs.high == t.high) && (lhs.low == t.low))
        return 1;

    if (lhs.high < t.high)
        return 1;
    if (t.high < lhs.high)
        return 0;
    // high components are equal
    if (lhs.low < t.low)
        return 1;
    return 0;
}

int operator == (const s128_o & lhs, const s128_o & rhs)
{
    if (lhs.high != rhs.high)
        return 0;
    if (lhs.low != rhs.low)
        return 0;
    return 1;
}

int operator == (const s128_o & lhs, int rhs)
{
    s128_o t;

    t = rhs;

    return ((lhs.high == t.high) && (lhs.low == t.low));
}

int operator != (const s128_o & lhs, const s128_o & rhs)
{
    if (lhs.high != rhs.high)
        return 1;
    if (lhs.low != rhs.low)
        return 1;
    return 0;
}

int operator != (const s128_o & lhs, int rhs)
{
    s128_o t;

    t = rhs;

    return ((lhs.high != t.high) || (lhs.low != t.low));
}

int operator > (const s128_o & lhs, const s128_o & rhs)
{
    if (lhs.high > rhs.high)
        return 1;
    if (rhs.high > lhs.high)
        return 0;
    // high components are equal
    if (lhs.low > rhs.low)
        return 1;
    return 0;
}

int operator > (const s128_o & lhs, int rhs)
{
    s128_o t;

    t = rhs;

    if (lhs.high > t.high)
        return 1;
    if (t.high > lhs.high)
        return 0;
    // high components are equal
    if (lhs.low > t.low)
        return 1;
    return 0;
}

int operator >= (const s128_o & lhs, const s128_o & rhs)
{
    if ((lhs.high == rhs.high) && (lhs.low == rhs.low))
        return 1;

    if (lhs.high > rhs.high)
        return 1;
    if (rhs.high > lhs.high)
        return 0;
    // high components are equal
    if (lhs.low > rhs.low)
        return 1;
    return 0;
}

// Conversion for output

unsigned __int64 s128_u64(s128 x)
{
    return(x.low);
}

unsigned s128_u32(s128 x)
{
    unsigned rv;

    rv = (unsigned)x.low;
    return(rv);
}

void s128_str(s128 x, char *s)
{
    int d, nd, going;
    s128 t, p10;

    if (x < ((s128_o)0)) {
        *s = '-'; s++;
        x = -x;
    }
    if (x == ((s128_o)0)) {
        *s = '0'; s++;
        *s = 0; return;
    }

    // Count number of digits in x
    p10 = 1; nd = 0; going = 1;
    while ((p10 <= x) && (going)) {
        p10 = p10 + p10; if (p10 < ((s128_o)0)) { going = 0; }
        t = p10;
        p10 = p10 + p10; if (p10 < ((s128_o)0)) { going = 0; }
        p10 = p10 + p10; if (p10 < ((s128_o)0)) { going = 0; }
        p10 = p10 + t; if (p10 < ((s128_o)0)) { going = 0; }
        nd++;
    }

    // Extract each digit
    while (nd > 0) {
        int i;

        nd--;
        p10 = 1;
        for (i = 0; i<nd; i++) {
            p10 = p10 + p10;
            t = p10;
            p10 = p10 + p10;
            p10 = p10 + p10;
            p10 = p10 + t;
        }
        d = 0;
        while (x >= p10) {
            x = x - p10;
            d++;
        }
        *s = ('0' + d); s++;
    }
    *s = 0;
}

#endif

/* end of int128.c */