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


#define RH_RADIOGATUN32_BELT_COPY(dstA, dstI, srcA, srcI) { \
        uint32_t* dst = dstA + (3 * dstI);     \
        uint32_t* src = srcA + (3 * srcI);     \
        *dst++ = *src++;                       \
        *dst++ = *src++;                       \
        *dst++ = *src++;}                      \

#define RADIOGATUN32_BLOCK_SIZE 12


inline void RadiogatunRoundFunction(uint32_t* a, uint32_t* mill, uint32_t* belt)
{
    uint32_t q[3];
    RH_RADIOGATUN32_BELT_COPY(q, 0, belt, 12);
		
    uint32_t i = 12;
	while (i > 0)
	{
        RH_RADIOGATUN32_BELT_COPY(belt, i, belt, (i - 1));
		i--;
	}

    RH_RADIOGATUN32_BELT_COPY(belt, 0, q, 0);

    i = 0;
	while (i < 12)
	{
        uint32_t* dst = belt + (3 * (i + 1));
        dst[i % 3] ^= mill[i + 1];
		i++;
	}

	i = 0;
	while (i < 19)
	{
		a[i] = mill[i] ^ (mill[(i + 1) % 19] | ~mill[(i + 2) % 19]);
		i++;
	}

    i = 0;
	while (i < 19)
	{
		mill[i] = ROTR32(a[(7 * i) % 19], (i * (i + 1)) >> 1);
		i++;
	}

	i = 0;
	while (i < 19)
	{
		a[i] = mill[i] ^ mill[(i + 1) % 19] ^ mill[(i + 4) % 19];
		i++;
	}

	a[0] = a[0] ^ 1;
	i = 0;
	while (i < 19)
	{
		mill[i] = a[i];
		i++;
	}

	i = 0;
	while (i < 3)
	{
		mill[i + 13] = mill[i + 13] ^ q[i];
		i++;
	}
}


void RandomHash_RadioGatun32(RH_StridePtr roundInput, RH_StridePtr output)
{
    RH_ALIGN(64) uint32_t mill[/*19*/20];
    RH_ALIGN(64) uint32_t a[19];
    RH_ALIGN(64) uint32_t belt[/*13 * 3*/40];
    int32_t  len = (int32_t)RH_STRIDE_GET_SIZE(roundInput);
    uint32_t *inData = RH_STRIDE_GET_DATA(roundInput);
    uint32_t blockCount = len / RADIOGATUN32_BLOCK_SIZE;
    RH_memzero_of16(mill, sizeof(mill));
    RH_memzero_of16(belt, sizeof(belt));

    uint32_t pre = len % RADIOGATUN32_BLOCK_SIZE;

    memset(((uint8_t*)inData) + len, 0, RADIOGATUN32_BLOCK_SIZE - pre);
    ((uint8_t*)inData)[len] = 0x01;
    blockCount++;

    while (blockCount > 0)
    {
        RH_ALIGN(64) uint32_t  data[RADIOGATUN32_BLOCK_SIZE];
        memcpy(data, inData, RADIOGATUN32_BLOCK_SIZE);
        uint32_t i = 0;
        while (i < 3)
        {
            mill[i + 16] = mill[i + 16] ^ data[i];
            belt[0 + i] = belt[0 + i] ^ data[i];
            i++;
        }
        RadiogatunRoundFunction(a, mill, belt);

        len -= RADIOGATUN32_BLOCK_SIZE;
        inData += RADIOGATUN32_BLOCK_SIZE / 4;
        blockCount--;
    }

    for (uint32_t i = 0; i < 16; i++)
        RadiogatunRoundFunction(a, mill, belt);

    uint32_t* result = RH_STRIDE_GET_DATA(output);
    RH_STRIDE_SET_SIZE(output, 8 * sizeof(uint32_t));

    for (uint32_t i = 0; i < 4; i++)
    {
        RadiogatunRoundFunction(a, mill, belt);
        *result = mill[1];
        result++;
        *result = mill[2];
        result++;
    }

}
