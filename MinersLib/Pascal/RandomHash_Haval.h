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
#include "MinersLib/Algo/sph-haval.h"

void RandomHash_Haval4(RH_StridePtr roundInput, RH_StridePtr output, U32 bitSize)
{
    U32 msgLen = RH_STRIDE_GET_SIZE(roundInput);
    U32* message = RH_STRIDE_GET_DATA(roundInput);

    sph_haval_context cc;
    haval_init(&cc, bitSize >> 5, 4);
    haval4(&cc, message, msgLen);


    RH_STRIDE_SET_SIZE(output, bitSize >> 3);
    uint32_t* buf = RH_STRIDE_GET_DATA(output);
    haval4_close(&cc, 0, 0, buf);
}

void RandomHash_Haval3(RH_StridePtr roundInput, RH_StridePtr output, U32 bitSize)
{
    U32 msgLen = RH_STRIDE_GET_SIZE(roundInput);
    U32* message = RH_STRIDE_GET_DATA(roundInput);

    sph_haval_context cc;
    haval_init(&cc, bitSize >> 5, 3);
    haval3(&cc, message, msgLen);


    RH_STRIDE_SET_SIZE(output, bitSize >> 3);
    uint32_t* buf = RH_STRIDE_GET_DATA(output);
    haval3_close(&cc, 0, 0, buf);

}

