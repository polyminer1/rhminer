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
#include "MinersLib/Algo/sph_panama.h"

void RandomHash_Panama(RH_StridePtr roundInput, RH_StridePtr output)
{
    U32 msgLen = RH_STRIDE_GET_SIZE(roundInput);
    U64* message = RH_STRIDE_GET_DATA64(roundInput);

    sph_panama_context ctx;
    sph_panama_init(&ctx);
    sph_panama(&ctx, message, msgLen);

    //get the hash result
    sph_panama_close(&ctx, RH_STRIDE_GET_DATA(output));
    RH_STRIDE_SET_SIZE(output, 8*4);
}
