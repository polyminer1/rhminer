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

#define SHA2_512_BLOCK_SIZE 128

void SHA2_512_RoundFunction(uint64_t* data, uint64_t* state)
{
    uint64_t T0, T1, a, b, c, d, e, f, g, h;

    uint64_t beData[128];
    for (a = 0; a < 128 / 8; a++)
        beData[a] = ReverseBytesUInt64(data[a]); 
    T0 = beData[16 - 15];
    T1 = beData[16 - 2];
    beData[16] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[16 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[0];
    T0 = beData[17 - 15];
    T1 = beData[17 - 2];
    beData[17] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[17 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[17 - 16];
    T0 = beData[18 - 15];
    T1 = beData[18 - 2];
    beData[18] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[18 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[18 - 16];
    T0 = beData[19 - 15];
    T1 = beData[19 - 2];
    beData[19] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[19 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[19 - 16];
    T0 = beData[20 - 15];
    T1 = beData[20 - 2];
    beData[20] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[20 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[20 - 16];
    T0 = beData[21 - 15];
    T1 = beData[21 - 2];
    beData[21] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[21 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[21 - 16];
    T0 = beData[22 - 15];
    T1 = beData[22 - 2];
    beData[22] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[22 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[22 - 16];
    T0 = beData[23 - 15];
    T1 = beData[23 - 2];
    beData[23] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[23 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[23 - 16];
    T0 = beData[24 - 15];
    T1 = beData[24 - 2];
    beData[24] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[24 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[24 - 16];
    T0 = beData[25 - 15];
    T1 = beData[25 - 2];
    beData[25] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[25 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[25 - 16];
    T0 = beData[26 - 15];
    T1 = beData[26 - 2];
    beData[26] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[26 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[26 - 16];
    T0 = beData[27 - 15];
    T1 = beData[27 - 2];
    beData[27] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[27 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[27 - 16];
    T0 = beData[28 - 15];
    T1 = beData[28 - 2];
    beData[28] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[28 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[28 - 16];
    T0 = beData[29 - 15];
    T1 = beData[29 - 2];
    beData[29] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[29 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[29 - 16];
    T0 = beData[30 - 15];
    T1 = beData[30 - 2];
    beData[30] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[30 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[30 - 16];
    T0 = beData[31 - 15];
    T1 = beData[31 - 2];
    beData[31] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[31 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[31 - 16];
    T0 = beData[32 - 15];
    T1 = beData[32 - 2];
    beData[32] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[32 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[32 - 16];
    T0 = beData[33 - 15];
    T1 = beData[33 - 2];
    beData[33] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[33 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[33 - 16];
    T0 = beData[34 - 15];
    T1 = beData[34 - 2];
    beData[34] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[34 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[34 - 16];
    T0 = beData[35 - 15];
    T1 = beData[35 - 2];
    beData[35] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[35 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[35 - 16];
    T0 = beData[36 - 15];
    T1 = beData[36 - 2];
    beData[36] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[36 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[36 - 16];
    T0 = beData[37 - 15];
    T1 = beData[37 - 2];
    beData[37] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[37 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[37 - 16];
    T0 = beData[38 - 15];
    T1 = beData[38 - 2];
    beData[38] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[38 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[38 - 16];
    T0 = beData[39 - 15];
    T1 = beData[39 - 2];
    beData[39] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[39 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[39 - 16];
    T0 = beData[40 - 15];
    T1 = beData[40 - 2];
    beData[40] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[40 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[40 - 16];
    T0 = beData[41 - 15];
    T1 = beData[41 - 2];
    beData[41] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[41 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[41 - 16];
    T0 = beData[42 - 15];
    T1 = beData[42 - 2];
    beData[42] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[42 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[42 - 16];
    T0 = beData[43 - 15];
    T1 = beData[43 - 2];
    beData[43] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[43 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[43 - 16];
    T0 = beData[44 - 15];
    T1 = beData[44 - 2];
    beData[44] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[44 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[44 - 16];
    T0 = beData[45 - 15];
    T1 = beData[45 - 2];
    beData[45] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[45 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[45 - 16];
    T0 = beData[46 - 15];
    T1 = beData[46 - 2];
    beData[46] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[46 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[46 - 16];
    T0 = beData[47 - 15];
    T1 = beData[47 - 2];
    beData[47] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[47 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[47 - 16];
    T0 = beData[48 - 15];
    T1 = beData[48 - 2];
    beData[48] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[48 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[48 - 16];
    T0 = beData[49 - 15];
    T1 = beData[49 - 2];
    beData[49] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[49 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[49 - 16];
    T0 = beData[50 - 15];
    T1 = beData[50 - 2];
    beData[50] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[50 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[50 - 16];
    T0 = beData[51 - 15];
    T1 = beData[51 - 2];
    beData[51] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[51 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[51 - 16];
    T0 = beData[52 - 15];
    T1 = beData[52 - 2];
    beData[52] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[52 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[52 - 16];
    T0 = beData[53 - 15];
    T1 = beData[53 - 2];
    beData[53] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[53 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[53 - 16];
    T0 = beData[54 - 15];
    T1 = beData[54 - 2];
    beData[54] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[54 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[54 - 16];
    T0 = beData[55 - 15];
    T1 = beData[55 - 2];
    beData[55] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[55 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[55 - 16];
    T0 = beData[56 - 15];
    T1 = beData[56 - 2];
    beData[56] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[56 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[56 - 16];
    T0 = beData[57 - 15];
    T1 = beData[57 - 2];
    beData[57] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[57 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[57 - 16];
    T0 = beData[58 - 15];
    T1 = beData[58 - 2];
    beData[58] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[58 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[58 - 16];
    T0 = beData[59 - 15];
    T1 = beData[59 - 2];
    beData[59] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[59 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[59 - 16];
    T0 = beData[60 - 15];
    T1 = beData[60 - 2];
    beData[60] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[60 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[60 - 16];
    T0 = beData[61 - 15];
    T1 = beData[61 - 2];
    beData[61] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[61 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[61 - 16];
    T0 = beData[62 - 15];
    T1 = beData[62 - 2];
    beData[62] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[62 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[62 - 16];
    T0 = beData[63 - 15];
    T1 = beData[63 - 2];
    beData[63] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[63 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[63 - 16];
    T0 = beData[64 - 15];
    T1 = beData[64 - 2];
    beData[64] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[64 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[64 - 16];
    T0 = beData[65 - 15];
    T1 = beData[65 - 2];
    beData[65] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[65 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[65 - 16];
    T0 = beData[66 - 15];
    T1 = beData[66 - 2];
    beData[66] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[66 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[66 - 16];
    T0 = beData[67 - 15];
    T1 = beData[67 - 2];
    beData[67] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[67 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[67 - 16];
    T0 = beData[68 - 15];
    T1 = beData[68 - 2];
    beData[68] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[68 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[68 - 16];
    T0 = beData[69 - 15];
    T1 = beData[69 - 2];
    beData[69] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[69 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[69 - 16];
    T0 = beData[70 - 15];
    T1 = beData[70 - 2];
    beData[70] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[70 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[70 - 16];
    T0 = beData[71 - 15];
    T1 = beData[71 - 2];
    beData[71] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[71 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[71 - 16];
    T0 = beData[72 - 15];
    T1 = beData[72 - 2];
    beData[72] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[72 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[72 - 16];
    T0 = beData[73 - 15];
    T1 = beData[73 - 2];
    beData[73] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[73 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[73 - 16];
    T0 = beData[74 - 15];
    T1 = beData[74 - 2];
    beData[74] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[74 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[74 - 16];
    T0 = beData[75 - 15];
    T1 = beData[75 - 2];
    beData[75] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[75 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[75 - 16];
    T0 = beData[76 - 15];
    T1 = beData[76 - 2];
    beData[76] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[76 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[76 - 16];
    T0 = beData[77 - 15];
    T1 = beData[77 - 2];
    beData[77] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[77 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[77 - 16];
    T0 = beData[78 - 15];
    T1 = beData[78 - 2];
    beData[78] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[78 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[78 - 16];
    T0 = beData[79 - 15];
    T1 = beData[79 - 2];
    beData[79] = ((ROTL64(T1, 45)) ^ (ROTL64(T1, 3)) ^ (T1 >> 6)) + beData[79 - 7] + ((ROTL64(T0, 63)) ^ (ROTL64(T0, 56)) ^ (T0 >> 7)) + beData[79 - 16];

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    h = h + (0x428A2F98D728AE22 + beData[0] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0x7137449123EF65CD + beData[1] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0xB5C0FBCFEC4D3B2F + beData[2] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0xE9B5DBA58189DBBC + beData[3] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x3956C25BF348B538 + beData[4] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x59F111F1B605D019 + beData[5] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x923F82A4AF194F9B + beData[6] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0xAB1C5ED5DA6D8118 + beData[7] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));


    h = h + (0xD807AA98A3030242 + beData[8] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0x12835B0145706FBE + beData[9] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0x243185BE4EE4B28C + beData[10] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0x550C7DC3D5FFB4E2 + beData[11] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x72BE5D74F27B896F + beData[12] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x80DEB1FE3B1696B1 + beData[13] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x9BDC06A725C71235 + beData[14] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0xC19BF174CF692694 + beData[15] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));


    h = h + (0xE49B69C19EF14AD2 + beData[16] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0xEFBE4786384F25E3 + beData[17] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0x0FC19DC68B8CD5B5 + beData[18] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0x240CA1CC77AC9C65 + beData[19] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x2DE92C6F592B0275 + beData[20] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x4A7484AA6EA6E483 + beData[21] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x5CB0A9DCBD41FBD4 + beData[22] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x76F988DA831153B5 + beData[23] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0x983E5152EE66DFAB + beData[24] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0xA831C66D2DB43210 + beData[25] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0xB00327C898FB213F + beData[26] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0xBF597FC7BEEF0EE4 + beData[27] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0xC6E00BF33DA88FC2 + beData[28] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0xD5A79147930AA725 + beData[29] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x06CA6351E003826F + beData[30] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x142929670A0E6E70 + beData[31] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0x27B70A8546D22FFC + beData[32] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0x2E1B21385C26C926 + beData[33] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0x4D2C6DFC5AC42AED + beData[34] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0x53380D139D95B3DF + beData[35] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x650A73548BAF63DE + beData[36] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x766A0ABB3C77B2A8 + beData[37] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x81C2C92E47EDAEE6 + beData[38] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x92722C851482353B + beData[39] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0xA2BFE8A14CF10364 + beData[40] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0xA81A664BBC423001 + beData[41] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0xC24B8B70D0F89791 + beData[42] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0xC76C51A30654BE30 + beData[43] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0xD192E819D6EF5218 + beData[44] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0xD69906245565A910 + beData[45] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0xF40E35855771202A + beData[46] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x106AA07032BBD1B8 + beData[47] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0x19A4C116B8D2D0C8 + beData[48] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0x1E376C085141AB53 + beData[49] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0x2748774CDF8EEB99 + beData[50] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0x34B0BCB5E19B48A8 + beData[51] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x391C0CB3C5C95A63 + beData[52] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x4ED8AA4AE3418ACB + beData[53] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x5B9CCA4F7763E373 + beData[54] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x682E6FF3D6B2B8A3 + beData[55] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0x748F82EE5DEFB2FC + beData[56] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0x78A5636F43172F60 + beData[57] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0x84C87814A1F0AB72 + beData[58] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0x8CC702081A6439EC + beData[59] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x90BEFFFA23631E28 + beData[60] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0xA4506CEBDE82BDE9 + beData[61] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0xBEF9A3F7B2C67915 + beData[62] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0xC67178F2E372532B + beData[63] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0xCA273ECEEA26619C + beData[64] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0xD186B8C721C0C207 + beData[65] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0xEADA7DD6CDE0EB1E + beData[66] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0xF57D4F7FEE6ED178 + beData[67] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x06F067AA72176FBA + beData[68] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x0A637DC5A2C898A6 + beData[69] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x113F9804BEF90DAE + beData[70] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x1B710B35131C471B + beData[71] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    h = h + (0x28DB77F523047D84 + beData[72] + ((ROTL64(e, 50)) ^ (ROTL64(e, 46)) ^ (ROTL64(e, 23))) + ((e & f) ^ (~e & g)));

    d = d + h;
    h = h + (((ROTL64(a, 36)) ^ (ROTL64(a, 30)) ^ (ROTL64(a, 25))) + ((a & b) ^ (a & c) ^ (b & c)));

    g = g + (0x32CAAB7B40C72493 + beData[73] + ((ROTL64(d, 50)) ^ (ROTL64(d, 46)) ^ (ROTL64(d, 23))) + ((d & e) ^ (~d & f)));

    c = c + g;
    g = g + (((ROTL64(h, 36)) ^ (ROTL64(h, 30)) ^ (ROTL64(h, 25))) + ((h & a) ^ (h & b) ^ (a & b)));

    f = f + (0x3C9EBE0A15C9BEBC + beData[74] + ((ROTL64(c, 50)) ^ (ROTL64(c, 46)) ^ (ROTL64(c, 23))) + ((c & d) ^ (~c & e)));

    b = b + f;
    f = f + (((ROTL64(g, 36)) ^ (ROTL64(g, 30)) ^ (ROTL64(g, 25))) + ((g & h) ^ (g & a) ^ (h & a)));

    e = e + (0x431D67C49C100D4C + beData[75] + ((ROTL64(b, 50)) ^ (ROTL64(b, 46)) ^ (ROTL64(b, 23))) + ((b & c) ^ (~b & d)));

    a = a + e;
    e = e + (((ROTL64(f, 36)) ^ (ROTL64(f, 30)) ^ (ROTL64(f, 25))) + ((f & g) ^ (f & h) ^ (g & h)));

    d = d + (0x4CC5D4BECB3E42B6 + beData[76] + ((ROTL64(a, 50)) ^ (ROTL64(a, 46)) ^ (ROTL64(a, 23))) + ((a & b) ^ (~a & c)));

    h = h + d;
    d = d + (((ROTL64(e, 36)) ^ (ROTL64(e, 30)) ^ (ROTL64(e, 25))) + ((e & f) ^ (e & g) ^ (f & g)));

    c = c + (0x597F299CFC657E2A + beData[77] + ((ROTL64(h, 50)) ^ (ROTL64(h, 46)) ^ (ROTL64(h, 23))) + ((h & a) ^ (~h & b)));

    g = g + c;
    c = c + (((ROTL64(d, 36)) ^ (ROTL64(d, 30)) ^ (ROTL64(d, 25))) + ((d & e) ^ (d & f) ^ (e & f)));

    b = b + (0x5FCB6FAB3AD6FAEC + beData[78] + ((ROTL64(g, 50)) ^ (ROTL64(g, 46)) ^ (ROTL64(g, 23))) + ((g & h) ^ (~g & a)));

    f = f + b;
    b = b + (((ROTL64(c, 36)) ^ (ROTL64(c, 30)) ^ (ROTL64(c, 25))) + ((c & d) ^ (c & e) ^ (d & e)));

    a = a + (0x6C44198C4A475817 + beData[79] + ((ROTL64(f, 50)) ^ (ROTL64(f, 46)) ^ (ROTL64(f, 23))) + ((f & g) ^ (~f & h)));

    e = e + a;
    a = a + (((ROTL64(b, 36)) ^ (ROTL64(b, 30)) ^ (ROTL64(b, 25))) + ((b & c) ^ (b & d) ^ (c & d)));

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;

}

inline void RandomHash_SHA2_512(RH_StridePtr roundInput, RH_StridePtr output, SHA2_512_MODE mode)
{
    RH_ALIGN(64) uint64_t state[8];
    switch (mode)
    {
        case SHA2_512_MODE_384:
        {
            state[0] = 0xCBBB9D5DC1059ED8;
            state[1] = 0x629A292A367CD507;
            state[2] = 0x9159015A3070DD17;
            state[3] = 0x152FECD8F70E5939;
            state[4] = 0x67332667FFC00B31;
            state[5] = 0x8EB44A8768581511;
            state[6] = 0xDB0C2E0D64F98FA7;
            state[7] = 0x47B5481DBEFA4FA4;
        }break;
        case SHA2_512_MODE_512:
        {
            state[0] = 0x6A09E667F3BCC908;
            state[1] = 0xBB67AE8584CAA73B;
            state[2] = 0x3C6EF372FE94F82B;
            state[3] = 0xA54FF53A5F1D36F1;
            state[4] = 0x510E527FADE682D1;
            state[5] = 0x9B05688C2B3E6C1F;
            state[6] = 0x1F83D9ABFB41BD6B;
            state[7] = 0x5BE0CD19137E2179;
        }break;
        case SHA2_512_MODE_512_224:
        {
            state[0] = U64(0x8C3D37C819544DA2);
            state[1] = 0x73E1996689DCD4D6;;
            state[2] = 0x1DFAB7AE32FF9C82;;
            state[3] = 0x679DD514582F9FCF;;
            state[4] = 0x0F6D2B697BD44DA8;;
            state[5] = 0x77E36F7304C48942;;
            state[6] = 0x3F9D85A86A1D36C8;;
            state[7] = 0x1112E6AD91D692A1;;
        }break;
        case SHA2_512_MODE_512_256:
        {
            state[0] = 0x22312194FC2BF72C;;
            state[1] = U64(0x9F555FA3C84C64C2);;
            state[2] = 0x2393B86B6F53B151;;
            state[3] = U64(0x963877195940EABD);;
            state[4] = U64(0x96283EE2A88EFFE3);;
            state[5] = U64(0xBE5E1E2553863992);;
            state[6] = 0x2B0199FC2C85B8AA;;
            state[7] = 0x0EB72DDC81C52CA2;;
        }break;
    }
    int64_t oriLen = RH_STRIDE_GET_SIZE(roundInput);
    int64_t len = oriLen;
    uint32_t blockCount = (uint32_t)len / SHA2_512_BLOCK_SIZE;
    uint64_t *dataPtr = RH_STRIDE_GET_DATA64(roundInput);
    while(blockCount > 0)
    {
        SHA2_512_RoundFunction(dataPtr, state);
        len -= SHA2_512_BLOCK_SIZE;
        dataPtr += SHA2_512_BLOCK_SIZE / 8;
        blockCount--;
    }
	register uint64_t lowBits, hiBits;
	register int32_t padindex;
    RH_ALIGN(64) uint8_t pad[256];
	    
	lowBits = oriLen << 3;
	hiBits = oriLen >> 61;

	if (len < 112)
		padindex = 111 - (uint32_t)len;
	else
		padindex = 239 - (uint32_t)len;

	padindex++;
    RH_memzero_of16(pad, sizeof(pad));
	pad[0] = 0x80;

	hiBits = ReverseBytesUInt64(hiBits);

	ReadUInt64AsBytesLE(hiBits, pad+padindex);

	padindex = padindex + 8;

	lowBits = ReverseBytesUInt64(lowBits);

	ReadUInt64AsBytesLE(lowBits, pad+padindex);

	padindex = padindex + 8;

    memcpy(((uint8_t*)dataPtr) + len, pad, padindex);

    RH_ASSERT(((padindex + len) % SHA2_512_BLOCK_SIZE)==0);

    SHA2_512_RoundFunction(dataPtr, state);
    padindex -= SHA2_512_BLOCK_SIZE;
    if (padindex > 0)
        SHA2_512_RoundFunction(dataPtr + (SHA2_512_BLOCK_SIZE/8), state);
    RH_ASSERT(padindex > -SHA2_512_BLOCK_SIZE);

    dataPtr = RH_STRIDE_GET_DATA64(output);
    switch (mode)
    {
        case SHA2_512_MODE_384:
        {
            RH_STRIDE_SET_SIZE(output, 6 * 8);
            copy6_op(dataPtr, state, ReverseBytesUInt64);
        }break;
        case SHA2_512_MODE_512:
        {
            RH_STRIDE_SET_SIZE(output, 8 * 8);
            copy8_op(dataPtr, state, ReverseBytesUInt64);
        }break;
        case SHA2_512_MODE_512_224:
        {
            RH_STRIDE_SET_SIZE(output, 7 * 4);
            copy8_op(dataPtr, state, ReverseBytesUInt64);
        }break;
        case SHA2_512_MODE_512_256:
        {
            RH_STRIDE_SET_SIZE(output, 4 * 8);
            copy8_op(dataPtr, state, ReverseBytesUInt64);
        }break;
    }
    

}

