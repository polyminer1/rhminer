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

#define SHA2_256_BLOCK_SIZE 64

void CUDA_SYM_DECL(SHA2_256_RoundFunction)(uint32_t* data, uint32_t* state)
{
    uint32_t A, B, C, D, E, F, G, H, T, T2;

    //scratch buffer 64 uint32
    uint32_t beData[64];
    for (A = 0; A < 64 / 4; A++)
        beData[A] = ReverseBytesUInt32(data[A]); //swap 64 first bytes

    A = state[0];
    B = state[1];
    C = state[2];
    D = state[3];
    E = state[4];
    F = state[5];
    G = state[6];
    H = state[7];


    // Step 1

    T = beData[14];
    T2 = beData[1];
    beData[16] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[9] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[0];

    T = beData[15];
    T2 = beData[2];
    beData[17] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[10] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[1];

    T = beData[16];
    T2 = beData[3];
    beData[18] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[11] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[2];

    T = beData[17];
    T2 = beData[4];
    beData[19] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[12] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[3];

    T = beData[18];
    T2 = beData[5];
    beData[20] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[13] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[4];

    T = beData[19];
    T2 = beData[6];
    beData[21] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[14] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[5];

    T = beData[20];
    T2 = beData[7];
    beData[22] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[15] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[6];

    T = beData[21];
    T2 = beData[8];
    beData[23] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[16] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[7];

    T = beData[22];
    T2 = beData[9];
    beData[24] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[17] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[8];

    T = beData[23];
    T2 = beData[10];
    beData[25] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[18] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[9];

    T = beData[24];
    T2 = beData[11];
    beData[26] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[19] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[10];

    T = beData[25];
    T2 = beData[12];
    beData[27] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[20] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[11];

    T = beData[26];
    T2 = beData[13];
    beData[28] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[21] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[12];

    T = beData[27];
    T2 = beData[14];
    beData[29] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[22] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[13];

    T = beData[28];
    T2 = beData[15];
    beData[30] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[23] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[14];

    T = beData[29];
    T2 = beData[16];
    beData[31] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[24] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[15];

    T = beData[30];
    T2 = beData[17];
    beData[32] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[25] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[16];

    T = beData[31];
    T2 = beData[18];
    beData[33] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[26] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[17];

    T = beData[32];
    T2 = beData[19];
    beData[34] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[27] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[18];

    T = beData[33];
    T2 = beData[20];
    beData[35] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[28] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[19];

    T = beData[34];
    T2 = beData[21];
    beData[36] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[29] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[20];

    T = beData[35];
    T2 = beData[22];
    beData[37] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[30] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[21];

    T = beData[36];
    T2 = beData[23];
    beData[38] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[31] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[22];

    T = beData[37];
    T2 = beData[24];
    beData[39] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[32] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[23];

    T = beData[38];
    T2 = beData[25];
    beData[40] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[33] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[24];

    T = beData[39];
    T2 = beData[26];
    beData[41] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[34] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[25];

    T = beData[40];
    T2 = beData[27];
    beData[42] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[35] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[26];

    T = beData[41];
    T2 = beData[28];
    beData[43] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[36] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[27];

    T = beData[42];
    T2 = beData[29];
    beData[44] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[37] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[28];

    T = beData[43];
    T2 = beData[30];
    beData[45] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[38] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[29];

    T = beData[44];
    T2 = beData[31];
    beData[46] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[39] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[30];

    T = beData[45];
    T2 = beData[32];
    beData[47] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[40] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[31];

    T = beData[46];
    T2 = beData[33];
    beData[48] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[41] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[32];

    T = beData[47];
    T2 = beData[34];
    beData[49] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[42] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[33];

    T = beData[48];
    T2 = beData[35];
    beData[50] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[43] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[34];

    T = beData[49];
    T2 = beData[36];
    beData[51] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[44] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[35];

    T = beData[50];
    T2 = beData[37];
    beData[52] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[45] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[36];

    T = beData[51];
    T2 = beData[38];
    beData[53] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[46] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[37];

    T = beData[52];
    T2 = beData[39];
    beData[54] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[47] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[38];

    T = beData[53];
    T2 = beData[40];
    beData[55] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[48] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[39];

    T = beData[54];
    T2 = beData[41];
    beData[56] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[49] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[40];

    T = beData[55];
    T2 = beData[42];
    beData[57] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[50] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[41];

    T = beData[56];
    T2 = beData[43];
    beData[58] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[51] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[42];

    T = beData[57];
    T2 = beData[44];
    beData[59] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[52] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[43];

    T = beData[58];
    T2 = beData[45];
    beData[60] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[53] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[44];

    T = beData[59];
    T2 = beData[46];
    beData[61] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[54] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[45];

    T = beData[60];
    T2 = beData[47];
    beData[62] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[55] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[46];

    T = beData[61];
    T2 = beData[48];
    beData[63] = ((ROTR32(T, 17)) ^ (ROTR32(T, 19)) ^ (T >> 10)) + beData[56] + ((ROTR32(T2, 7)) ^ (ROTR32(T2, 18)) ^ (T2 >> 3)) + beData[47];

    // Step 2

    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0x428A2F98 + beData[0];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0x71374491 + beData[1];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0xB5C0FBCF + beData[2];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0xE9B5DBA5 + beData[3];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0x3956C25B + beData[4];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0x59F111F1 + beData[5];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0x923F82A4 + beData[6];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0xAB1C5ED5 + beData[7];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0xD807AA98 + beData[8];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0x12835B01 + beData[9];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0x243185BE + beData[10];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0x550C7DC3 + beData[11];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0x72BE5D74 + beData[12];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0x80DEB1FE + beData[13];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0x9BDC06A7 + beData[14];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0xC19BF174 + beData[15];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0xE49B69C1 + beData[16];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0xEFBE4786 + beData[17];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0x0FC19DC6 + beData[18];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0x240CA1CC + beData[19];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0x2DE92C6F + beData[20];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0x4A7484AA + beData[21];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0x5CB0A9DC + beData[22];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0x76F988DA + beData[23];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0x983E5152 + beData[24];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0xA831C66D + beData[25];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0xB00327C8 + beData[26];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0xBF597FC7 + beData[27];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0xC6E00BF3 + beData[28];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0xD5A79147 + beData[29];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0x06CA6351 + beData[30];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0x14292967 + beData[31];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0x27B70A85 + beData[32];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0x2E1B2138 + beData[33];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0x4D2C6DFC + beData[34];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0x53380D13 + beData[35];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0x650A7354 + beData[36];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0x766A0ABB + beData[37];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0x81C2C92E + beData[38];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0x92722C85 + beData[39];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0xA2BFE8A1 + beData[40];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0xA81A664B + beData[41];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0xC24B8B70 + beData[42];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0xC76C51A3 + beData[43];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0xD192E819 + beData[44];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0xD6990624 + beData[45];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0xF40E3585 + beData[46];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0x106AA070 + beData[47];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0x19A4C116 + beData[48];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0x1E376C08 + beData[49];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0x2748774C + beData[50];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0x34B0BCB5 + beData[51];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0x391C0CB3 + beData[52];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0x4ED8AA4A + beData[53];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0x5B9CCA4F + beData[54];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0x682E6FF3 + beData[55];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;
    T = H + ((ROTR32(E, 6)) ^ (ROTR32(E, 11)) ^ (ROTR32(E, 25))) + ((E & F) ^ (~E & G)) + 0x748F82EE + beData[56];
    T2 = ((ROTR32(A, 2)) ^ (ROTR32(A, 13)) ^ ((A >> 22) ^ (A << 10))) + ((A & B) ^ (A & C) ^ (B & C));
    H = T + T2;
    D = D + T;
    T = G + ((ROTR32(D, 6)) ^ (ROTR32(D, 11)) ^ (ROTR32(D, 25))) + ((D & E) ^ (~D & F)) + 0x78A5636F + beData[57];
    T2 = ((ROTR32(H, 2)) ^ (ROTR32(H, 13)) ^ ((H >> 22) ^ (H << 10))) + ((H & A) ^ (H & B) ^ (A & B));
    G = T + T2;
    C = C + T;
    T = F + ((ROTR32(C, 6)) ^ (ROTR32(C, 11)) ^ (ROTR32(C, 25))) + ((C & D) ^ (~C & E)) + 0x84C87814 + beData[58];
    T2 = ((ROTR32(G, 2)) ^ (ROTR32(G, 13)) ^ ((G >> 22) ^ (G << 10))) + ((G & H) ^ (G & A) ^ (H & A));
    F = T + T2;
    B = B + T;
    T = E + ((ROTR32(B, 6)) ^ (ROTR32(B, 11)) ^ (ROTR32(B, 25))) + ((B & C) ^ (~B & D)) + 0x8CC70208 + beData[59];
    T2 = ((ROTR32(F, 2)) ^ (ROTR32(F, 13)) ^ ((F >> 22) ^ (F << 10))) + ((F & G) ^ (F & H) ^ (G & H));
    E = T + T2;
    A = A + T;
    T = D + ((ROTR32(A, 6)) ^ (ROTR32(A, 11)) ^ (ROTR32(A, 25))) + ((A & B) ^ (~A & C)) + 0x90BEFFFA + beData[60];
    T2 = ((ROTR32(E, 2)) ^ (ROTR32(E, 13)) ^ ((E >> 22) ^ (E << 10))) + ((E & F) ^ (E & G) ^ (F & G));
    D = T + T2;
    H = H + T;
    T = C + ((ROTR32(H, 6)) ^ (ROTR32(H, 11)) ^ (ROTR32(H, 25))) + ((H & A) ^ (~H & B)) + 0xA4506CEB + beData[61];
    T2 = ((ROTR32(D, 2)) ^ (ROTR32(D, 13)) ^ ((D >> 22) ^ (D << 10))) + ((D & E) ^ (D & F) ^ (E & F));
    C = T + T2;
    G = G + T;
    T = B + ((ROTR32(G, 6)) ^ (ROTR32(G, 11)) ^ (ROTR32(G, 25))) + ((G & H) ^ (~G & A)) + 0xBEF9A3F7 + beData[62];
    T2 = ((ROTR32(C, 2)) ^ (ROTR32(C, 13)) ^ ((C >> 22) ^ (C << 10))) + ((C & D) ^ (C & E) ^ (D & E));
    B = T + T2;
    F = F + T;
    T = A + ((ROTR32(F, 6)) ^ (ROTR32(F, 11)) ^ (ROTR32(F, 25))) + ((F & G) ^ (~F & H)) + 0xC67178F2 + beData[63];
    T2 = ((ROTR32(B, 2)) ^ (ROTR32(B, 13)) ^ ((B >> 22) ^ (B << 10))) + ((B & C) ^ (B & D) ^ (C & D));
    A = T + T2;
    E = E + T;

    state[0] += A;
    state[1] += B;
    state[2] += C;
    state[3] += D;
    state[4] += E;
    state[5] += F;
    state[6] += G;
    state[7] += H;
}


void CUDA_SYM_DECL(RandomHash_SHA2_256)(RH_StridePtr roundInput, RH_StridePtr output)
{

    // optimized algo
    RH_ALIGN(64) uint32_t state[8] = { 
        0x6A09E667, 
        0xBB67AE85, 
        0x3C6EF372, 
        0xA54FF53A, 
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19};

    //RandomHash_MD_BASE_MAIN_LOOP(SHA2_256_BLOCK_SIZE, _CM(SHA2_256_RoundFunction), ReverseBytesUInt64);
    {
    int32_t len = (int32_t)RH_STRIDE_GET_SIZE(roundInput);                             
    uint32_t blockCount = len / SHA2_256_BLOCK_SIZE;                                       
    uint32_t *dataPtr = (uint32_t *)RH_STRIDE_GET_DATA(roundInput);                    
    uint64_t bits = len * 8;                                                           
    while(blockCount > 0)                                                              
    {                                                                                  
        _CM(SHA2_256_RoundFunction)(dataPtr, state);                                                     
        len -= SHA2_256_BLOCK_SIZE;                                                        
        dataPtr += SHA2_256_BLOCK_SIZE / 4;                                                
        blockCount--;                                                                  
    }                                                                                  
    {                                                                                  
		int32_t padindex;                                                              
        RH_ALIGN(64) uint8_t pad[72];                                                               
		                                                                               
		if (len < 56)                                                                  
			padindex = 56 - len;                                                       
		else                                                                           
			padindex = 120 - len;                                                      
                                                                                       
        PLATFORM_MEMSET(pad, 0, sizeof(pad));
		pad[0] = 0x80;                                                                 
        bits = ReverseBytesUInt64(bits);                                                    
		ReadUInt64AsBytesLE(bits, pad+padindex);                                       
                                                                                       
		padindex = padindex + 8;                                                       
        memcpy(((uint8_t*)dataPtr) + len, pad, padindex);                              
        RH_ASSERT(padindex <= 72);                                                   
        RH_ASSERT(((padindex + len) % SHA2_256_BLOCK_SIZE)==0);                          
                                                                                       
		_CM(SHA2_256_RoundFunction)(dataPtr, state);                                                     
        padindex -= SHA2_256_BLOCK_SIZE;                                                   
        if (padindex > 0)                                                              
            _CM(SHA2_256_RoundFunction)(dataPtr+(SHA2_256_BLOCK_SIZE/4), state);                             
        RH_ASSERT(padindex > -SHA2_256_BLOCK_SIZE);                                      
    }}



    //get the hash result IN BE
    RH_STRIDE_SET_SIZE(output, 8 * 4);
    uint32_t* dataPtr = (uint32_t*)RH_STRIDE_GET_DATA(output);
    copy8_op(dataPtr, state, ReverseBytesUInt32);
}
