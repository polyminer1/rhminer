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

#pragma once
#include "corelib/basetypes.h"

const static U32 PascalHeaderSize        = 200; //Fixed header size for pool mining
const static U32 PascalHeaderNonce2Pos   = 116;
const static U32 PascalHeaderNoncePosV3  = (PascalHeaderSize-4);
#define          PascalHeaderNoncePosV4(headerSize) (headerSize - 4)


#define RH_M                    (RHMINER_KB(10)*5)
#define RH_M_MEM_PER_THREAD_MB  8.8f 

#define RH_N                    5
#define RH_STRIDE_SIZE_FACTOR   32

// M*=5
#define RH_StrideArrayCount     31
#define RH_StrideSize           208896

#define RH_CheckerSize          (sizeof(U64))
#define RH_WorkSize             RH_StrideSize 
#define RH_IDEAL_ALIGNMENT      64  //NOTE : optimiz -> This should be changed for CUDA (some gpu are 256 bytes align, TODO: test ! )

