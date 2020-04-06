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

const static U32 PascalHeaderSize        = 200; //Fixed header size for pool mining RandomHash1
const static U32 PascalHeaderSizeV5      = 236; //Fixed header size for pool mining RandomHash2
#define          PascalHeaderNoncePosV4(headerSize) (headerSize - 4)

#define RH2_StrideArrayCount         (1024+64)

#define RH2_MIN_N 2
#define RH2_MAX_N 4
#define RH2_MIN_J 1
#define RH2_MAX_J 8
#define RH2_M 64
#define RH2_StrideSize           (208896)


#define RH_CheckerSize          (sizeof(U64))

#define RH_IDEAL_ALIGNMENT      64  
#define RH_IDEAL_ALIGNMENT32    (RH_IDEAL_ALIGNMENT/4)


