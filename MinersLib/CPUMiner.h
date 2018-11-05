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

#include "corelib/Worker.h"
#include "corelib/PascalWork.h"

struct CPUKernelData
{
    union
    {
        U8  asU8[256];
        U32 asU32[256/4];
    }                               m_header;
    RH_ALIGN(64) U8           m_targetFull[32];
    RH_ALIGN(64) U64          m_target;
    RH_ALIGN(64) U64          m_arg1;
    RH_ALIGN(64) U64          m_nonce2;
    RH_ALIGN(64) U64          m_startNonce;
    RH_ALIGN(64) U64          m_headerSize;
    RH_ALIGN(64) U64          m_id;               //id in the array of cpu kernels
    RH_ALIGN(64) U64          m_abordLoop;
    RH_ALIGN(64) U64          m_abordThread;
    RH_ALIGN(64) U64          m_running;
    RH_ALIGN(64) U64          m_itterations;
    RH_ALIGN(64) U64          m_isSolo;
    RH_ALIGN(64) U8           m_work1[256];       //return state for RandomHash
    RH_ALIGN(64) U8           m_work2[256];       //temp swapb buffer
    RH_ALIGN(64) std::thread* m_thread;
    RH_ALIGN(64) Event*       m_cpuKernelReadyEvent;
};
