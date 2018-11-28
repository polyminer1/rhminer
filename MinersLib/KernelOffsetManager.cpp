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

#include "precomp.h"
#include "KernelOffsetManager.h"

inline U32 Rand32Range(U32 _min, U32 _max)
{
    U32 spot= 0;
    spot = _min + (rand32() % (_max-_min));
    return spot;
}

extern bool g_forceSequentialNonce;
U64 KernelOffsetManager::m_value = 0;

void KernelOffsetManager::Reset(U64 val)
{ 
    if (!g_forceSequentialNonce)
        val = Rand32Range(0, U32_Max - RHMINER_KB(10));

    AtomicSet(m_value, val);
}

U64 KernelOffsetManager::Increment(U32 increment)
{ 
    U64 val;
    val = AtomicAdd(m_value, increment) % U32_Max;
    return val;
}

