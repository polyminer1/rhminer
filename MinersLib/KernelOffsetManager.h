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

#include "corelib/Log.h"
#include "corelib/Worker.h"
#include "corelib/PascalWork.h"

using namespace std;

struct KernelOffsetManager
{
    static U64      GetCurrentValue() { return AtomicGet(m_value);  }
    static void     Reset(U64 val);
    static U64      Increment(U32 increment); //return val += inc;
    static U32      GetNextSearchNonce(){ return AtomicIncrement(m_searchNonce);  }
    static void     ResetSearchNonce(U32 v) { AtomicSet(m_searchNonce, v);  }

protected:
    static U64 m_value;
    static U32 m_searchNonce;
};

