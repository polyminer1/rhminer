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

#include "MinersLib/CLMinerBase.h"
#include "cudalib/RandomHashCUDAMiner.h"


//Hold the RandomHashCUDAMiner instance
class CudaHostMinerProxy
{
public:
    CudaHostMinerProxy(RandomHashCUDAMiner* _cudaMiner):miner(_cudaMiner){}
    ~CudaHostMinerProxy()
    {
        DestroyCudaMiner(miner);
        miner = 0;
    }

    RandomHashCUDAMiner* operator->() const { return miner; }

protected:
    RandomHashCUDAMiner*   miner;
};
