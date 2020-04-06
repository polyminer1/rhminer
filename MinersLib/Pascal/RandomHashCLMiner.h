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
#include "MinersLib/GenericCLMiner.h"




class RandomHashCLMiner: public GenericCLMiner
{
    
public:
    RandomHashCLMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex);
    virtual bool init(const PascalWorkSptr& work);

    //called only once
    static bool configureGPU();

protected:
    virtual void            QueueKernel();
};

