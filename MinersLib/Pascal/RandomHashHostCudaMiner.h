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

#include "RandomHashCLMiner.h"

#ifndef RH_COMPILE_CPU_ONLY
#include "MinersLib/CudaHostMinerProxy.h"

class RandomHashHostCudaMiner: public RandomHashCLMiner
{
    
public:
	RandomHashHostCudaMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex);
	~RandomHashHostCudaMiner() override;

    static bool  configureGPU();
    virtual U32  GetOutputBufferSize() {return (/*CUDA_SEARCH_RESULT_BUFFER_SIZE*/1024 + 1)*sizeof(U32);}
    
    virtual void ClearKernelOutputBuffer() { m_cudaMinerProxy->ClearKernelOutputBuffer();  }
    virtual PlatformType GetPlatformType() { return PlatformType_CUDA; }

    virtual bool init(const PascalWorkSptr& work);
    virtual void InitFromFarm(U32 relativeIndex);

protected:
    virtual PrepareWorkStatus PrepareWork(const PascalWorkSptr& workTempl, bool reuseCurrentWP = false);

    bool BuildKernels(const PascalWorkSptr& work) { return true; }

    virtual void QueueKernel()
    {
        m_cudaMinerProxy->QueueKernel();

        //inc stats
        m_kernelItterations++;

        m_workOffset = KernelOffsetManager::Increment(m_globalWorkSize) - m_globalWorkSize;
        m_cudaMinerProxy->SetStartNonce(m_workOffset);
    }

    virtual void EvalKernelResult();
    CudaHostMinerProxy m_cudaMinerProxy;
};

#endif //RH_COMPILE_CPU_ONLY