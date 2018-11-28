/**
 * RandomHash source code implementation
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
/// @copyright Polyminer1

#pragma once

#include "corelib/Worker.h"
#include "corelib/PascalWork.h"
#include "MinersLib/CLMinerBase.h"
#include "MinersLib/Pascal/PascalCommon.h"
 
class GenericCLMiner: public CLMinerBase
{    
public:
    GenericCLMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex);
    ~GenericCLMiner();
    void KernelCallBack();
    virtual bool init(const PascalWorkSptr& work);

protected:
    bool IsWorkStalled();
    virtual bool WorkLoopStep();

    // Get all codes and kernel name
    virtual U32 GetOutputMaxCount(){ return MAX_GPUS; }
    virtual U32 GetOutputBufferSize() {return (GetOutputMaxCount() + 1)*sizeof(U32);}
    virtual U32 GetHeaderBufferSize() { return PascalHeaderSize; }


    // must use m_queue in m_context to load/store buffers
    virtual void                  QueueKernel();
    virtual PrepareWorkStatus     PrepareWork(const PascalWorkSptr& workTempl, bool reuseCurrentWP = false);
    virtual void                  EvalKernelResult();
    virtual SolutionSptr          MakeSubmitSolution(const std::vector<U64>& nonces, bool isFromCpuMiner);
    virtual void                  SetSearchKernelCurrentTarget(U32 paramIndex, cl::Kernel& searchKernel);
    virtual void                  ClearKernelOutputBuffer(); 
    virtual KernelCodeAndFuctions GetKernelsCodeAndFunctions(); 

    //work management
    U64             m_lastWorkStartTimeMs = 0;
    U32             m_kernelItterations = 0;
    U64             m_workOffset = 0; 
    U64             m_startNonce = 0;
    unsigned        m_recycleCount = 0;
    bool            m_sleepWhenWorkFinished= false;
    
    static const int MaxWorkPackageTimeout = (5 * 60);
    unsigned        m_maxRetryCycleCount = 2;

    PascalWorkSptr m_lastWorkTemplate; 
    PascalWorkSptr m_currentWp;  

    //make a zero buff for fast reset later on
    bytes            m_zeroBuffer;
    std::vector<U32> m_results;

    cl::Kernel m_searchKernel; //default kernel
    cl::Buffer m_kernelHeader;  
    cl::Buffer m_kernelOutput;
};


