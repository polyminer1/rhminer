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

#include "MinersLib/GenericCLMiner.h"
#include "MinersLib/CPUMiner.h"
#include "MinersLib/Pascal/RandomHash.h"
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("cputhrottling", g_cputhrottling, "General", "Slow down mining by internally throttling the cpu. \nThis is usefull to prevent virtual computer provider throttling vCpu when mining softwares are detected\nMin-Max are 0 and 99.\nEx. -cputhrottling 12 will throttle the cpu 12% of the time", 0, 99);

class RandomHashCPUMiner: public GenericCLMiner
{
    
public:
    RandomHashCPUMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex);
    ~RandomHashCPUMiner();

    virtual bool init(const PascalWorkSptr& work);
    virtual void InitFromFarm(U32 relativeIndex);
    static bool configureGPU();
    virtual PlatformType GetPlatformType() { return PlatformType_CPU; }

    virtual void Pause();
    virtual void Kill();
    virtual void SetWork(PascalWorkSptr _work);


protected:
    
    vector<CPUKernelData*>   m_cpuKernels; //paged alloc
    Event  m_firstKernelCycleDone;
    U32    m_setWorkComming = 0;
    U32    m_waitingForKernel = 1;
    U32    m_lastIttCount = 0;
    U32    m_isPaused = 0;
    std::mutex  m_pauseMutex;
    U32    m_globalWorkSizePerCPUMiner = 0;
    mersenne_twister_state   m_rnd32;

    virtual KernelCodeAndFuctions GetKernelsCodeAndFunctions() { return KernelCodeAndFuctions(); }
    virtual void ClearKernelOutputBuffer() {}
    virtual void EvalKernelResult() {}

    virtual PrepareWorkStatus PrepareWork(const PascalWorkSptr& workTempl, bool reuseCurrentWP = false);
    virtual void SendWorkPackageToKernels(PascalWorkPackage* wp, bool requestPause = false);
    virtual void QueueKernel();
    virtual void AddHashCount(U64 hashes);
    virtual U64 GetHashRatePerSec();
    std::vector<U64> m_lastHashReading;

    void PauseCpuKernel();
    void UpdateWorkSize(U32 absoluteVal);
    void RandomHashCpuKernel(CPUKernelData* kernelData); 
    RandomHash_State* m_randomHash2Array = 0;    
};

