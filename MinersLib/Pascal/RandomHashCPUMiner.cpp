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
#include "MinersLib/Global.h"
#include "MinersLib/Pascal/RandomHashCPUMiner.h"
#include "MinersLib/Algo/sph_sha2.h"
#include "MinersLib/Algo/sph_blake.h"
#include "rhminer/ClientManager.h"
extern bool g_useGPU;

RandomHashCPUMiner::RandomHashCPUMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex) :
    GenericCLMiner(_farm, globalWorkMult, localWorkSize, gpuIndex),
    m_firstKernelCycleDone(false, false)
{
#ifdef RH_DISABLE_NONCE_REUSE_AND_CACHE
    g_disableCachedNonceReuse = true;
#endif
}

RandomHashCPUMiner::~RandomHashCPUMiner()
{
    RandomHash_DestroyMany(m_randomHashArray, g_cpuMinerThreads);
}

void RandomHashCPUMiner::InitFromFarm(U32 relativeIndex)
{
    //NOTE: WE need to force the local WS to 64 for UpdateWorkSize
    m_localWorkSize = 64;

    //Set a worksize for EACH CPU miner thread (TODO: tune to 100 ms/run)
    UpdateWorkSize(g_cpuRoundsThread * g_cpuMinerThreads);

    RandomHash_DestroyMany(m_randomHashArray, g_cpuMinerThreads);
    RandomHash_CreateMany(&m_randomHashArray, g_cpuMinerThreads); 

    //Make all CPU miner threads
    for (U32 i=0; i < (U32)g_cpuMinerThreads; i++)
    {
        CPUKernelData* kdata = (CPUKernelData*)RH_SysAlloc(sizeof(CPUKernelData));
        memset(kdata, 0, sizeof(CPUKernelData));
        kdata->m_id = i;
        kdata->m_cpuKernelReadyEvent = new Event(false, false);
        kdata->m_thread = new std::thread([&,kdata] { RandomHashCpuKernel(kdata); });
        m_cpuKernels.push_back(kdata);
        kdata->m_thread->detach();
    }
}

void RandomHashCPUMiner::RandomHashCpuKernel(CPUKernelData* kernelData)
{
    char* tname = (char*)malloc(64);
    snprintf(tname, 64, "Cpu%d", (int)kernelData->m_id);
    setThreadName(tname);
    
    if (g_setProcessPrio != 1)
    {
        if (kernelData->m_id == GpuManager::CpuInfos.numberOfProcessors-1)
            RH_SetThreadPriority(RH_ThreadPrio_Low);
        else
        {
            if (g_useCPU && !g_useGPU)
                RH_SetThreadPriority(RH_ThreadPrio_High);
        }
    }

    while(!kernelData->m_abordThread)
    {
        AtomicSet(kernelData->m_abordLoop, 0); 
        AtomicSet(kernelData->m_running, 0);
        kernelData->m_cpuKernelReadyEvent->WaitUntilDone();
        AtomicSet(kernelData->m_running, 1);

        RHMINER_RETURN_ON_EXIT_FLAG(); 
        if (kernelData->m_abordThread)
            break;
        
        U32 workSize = m_globalWorkSizePerCPUMiner;
        U32 gid = (U32)KernelOffsetManager::Increment(workSize) - workSize;
        
        if (g_disableCachedNonceReuse == true || 
            (g_disableCachedNonceReuse == false && memcmp(m_randomHashArray[kernelData->m_id].m_cachedHheader, kernelData->m_header.asU8, PascalHeaderSize - 4) != 0))
        {
            RandomHash_SetHeader(&m_randomHashArray[kernelData->m_id], kernelData->m_header.asU8, (U32)kernelData->m_nonce2); //copy header
        }
        
        while(workSize)
        {
            U64 isAbort = AtomicGet(kernelData->m_abordLoop) || kernelData->m_abordThread;
            if (isAbort)
            {
                break;
            }

#ifdef RH_FORCE_PASCAL_V3_ON_CPU
            extern void PascalHashV3(void *state, const void *input);
            U32 gidBE = RH_swap_u32(gid); 
            kernelData->m_header.asU32[PascalHeaderNoncePosV3 / 4] = gidBE;
            PascalHashV3(kernelData->m_work1, kernelData->m_header.asU8);
#else
            //set start nonce here
            RandomHash_Search(&m_randomHashArray[kernelData->m_id], (U8*)kernelData->m_work1, gid);
#endif            
            swab256(kernelData->m_work2, kernelData->m_work1);
            U32 leftMost256 = *(U32*)(kernelData->m_work2 + 28);
            if (leftMost256 <= kernelData->m_target)
            {
                if (IsHashLessThan_32(kernelData->m_work2, kernelData->m_targetFull))
                {
                    std::vector<U64> foundNonce;
#ifdef RH_FORCE_PASCAL_V3_ON_CPU
                    foundNonce.push_back(gidBE);
#else                    
                    foundNonce.push_back(m_randomHashArray[kernelData->m_id].m_startNonce);
#endif              
                    SolutionSptr solPtr = MakeSubmitSolution(foundNonce, true);
                    m_farm.submitProof(solPtr);
                    CpuSleep(100);

                    break;
                }
            }        
            gid++;
            workSize--;
        }
        
        kernelData->m_itterations++;

        if (kernelData->m_id == 0)
        {
            m_firstKernelCycleDone.SetDone();
        }
    }
    AtomicSet(kernelData->m_abordThread, U64_Max);
}

void RandomHashCPUMiner::UpdateWorkSize(U32 absoluteVal)
{
    if (!absoluteVal)
        return;

    if (m_globalWorkSize && m_globalWorkSize != absoluteVal)
        RHMINER_EXIT_APP("Cpu miner cannot set an arbitrary worksize\n");    

    CLMinerBase::UpdateWorkSize(absoluteVal);

    m_globalWorkSize = absoluteVal;

    //spread WORK among all cpu miners -> absoluteVal = x * g_cpuMinerThreads
    m_globalWorkSizePerCPUMiner = m_globalWorkSize / g_cpuMinerThreads;
    if (!m_globalWorkSizePerCPUMiner)
        m_globalWorkSizePerCPUMiner = 1;
}


bool RandomHashCPUMiner::init(const PascalWorkSptr& work)
{
    //Generic CPU thread init. Set this thread at high prio
    m_isInitialized = true;
    //start hashrate counting
    if (m_hashCountTime == 0)
        m_hashCountTime = TimeGetMilliSec();
    
    //set this thread at high prio
    if (g_useCPU && !g_useGPU && g_setProcessPrio != 1)
        RH_SetThreadPriority(RH_ThreadPrio_High);
    return true;
}


bool RandomHashCPUMiner::configureGPU()
{
    return true;
}

PrepareWorkStatus RandomHashCPUMiner::PrepareWork(const PascalWorkSptr& workTempl, bool reuseCurrentWP)
{
    PrepareWorkStatus workStatus = GenericCLMiner::PrepareWork(workTempl, reuseCurrentWP);    
    
    //in case we're pause, the workStatus will be PrepareWork_Nothing, BUT we need to restart the cpu kernel...
    U32 oldPause = AtomicSet(m_isPaused, 0);
    if (workStatus == PrepareWork_Nothing && oldPause == 1)
    {
        for (auto& k : m_cpuKernels)
        {
            //Start all cpu kernels
            k->m_cpuKernelReadyEvent->SetDone();
        }
    }
    else if (workStatus == PrepareWork_NewWork)
    {
        //was not allready paused ??
        if (oldPause == 0)
        {
            //pause the cpu kernels
            PauseCpuKernel();
            AtomicSet(m_isPaused, 0);
        }

        //wait for the threads to pause
        U32 doneCnt = 0;
        U64 timtoutSec = TimeGetMilliSec() + 5000;
        while(doneCnt <= m_cpuKernels.size())
        {
            if (TimeGetMilliSec() > timtoutSec)
            {
                PrintOut("Error. A cpu miner is stalled. Abording new work\n");
                return workStatus;
            }

            for (auto& k : m_cpuKernels)
            {
                U64 isRunning = AtomicGet(k->m_running);
                if (!k->m_abordLoop || isRunning == 0)
                    doneCnt++;
            }
        }
        
        m_firstKernelCycleDone.Reset();
        SendWorkPackageToKernels(m_currentWp.get());
    }

    return workStatus;
}

void RandomHashCPUMiner::SendWorkPackageToKernels(PascalWorkPackage* wp)
{
    // makeup CPU work pakcage
    const U32 target = m_currentWp->GetDeviceTargetUpperBits();

    U32 savedNonce2 = wp->m_nonce2;
    for (auto& k : m_cpuKernels)
    {
        k->m_headerSize = wp->m_fullHeader.size();
        memcpy(k->m_header.asU8, &wp->m_fullHeader[0], wp->m_fullHeader.size());
        RHMINER_ASSERT(wp->m_fullHeader.size() <= sizeof(k->m_header.asU8));

        if (wp->m_isSolo)
            memcpy(k->m_targetFull, wp->m_soloTargetPow.data(), 32);
        else
            memcpy(k->m_targetFull, wp->m_deviceBoundary.data(), 32);

        k->m_nonce2 = savedNonce2;
        k->m_target = target;
        k->m_isSolo = wp->m_isSolo;

        //Start all cpu kernels
        k->m_cpuKernelReadyEvent->SetDone();
    }
}

void RandomHashCPUMiner::PauseCpuKernel()
{
    U32 oldPause = AtomicSet(m_isPaused, 1);
    if (oldPause)
    {
        return;
    }

    for (auto& k : m_cpuKernels)
    {
        k->m_cpuKernelReadyEvent->Reset();
        AtomicSet(k->m_abordLoop, 1);
    }
    CpuSleep(20);
}

void RandomHashCPUMiner::Pause()
{
    GenericCLMiner::Pause();
    PauseCpuKernel();
}

void RandomHashCPUMiner::Kill()
{
    for (auto& k : m_cpuKernels)
    {
        k->m_abordThread = true;
        k->m_cpuKernelReadyEvent->SetDone();
    }
    CpuSleep(50);
    
    AtomicSet(m_setWorkComming, 1);
    m_firstKernelCycleDone.SetDone();
    CpuSleep(20);
    GenericCLMiner::Kill();
}

//unjam the queueKernel method
void RandomHashCPUMiner::SetWork(PascalWorkSptr _work)
{ 
    if (AtomicGet(m_waitingForKernel))
    {
        AtomicSet(m_setWorkComming, 1);
        m_firstKernelCycleDone.SetDone();
        CpuSleep(20);
    }

    GenericCLMiner::SetWork(_work);
}

void RandomHashCPUMiner::QueueKernel()
{
    AtomicSet(m_waitingForKernel, 1);
    m_firstKernelCycleDone.WaitUntilDone(); 
    AtomicSet(m_waitingForKernel, 0);
    RHMINER_RETURN_ON_EXIT_FLAG();

    U32 ittCount = (U32)m_cpuKernels[0]->m_itterations;
    U32 deltaItt = ittCount - m_lastIttCount;
    m_lastIttCount = ittCount;

    //aborting due to new work pending
    if (AtomicSet(m_setWorkComming, 0) == 1)
    {
        return;
    }
    m_firstKernelCycleDone.Reset();

    m_totalKernelItterations += deltaItt;
}


