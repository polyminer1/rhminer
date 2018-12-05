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
    const U32 g_cpuRoundsThread = 64;
    m_localWorkSize = 64;
    UpdateWorkSize(g_cpuRoundsThread * g_cpuMinerThreads); //64

    RandomHash_DestroyMany(m_randomHashArray, g_cpuMinerThreads);
    RandomHash_CreateMany(&m_randomHashArray, g_cpuMinerThreads); 

    //Make all CPU miner threads
    for (U32 i=0; i < (U32)g_cpuMinerThreads; i++)
    {
        CPUKernelData* kdata = (CPUKernelData*)RH_SysAlloc(sizeof(CPUKernelData));
        memset(kdata, 0, sizeof(CPUKernelData));
        kdata->m_id = i;
        //kdata->m_signalPause = 1; //start paused
        kdata->m_thread = new std::thread([&,kdata] { RandomHashCpuKernel(kdata); });
        m_cpuKernels.push_back(kdata);
        kdata->m_thread->detach();
    }
}

void RandomHashCPUMiner::RandomHashCpuKernel(CPUKernelData* kernelData)
{
    char tname[64];
    snprintf(tname, 64, "Cpu%d", (int)kernelData->m_id);
    setThreadName(tname);

    if (g_setProcessPrio != 1)
    {
        if (kernelData->m_id == GpuManager::CpuInfos.numberOfProcessors-1) 
            RH_SetThreadPriority(RH_ThreadPrio_Normal);
        else
        {
            if (g_useCPU && !g_useGPU)
                RH_SetThreadPriority(RH_ThreadPrio_High);
        }
    }

    U32 workWindow = m_globalWorkSizePerCPUMiner;
    U32 gid = (U32)KernelOffsetManager::Increment(workWindow) - workWindow;
    U32 endFrame = gid + workWindow;
    bool paused = false;
    U64 oldID = U64_Max;
    while(!kernelData->m_abortThread)
    {
        RHMINER_RETURN_ON_EXIT_FLAG();
        U64 packageID = AtomicGet(kernelData->m_packageID);
        CPUKernelData::DataPackage* packageData = &kernelData->m_packages[packageID % CPUKernelData::PackagesCount];

        //handle internal pause
        if (oldID != packageID && paused)
        {
            paused = false;
        }

        //handle pause request from ::Pause()
        if (packageData->m_requestPause)
        {
            packageData->m_requestPause = 0;
            paused = true;
        }
        
        if (!paused)
        {
            if (g_disableCachedNonceReuse == true ||
                (g_disableCachedNonceReuse == false && oldID != packageID))
            {
                RandomHash_SetHeader(&m_randomHashArray[kernelData->m_id], packageData->m_header.asU8, (U32)packageData->m_nonce2); //copy header                
            }

    #ifdef RH_FORCE_PASCAL_V3_ON_CPU
            extern void PascalHashV3(void *state, const void *input);
            U32 gidBE = RH_swap_u32(gid); 
            packageData->m_header.asU32[PascalHeaderNoncePosV3 / 4] = gidBE;
            PascalHashV3(packageData->m_work1, packageData->m_header.asU8);
    #else
            //set start nonce here
            RandomHash_Search(&m_randomHashArray[kernelData->m_id], (U8*)packageData->m_work1, gid);
    #endif            
            if (RH_swap_u32(*(U32*)packageData->m_work1) <= packageData->m_target)
            {
                //Swapb256
                U32 *work = (uint32_t *)packageData->m_work1;
                U32 tmp[4] = {work[0], work[1], work[2], work[3]};
                work[0] = RH_swap_u32(work[7]);            
                work[1] = RH_swap_u32(work[6]);
                work[2] = RH_swap_u32(work[5]);
                work[3] = RH_swap_u32(work[4]);
                work[4] = RH_swap_u32(tmp[3]);
                work[5] = RH_swap_u32(tmp[2]);
                work[6] = RH_swap_u32(tmp[1]);
                work[7] = RH_swap_u32(tmp[0]);
                if (IsHashLessThan_32(work, packageData->m_targetFull))
                {
                    std::vector<U64> foundNonce;
#ifdef RH_FORCE_PASCAL_V3_ON_CPU
                    foundNonce.push_back(gidBE);
#else
                    foundNonce.push_back(m_randomHashArray[kernelData->m_id].m_startNonce);
#endif              
                    SolutionSptr solPtr = MakeSubmitSolution(foundNonce, true);
                    m_farm.submitProof(solPtr);

                    //pause all solutions until next package in solo
                    if (kernelData->m_isSolo)
                    {
                        paused = true;
                    }
                }
            }
            gid++;
            if (gid == endFrame)
            {
                gid = (U32)KernelOffsetManager::Increment(workWindow) - workWindow;
                endFrame = gid + workWindow;
            }
        }
        else
        {
            CpuSleep(20);
        }
        oldID = packageID;
        kernelData->m_hashes++;
    }
    AtomicSet(kernelData->m_abortThread, U32_Max);
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
    m_isInitialized = true;
    //start hashrate counting
    if (m_hashCountTime == 0)
        m_hashCountTime = TimeGetMilliSec();

    m_lastHashReading.resize(g_cpuMinerThreads);

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
    Guard g(m_pauseMutex);
    if (workStatus == PrepareWork_Nothing && m_isPaused == 1)
    {        
        CpuSleep(20);
    }
    else if (workStatus == PrepareWork_NewWork)
    {
        PascalWorkSptr wp = m_currentWp;
        SendWorkPackageToKernels(wp.get(), false);
    }
    
    m_isPaused = 0;
    return workStatus;
}

void RandomHashCPUMiner::SendWorkPackageToKernels(PascalWorkPackage* wp, bool requestPause)
{
    // makeup CPU work pakcage
    const U32 target = m_currentWp->GetDeviceTargetUpperBits();
    U32 savedNonce2 = wp->m_nonce2;
    for (auto& data : m_cpuKernels)
    {
        data->m_isSolo = wp->m_isSolo;
        U64 nextPackage = (AtomicGet(data->m_packageID) + 1);
        CPUKernelData::DataPackage* kernelData = &data->m_packages[nextPackage % CPUKernelData::PackagesCount ];
        
        memset(kernelData, 0, sizeof(CPUKernelData::DataPackage));
        kernelData->m_requestPause = !!requestPause;
        kernelData->m_headerSize = wp->m_fullHeader.size();
        memcpy(kernelData->m_header.asU8, &wp->m_fullHeader[0], wp->m_fullHeader.size());
        RHMINER_ASSERT(wp->m_fullHeader.size() <= sizeof(kernelData->m_header.asU8));

        if (wp->m_isSolo)
            memcpy(kernelData->m_targetFull, wp->m_soloTargetPow.data(), 32);
        else
            memcpy(kernelData->m_targetFull, wp->m_deviceBoundary.data(), 32);

        kernelData->m_nonce2 = savedNonce2;
        kernelData->m_target = target;
        
        RHMINER_ASSERT(wp->m_jobID.length() < sizeof(kernelData->m_workID)-1);
        memcpy(&kernelData->m_workID[0], wp->m_jobID.c_str(), wp->m_jobID.length()+1);
        kernelData->m_workID[wp->m_jobID.length()] = 0;

        //set next wp
        AtomicSet(data->m_packageID, nextPackage);
    }
}

void RandomHashCPUMiner::PauseCpuKernel()
{
    Guard g(m_pauseMutex);
    if (m_isPaused)
    {
        return;
    }
    
    //send pause package !
    PascalWorkSptr wp = m_currentWp;
    SendWorkPackageToKernels(wp.get(), true);
    m_isPaused = 1;
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
        k->m_abortThread = true;
    }
    CpuSleep(150);
    
    GenericCLMiner::Kill();
}



//unjam the queueKernel method
void RandomHashCPUMiner::SetWork(PascalWorkSptr _work)
{ 
    GenericCLMiner::SetWork(_work);
}

void RandomHashCPUMiner::QueueKernel()
{
    //wait 100ms
    S32 cnt = 5;
    while (!m_cpuKernels[0]->m_abortThread && cnt >= 0)
    {
        CpuSleep(20);
        cnt--;
        RHMINER_RETURN_ON_EXIT_FLAG();
    }

}

U64 RandomHashCPUMiner::GetHashRatePerSec() 
{
    U64 rate = 0;
    if (m_hashCountTime)
    {
        for(U32 i=0; i < m_cpuKernels.size(); i++)
        {
            U64 dt = 1;
            U64 kHash = m_cpuKernels[i]->m_hashes;
            if (kHash > m_lastHashReading[i])
            {
                dt = kHash - m_lastHashReading[i];
                m_lastHashReading[i] = kHash;
            }
            rate += dt;
        }
    }

    return rate;
}

void RandomHashCPUMiner::AddHashCount(U64 hashes)
{ 
    m_hashCountTime = TimeGetMilliSec();
}
