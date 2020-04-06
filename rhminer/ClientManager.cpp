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

#include "precomp.h"
#include "BuildInfo.h"
#include "MinersLib/StratumClient.h"
#include "MinersLib/Pascal/RandomHashCLMiner.h"
#include "ClientManager.h"

extern bool g_appActive;
extern void StopMiniWeb();

ClientManager & ClientManager::I()
{
    static ClientManager* i = 0;
    if (i == 0)
        i = new ClientManager;
    return *i;
}

U64 g_shutdownWD = 0;
U32 g_shutdownDone = 0;
void ClientManager::Shutdown(eShutdownMode esmode)
{
    if (esmode ==  eShutdownLite)
    {
        if (ActiveClients.client.get())
            ActiveClients.client->GetFarm().Pause();
        CpuSleep(200);
    }
    else if (esmode == eShutdownRestart || esmode == eShutdownFull)
    {
    }
    else if (esmode == eShutdownFull)
    {
        /*
        auto t = new std::thread([&]()
        {
            g_shutdownWD = TimeGetMilliSec() + 60 * 1000;
            while (TimeGetMilliSec() < g_shutdownWD)
                CpuSleep(20);
            
            if (AtomicGet(g_shutdownDone) == 0)
                RHMINER_EXIT_APP("Shutdown takes to long, exiting miner now.\n");

        });
        */

        //stop all now!
        StopMiniWeb();

        PrintOutSilent("Shutdown: stratum client\n");
        ActiveClients.stratum->Disconnect();

        PrintOutSilent("Shutdown: miner threads\n");
        if (ActiveClients.client.get())
            ActiveClients.client->GetFarm().Stop();

        if (esmode == eShutdownFull)
            ActiveClients.stratum->Kill();
        
        PrintOutSilent("Shutdown: logs\n");
        AtomicSet(g_shutdownDone, 1);
        CpuSleep(200);

        CloseLog();
    }
    
}

void ClientManager::Initialize()
{
    ////////////////////////////////////////////////////////////
    //
    //          Setup gpu map
    bool gpuSwitchPresent = CmdLineManager::GlobalOptions().FindSwitch("-gpu") || 
                            (CmdLineManager::GlobalOptions().FindSwitch("-cpu"));
    bool disableGpuSwitcht = CmdLineManager::GlobalOptions().FindSwitch("-gpu") == false &&  
                             CmdLineManager::GlobalOptions().FindSwitch("-gputhreads") == false;

    /////////////////////////////////////////////////
    //with no cmdline ,enable mining for GPUS+cpu
    if (!gpuSwitchPresent)
    {
        for(U32 i = 0; i < GpuManager::Gpus.size(); ++i)
        {
            auto&g = GpuManager::Gpus[i];
            g.enabled = true;
            
            //at leaseone thread on cpu with default options
            if (RHMINER_TEST_BIT(g.gpuType, GpuType_CPU))
            {
                g_useCPU = true;
                if (g_cpuMinerThreads == 0)
                    g_cpuMinerThreads = GpuManager::CpuInfos.numberOfProcessors;
            }
            else if (disableGpuSwitcht)
            {
                g.enabled = false;
            }
        }
    }

    /////////////////////////////////////////////////
    //mining for CPU
    if (g_useCPU)
    {
#ifdef _WIN32_WINNT
        if (GpuManager::CpuInfos.UserSelectedCoresCount != 0)
        {
            DWORD_PTR vProcessMask;
            DWORD_PTR vSystemMask;
            HANDLE h;
            int targetPid = (int)GetCurrentProcessId();
            h = OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_SET_INFORMATION, FALSE, (DWORD) targetPid);
            if (h == NULL)
                PrintOut("Error. Cannot acquire process to set affinity mask\n");
            else
            {
                if (GetProcessAffinityMask(h, &vProcessMask, &vSystemMask))
                {
                    DWORD_PTR newMask = vSystemMask & GpuManager::CpuInfos.UserSelectedCores;
                    if (newMask)
                        SetProcessAffinityMask(h, newMask);
                }
            }
        }
#endif
        RHMINER_ASSERT(GpuManager::CpuInfos.numberOfProcessors != 0);

        //calc ideal CPU count if count is 0
        if (g_cpuMinerThreads == 0)
            g_cpuMinerThreads = GpuManager::CpuInfos.numberOfProcessors;

        if (g_cpuMinerThreads > (int)GpuManager::CpuInfos.numberOfProcessors &&
            g_disableMaxGpuThreadSafety == false)
            g_cpuMinerThreads = (int)GpuManager::CpuInfos.numberOfProcessors;

        //manually enable the CPU in gpumanager
        for(auto& g : GpuManager::Gpus)
        {
            if (RHMINER_TEST_BIT(g.gpuType, GpuType_CPU))
                g.enabled = true;
        }
    }

    ActiveClients.client = std::shared_ptr<GenericMinerClient>(new GenericMinerClient());
    ActiveClients.client->SetStratumClient<StratumClient>(ActiveClients.stratum);
    ActiveClients.client->InitGpu<RandomHashCLMiner>();

    if (g_testPerformance)
        GlobalMiningPreset::I().DoPerformanceTest();

    ////////////////////////////////////////////////////////////
    //
    //  start all client's thread !
    //
    if (ActiveClients.client.get())
    {
        ActiveClients.client->StartWorking();
        CpuSleep(100);
    }
}
