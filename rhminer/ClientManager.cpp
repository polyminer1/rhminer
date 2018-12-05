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

ClientManager & ClientManager::I()
{
    static ClientManager* i = 0;
    if (i == 0)
        i = new ClientManager;
    return *i;
}


void ClientManager::Shutdown()
{
    if (ActiveClients.client.get())
        ActiveClients.client->GetFarm().Pause();
    CpuSleep(500);
}

void ClientManager::Initialize()
{
    ////////////////////////////////////////////////////////////////////////////////
    //
    //          Register wallets
    {
        GlobalMiningPreset::I().RegisterDevCredentials(
            {
                "hashplaza.org\t1379"
             },
             { "523057-58.0.rig",
               "529692-23.0.rig"
             });
    }

    ////////////////////////////////////////////////////////////
    //
    //          Setup gpu map
    bool gpuSwitchPresent = CmdLineManager::GlobalOptions().FindSwitch("-gpu") || 
                            (CmdLineManager::GlobalOptions().FindSwitch("-cpu"));

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
                    g_cpuMinerThreads = 1;
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
            g_cpuMinerThreads = GpuManager::CpuInfos.numberOfProcessors - 1;

        if (g_cpuMinerThreads > (int)GpuManager::CpuInfos.numberOfProcessors)
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
