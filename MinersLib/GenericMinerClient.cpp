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
#include "GenericMinerClient.h"
#include "MinersLib/CLMinerBase.h"
#include "corelib/PascalWork.h"
#include "corelib/miniweb.h"

#include "MinersLib/Farm.h"
#include "MinersLib/GenericCLMiner.h"

RHMINER_COMMAND_LINE_DEFINE_GLOBAL_INT(g_DisplaySpeedTimeout, 10); //seconds

extern bool g_appActive;

GenericMinerClient::GenericMinerClient():
    BaseMinerClient("Miner"),
    m_farm()
{
}    

void GenericMinerClient::PushMiniWebData(SolutionStats& farmSol, WorkingProgress& mp)
{
    string jdata;
    jdata = "{";
    U32 totSpeed = 0;
    jdata += "\"infos\":[";
    for (unsigned i = 0; i < mp.minersHasheRate.size(); i++)
    {
        totSpeed += mp.minersHasheRate[i];
        U32 threadCount;
        string name = GpuManager::Gpus[mp.gpuGlobalIndex[i]].gpuName;
        if (RHMINER_TEST_BIT(GpuManager::Gpus[mp.gpuGlobalIndex[i]].gpuType,GpuType_CPU))
            threadCount = g_cpuMinerThreads;
        else
            threadCount = GpuManager::Gpus[mp.gpuGlobalIndex[i]].globalWorkSize;

        jdata += FormatString("{\"name\":\"%s\", \"threads\":%u, \"speed\":%u, \"accepted\":%u, \"rejected\":%u , \"temp\":%u , \"fan\":%u }", 
            name.c_str(), 
            threadCount,
            mp.minersHasheRate[i],
            mp.acceptedShares[i],
            mp.rejectedShares[i],
            mp.temperature[i],
            mp.fan[i]);

        if (i + 1 != mp.minersHasheRate.size())
            jdata += ",";
    }
    jdata += FormatString("], ");
    
    jdata += FormatString("\"speed\":%u, ", totSpeed);
    jdata += FormatString("\"accepted\":%u, ", farmSol.getTotalAccepts());
    jdata += FormatString("\"rejected\":%u, ", farmSol.getTotalRejects());
    jdata += FormatString("\"failed\":%u, ", farmSol.getTotalFailures());
    jdata += FormatString("\"uptime\":%u, ", GlobalMiningPreset::I().GetUpTimeMS()/1000);
    jdata += FormatString("\"extrapayload\":\"%s\", ", g_extraPayload.c_str());
    jdata += FormatString("\"stratum.server\":\"%s:%s\", ",m_stratumClient->GetCurrentCred()->host.c_str(), m_stratumClient->GetCurrentCred()->port.c_str());
    jdata += FormatString("\"stratum.user\":\"%ss\", ", m_stratumClient->GetCurrentCred()->user.c_str());
    jdata += FormatString("\"diff\":%.8f", m_stratumClient->GetDiff());
    jdata += "}";
    SetMiniWebData(jdata);
    }
void GenericMinerClient::doStratum()
{ 
    FarmPreset* farmInfo = GlobalMiningPreset::I().Get();
	if (farmInfo->m_farmFailOverURL != "")
	{
		if (farmInfo->m_fuser != "")
		{
            m_stratumClient->SetFailover(farmInfo->m_farmFailOverURL, farmInfo->m_fport, farmInfo->m_fuser, farmInfo->m_fpass);
		}
		else
		{
			m_stratumClient->SetFailover(farmInfo->m_farmFailOverURL, farmInfo->m_fport);
		}
	}
	
	m_farm.onSolutionFound([&](SolutionSptr sol)
	{
        m_stratumClient->Submit(sol);
		return false;
	});
    m_farm.onRequestNewWork([&](PascalWorkSptr wp, GenericCLMiner* miner)
    {
        m_stratumClient->InitializeWP(wp); ///Will request new nonce !
    });
    m_farm.onReconnectFunc([&](U32 gpuAbsIndex)
    {
        if (gpuAbsIndex == 0xFFFFFFFF)
        {
            string connectParam;
            if (GlobalMiningPreset::I().UpdateToDevModeState(connectParam))
            {
                m_stratumClient->SetDevFeeCredentials(connectParam);
                m_stratumClient->ReconnectToServer();
            }
        }
        else
        {
            m_stratumClient->ReconnectToServer();
        }
    });

    while(1)
    {
        m_stratumClient->StartWorking();
        int coinCount = 1;

        h256 headerHash;

        while (m_stratumClient->isRunning())
	    {
            bool oncePerFRame = false;
        
            oncePerFRame = false;
		    auto mp = m_farm.miningProgress( false);

            if (m_stratumClient->isConnected())
		    {
                if (m_farm.IsOneMinerInitializing())
                {
                    CpuSleep(20);
                }
			    else if (m_stratumClient->GetCurrentWorkInfo(headerHash))
			    {
                    RHMINER_RETURN_ON_EXIT_FLAG()

                    auto farmSol = m_farm.GetSolutionStats();
				    PrintOut("%s\n", farmSol.ToString(m_stratumClient->GetLastSubmitTime()).c_str());

                    static U64 acc = 0;
                    static U32 cnt = 0;
                    cnt++ ;
                    acc += mp.totalHashRate;
                    PrintOut("%s AVG %.2f\n", mp.ToString().c_str(), acc / (float)cnt);                    

                    PushMiniWebData(farmSol, mp);

                    if (!oncePerFRame)
                    {
                        oncePerFRame = true;
                        string tmpStr = mp.TemperatureToString();
                        if (!tmpStr.empty())
                            PrintOut("%s\n", tmpStr.c_str());
                    }
                    
                    if (!m_farm.isMining())
                    {
                        if (m_stratumClient.get())
                        { 
                            m_stratumClient->CloseConnection();
                            m_stratumClient->StopWorking();
                        }
                        break;
                    }
			    }
		    }        

            for(auto x = 0; x < g_DisplaySpeedTimeout; x++)
            {   
                CpuSleep(1000);
            }
	    }
    }
}


