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
    jdata += FormatString("\"stratum.server\":\"%s:%s\", ", m_stratumClient->GetCurrentCred()->host.c_str(), m_stratumClient->GetCurrentCred()->port.c_str());
    jdata += FormatString("\"stratum.user\":\"%s\", ", m_stratumClient->GetCurrentCred()->user.c_str());
    jdata += FormatString("\"diff\":%.8f", m_stratumClient->GetDiff());
    jdata += "}";

    U32 totalAcc = 0;
    U32 totalRej = 0;
    string detailHR;
    string tempFan;
    for (unsigned i = 0; i < mp.minersHasheRate.size(); i++)
    {
        totalAcc += mp.acceptedShares[i];
        totalRej += mp.rejectedShares[i];
        detailHR += FormatString("%u", mp.minersHasheRate[i]);
        tempFan += FormatString("%u;%u", mp.temperature[i], mp.fan[i]);

        if (i + 1 != mp.minersHasheRate.size())
        {
            detailHR += ";";
            tempFan += ";";
        }
    }


    string ethManData = "{\"result\": [";
    //Version
    ethManData += FormatString("\"%s - PASC\", ", RH_PROJECT_VERSION);
    
    //Running time in minutes
    ethManData += FormatString("\"%u\", ", GlobalMiningPreset::I().GetUpTimeMS() / 1000 / 60);
    
    //total ETH hashrate in MH / s, number of ETH shares, number of ETH rejected shares.
    ethManData += FormatString("\"%u;%u;%u\",", totSpeed, totalAcc, totalRej);

    //detailed ETH hashrate for all GPUs.
    ethManData += FormatString("\"%s\",", detailHR.c_str());
    ethManData += "\"0;0;0\", ";
    ethManData += "\"off;off;off;off;off;off\", ";
    
    //Temperature and Fan speed(%) pairs for all GPUs.
    ethManData += FormatString("\"%s\",", tempFan.c_str());

    //current mining pool.For dual mode, there will be two pools here.
    ethManData += FormatString("\"%s:%s\", ", m_stratumClient->GetCurrentCred()->host.c_str(), m_stratumClient->GetCurrentCred()->port.c_str());
    //number of ETH invalid shares, number of ETH pool switches, number of DCR invalid shares, number of DCR pool switches.
    ethManData += "\"0;0;0;0\"";

    ethManData += "]}";

    SetMiniWebData(jdata, ethManData);
}

void GenericMinerClient::doStratum()
{ 
    FarmPreset* farmInfo = GlobalMiningPreset::I().Get();
	if (farmInfo->m_farmFailOverURL != "")
	{
        m_stratumClient->SetFailover(farmInfo->m_farmFailOverURL, farmInfo->m_fport, farmInfo->m_fuser, farmInfo->m_fpass);
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

                //start devfee time watchdog
                if (!m_WatchdogDevFee)
                {
                    m_WatchdogDevFee = new std::thread([&]()
                    {
                        while(1)
                        {
                            try
                            {
                                CpuSleep(1000);
                                if (!g_ExitApplication)
                                {
                                    if (GlobalMiningPreset::I().DetectDevfeeOvertime())
                                        m_stratumClient->ReconnectToServer();
                                    }
                                }
                            catch (...){}
                        }
                    });
                }
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

                    SolutionStats farmSol = m_farm.GetSolutionStats();
				    PrintOut("%s\n", farmSol.ToString(m_stratumClient->GetLastSubmitTime()).c_str());

                    //calc SMA5
                    m_speedSMA.push_back(mp.totalHashRate);
                    if (m_speedSMA.size() > m_speedSMACount)
                        m_speedSMA.erase(m_speedSMA.begin());
                    U64 sma = 0;
                    for (auto speed : m_speedSMA)
                        sma += speed;
                    
                    string str;                    
                    str = FormatString("Speed : %s ", HashrateToString(sma/(float)m_speedSMA.size()));
                    str += "(";
                    for (unsigned i = 0; i < mp.minersHasheRate.size(); i++)
                    {
                        auto mh = pround(mp.minersHasheRate[i], 2);
                        str += FormatString("%s %s", GpuManager::Gpus[mp.gpuGlobalIndex[i]].gpuName.c_str(), HashrateToString(mh));
                        if (i + 1 != mp.minersHasheRate.size())
                            str += " ";
                    }
                    str += "). ";

                    PrintOut("%s\n", str.c_str()); 

                    //check if all are not disabled !
                    U32 enabled = 0;
                    for (auto& g : GpuManager::Gpus)
                        if (g.enabled)
                            enabled++;

                    // Hanle zero speed watchdog
                    if (mp.totalHashRate < 2 && enabled > 0)
                    {
                        if (!m_zeroSpeedWD)
                            m_zeroSpeedWD = TimeGetMilliSec() + 60*1000; //60 sec zero speed watchtog
                    }
                    else
                        m_zeroSpeedWD = 0;

                    if (m_zeroSpeedWD && TimeGetMilliSec() > m_zeroSpeedWD)
                    {
                        RHMINER_EXIT_APP("Speed to low.");
                    }

                    PushMiniWebData(farmSol, mp);

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
                if (m_stratumClient->IsWorkTimedOut() /*&& m_stratumClient->isConnected()*/)
                {
                    PrintOut("WorkTimeout reached. No new work received after %u seconds.\n",  g_workTimeout);
                    if (m_stratumClient.get())
#ifdef _WIN32_WINNT
                        m_stratumClient->CloseConnection();
                    else
#endif
                        RHMINER_EXIT_APP("Connection timeout.\n");
                }
                CpuSleep(1000);

                if (GlobalMiningPreset::I().RestartRequested() == GlobalMiningPreset::eInteralRestart /*&& m_farm.isMining()*/)
                {
                    PrintOutCritical("Restarging rhminer...\n");
                    //once every 1 sec
                    if (m_stratumClient.get())
                    {
                        m_farm.Pause();
                        m_stratumClient->CloseConnection();
                        if (g_appActive)
                            m_stratumClient->StopWorking();
                    }
                
                    PrintOut("Force stoping farm ...\n");
                    m_farm.Stop();

                    if (GlobalMiningPreset::I().GetPendingConfigFile().length())
                    {
                        string cfg = GlobalMiningPreset::I().GetPendingConfigFile().c_str();
                        GlobalMiningPreset::I().SetLastConfigFile("");
                        CmdLineManager::GlobalOptions().LoadFromXml(cfg.c_str());

                        //apply new network cred

                    }

                    GlobalMiningPreset::I().SetRestart(GlobalMiningPreset::eExternalRestart);
                    break;
                }
            }
	    }
    }
}


