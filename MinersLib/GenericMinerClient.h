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

#include "GenericFarmClient.h"
#include "StratumClient.h"

using namespace std;

RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("displayspeedtimeout", g_DisplaySpeedTimeout, "General", "Display mining speeds every x seconds. Default is 10 ", 0, S32_Max)

class BaseMinerClient: public Worker
{
public:
    BaseMinerClient(const char* threadName) :Worker(threadName){}
    virtual ~BaseMinerClient() {};
    bool IsRunning() {return m_running;}

protected:
    virtual void WorkLoop()
    {
        try
        {
		    doStratum();
        }
        catch (...)
        {
            PrintOut("Error. Exception in ");
        }
        m_running = false;
    }
    virtual void doStratum() = 0;

	/// Mining options
	bool m_running = true;
};

//------------------------------------------------------------------------------------------------------------------

class GenericMinerClient: public BaseMinerClient
{
public:
    GenericMinerClient();
    template <typename CL_MINER>
    void InitGpu()
    {
        if (!CL_MINER::configureGPU())
        {
            RHMINER_EXIT_APP("No gpu enabled or found.\n");
        }
    }

    template <typename STRATUM_CLIENT>
    void SetStratumClient(StratumClientSptr& stratumAutoPtrRef)
    {
        FarmPreset* farmInfos = GlobalMiningPreset::I().Get();
        if ((farmInfos->m_farmURL.empty() || 
             farmInfos->m_port.empty()) &&
            !g_testPerformance)
        {
            PrintOut("No url provided to mine.\n"); 
            RHMINER_EXIT_APP("");
        }


        if (stratumAutoPtrRef.get() == 0)
        {
            stratumAutoPtrRef = StratumClientSptr(new STRATUM_CLIENT( 
                StratumInit(&m_farm,
                            farmInfos->m_farmURL, 
                            farmInfos->m_port, 
                            farmInfos->m_user, 
                            farmInfos->m_pass, 
                            farmInfos->m_maxFarmRetries+1, 
                            farmInfos->m_email, 
                            farmInfos->m_soloOvertStratum)));
        }

        m_stratumClient = stratumAutoPtrRef;
    }


    StratumClientSptr    GetStratum() { return m_stratumClient; }

    Farm&   GetFarm() { return m_farm; }

private:
    virtual void doStratum();
    void PushMiniWebData(SolutionStats& farmSol, WorkingProgress& mp);
    
    StratumClientSptr           m_stratumClient;
    GenericFarmClientSptr       m_farmCLient;
    Farm                        m_farm;

    static const U32            m_speedSMACount = 20;
    std::vector<U64>            m_speedSMA;
    
    //watchdogs
    std::thread*                m_WatchdogDevFee = 0;
    U64                         m_zeroSpeedWD = 0;
};
