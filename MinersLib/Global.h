/**
 * Global miner data source code implementation
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

#include "MinersLib/Miner.h"
#include "corelib/Worker.h"
#include "corelib/PascalWork.h"
#include "MinersLib/CLMinerBase.h"
#include "rhminer/CommandLineManager.h"

RHMINER_COMMAND_LINE_DECLARE_GLOBAL_STRING("logfilename", g_logFileName, "General", "Set the name of the log's filename. Note: the log file will be overwritten every time you start rhminer");
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL("cpu", g_useCPU, "Gpu", "Enable the use of CPU to mine. ex '-cpu -cputhreads 4' will enable mining on cpu while gpu mining.");
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("cputhreads", g_cpuMinerThreads, "Gpu", "Number of CPU miner threads when mining with CPU. ex: -cpu -cputhreads 4", 0, S32_Max);
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("testperformance", g_testPerformance, "Debug", "Run performance test for an amount of seconds", 0, 120)
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("testperformancethreads", g_testPerformanceThreads, "Debug", "Amount of threads to use for performance test", 0, 256)
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("processpriority", g_setProcessPrio, "General", "On windows only. Set miner's process priority. 0=Background Process, 1=Low Priority, 2=Normal Priority, 3=High Priority. Default is 3. WARNING: Changing this value will affect GPU mining.", 0, 10);
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL("disablefasttransfo", g_disableFastTransfo, "General", "Disable fast transfo. This can help old cpu go faster.");

class FarmFace;

using namespace std;

struct FarmPreset
{
    string m_farmURL = "http://127.0.0.1:4009";
	string m_user;
	string m_pass;
	string m_port;
	string m_farmFailOverURL = "";
	string m_fuser = "";
	string m_fpass = "";
	string m_fport = "";
	string m_email = "";
    unsigned m_maxFarmRetries = 3;          //retries retries2
    bool     m_soloOvertStratum = false; //when server starts with HTTP://
};

///////////////////////////////////////////////////
//  
//  Miner Creators
struct MinerCreatorFunc
{
    std::function<Miner*(FarmFace& farm, U32 gpuIindex) > OpenCLCreator;
    std::function<Miner*(FarmFace& farm, U32 gpuIindex) > NvidiaCreator;
    std::function<Miner*(FarmFace& farm, U32 gpuIindex) > CPUCreator;
};

typedef std::function < string(const string& args, const string& context)> ApiFunction;

class GlobalMiningPreset
{
    public:
        GlobalMiningPreset();
        static GlobalMiningPreset& I();

        void Initialize(char** argv, int argc);

        ///////////////////////////////////////////////////
        //  
        //  Preset management
        FarmPreset* Get();

        ///////////////////////////////////////////////////
        //
        //  Class Creator functions
        //
        template <class CL_CLASS, class NVIDIA_CLASS, class CPU_CLASS>
        inline MinerCreatorFunc MakeMinerCreator()
        {
            return { [](FarmFace& farm, U32 gpuIndex) {return new CL_CLASS(farm, 0, 0, gpuIndex); },
                     [](FarmFace& farm, U32 gpuIndex) {return new NVIDIA_CLASS(farm, 0, 0, gpuIndex); },
                     [](FarmFace& farm, U32 gpuIndex) {return new CPU_CLASS(farm, 0, 0, gpuIndex); }
            };
        }
        enum CreatorClasType {ClassOpenCL, ClassNvidia, ClassCPU};
        Miner* CreateMiner(CreatorClasType type, FarmFace& _farm, U32 gpuIndex);
        
        ///////////////////////////////////////////////////
        //  
        //  Stats
        U32 GetUpTimeMS();
        void DoPerformanceTest();

        ///////////////////////////////////////////////////
        //  
        //  Dev mode managements
        //
        U64         GetTotalDevFeeTime24H() { return m_totalDevFreeTimeToDayMS; }
        bool        IsInDevFeeMode() { return !!AtomicGet(m_endOfCurrentDevFeeTimesMS); }
        bool        DetectDevfeeOvertime();
        void        GetRandomDevCred(string& configStr);
        bool        UpdateToDevModeState(string& connectionParams);
        float       m_devfeePercent = 1.0f;
        inline void RegisterDevCredentials(const strings& servers, const strings& walletAddr)
        {
            for(auto& server : servers)
                for (auto& addr : walletAddr)
                    m_devModeWallets.push_back(server + "\t" + addr);
        }

        //Local difficulty
        float m_localDifficulty = 0.0f;

    protected:
        U64          m_startTimeMS = 0;
        
        std::mutex   *devFeeMutex;
        U64          m_devFeeTimer24hMS = 0;
        vector<U64>  m_nextDevFeeTimesMS;
        U64          m_currentDevFeeTimesMS = 0;
        U64          m_endOfCurrentDevFeeTimesMS = 0;
        U64          m_totalDevFreeTimeToDayMS = 0;
        strings      m_devModeWallets;

        void SetStratumInfo(const string& val);
        void FailOverURL(const string& val);
        FarmPreset  m_presets;
};

