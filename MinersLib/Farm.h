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
#include "MinersLib/Miner.h"

class GenericCLMiner;

class Farm: public FarmFace
{
public:
    Farm()
    {
        m_submitID = 0;
    }

	~Farm()
	{
		stop();
	}

    void SetWork(PascalWorkSptr _wp);
    void SetWorkpackageDirty();
    

    bool start();
	
    bool IsOneMinerInitializing();
    bool HasOneCPUMiner();


    void stop();
    void Pause();
    void PauseCpuMiners();

	virtual bool isMining() const	{ return m_isMining; }

    WorkingProgress const& miningProgress(bool reset = false);
    
    bool DetectDeadMiners();

	SolutionStats   GetSolutionStats() { return m_farmData.m_solutionStats;	}
    void            AddFailedSolution(int gpuAbsIndex) { m_farmData.m_solutionStats.failed(gpuAbsIndex); }
    void            AddAcceptedSolution(int gpuAbsIndex);
    void            AddRejectedSolution(int gpuAbsIndex);
    
    unsigned        GetMinersCount() { return m_minersCount; }
    MinerSptr       GetCPUMiner();

	using SolutionFound = std::function<bool(SolutionSptr)>;
    using RequestNewWorkFunc = std::function<void(PascalWorkSptr, GenericCLMiner*)>;    
    using ReconnectFunc = std::function<void(uint32_t )>;

	void onSolutionFound(SolutionFound const& _handler) { m_onSolutionFound = _handler; }
    void onRequestNewWork(RequestNewWorkFunc const& _handler) { m_requestNeWork = _handler;}
    void onReconnectFunc(ReconnectFunc const& _handler) { m_reconnect = _handler; }

    PascalWorkSptr GetWork() const { Guard l(m_farmData.m_workMutex); return m_farmData.m_work; }
    
    void ReconnectToServer(uint32_t minerRelIndex) override;
    void RequestNewWork(PascalWorkSptr wp, GenericCLMiner* miner) override;

private:
   
    void internalStop();
    void submitProof(SolutionSptr sol) override;

	void resetTimer(){ m_farmData.m_lastStart = std::chrono::steady_clock::now();	}

	mutable Mutex          m_minerWorkMutex;
	std::vector<MinerSptr> m_miners;
    std::atomic<unsigned>  m_minersCount;
    U64                    m_lastProgressTime = 0;

    Mutex m_sumbitMutex;
    std::map<unsigned, std::pair<std::thread*, bool>> m_submiters;
    std::atomic<uint32_t > m_submitID;

    struct Farmdata
    {
        mutable Mutex                           m_workMutex;
        PascalWorkSptr                          m_work;
	    mutable Mutex                           m_progressMutex;
	    mutable WorkingProgress                 m_progress;
        mutable std::vector<float>              m_minersHasheRatePeak;
	    std::chrono::steady_clock::time_point   m_lastStart;
	    mutable SolutionStats                   m_solutionStats;
        U32                                     m_consecutiveRejectedCount = 0;
        U64                                     m_lastRejectedTimeMS = 0;
    };
    Farmdata m_farmData;

	std::atomic<bool> m_isMining = {false};
    
	SolutionFound m_onSolutionFound;
    RequestNewWorkFunc m_requestNeWork;
    ReconnectFunc m_reconnect;
}; 
