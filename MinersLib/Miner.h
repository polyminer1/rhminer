/*
 This file is part of cpp-ethereum.

 cpp-ethereum is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 cpp-ethereum is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
 */
/** @file Miner.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 * @author Polyminer1 <https://github.com/polyminer1>
 * @date 2018
 */


#pragma once

#include "corelib/Log.h"
#include "corelib/Worker.h"
#include "corelib/PascalWork.h"
#include "wrapnvml.h"
#include "wrapadl.h"
#include "GpuManager.h"
#include "KernelOffsetManager.h"

#define MINER_WAIT_STATE_WORK	 1

using namespace std;

struct ServerCredential
{
	string host;
	string port;
	string user;
	string pass;
    const char* HostDescr() { return FormatString("%s:%s", host.c_str(), port.c_str()); }
};

enum PlatformType
{
    PlatformType_None,
    PlatformType_OpenCL,
    PlatformType_CUDA,
    PlatformType_CPU
};


class GenericCLMiner;

class Farm;
/// Describes the progress of a mining operation.
struct WorkingProgress
{
    std::vector<U32>    gpuGlobalIndex;
	std::vector<U32>    minersHasheRate;
    std::vector<U32>    acceptedShares;
    std::vector<U32>    rejectedShares;
    std::vector<float>  minersHasheRatePeak;
    std::vector<U32>    temperature;
    std::vector<U32>    fan;
    U64                 totalHashRate=0; //hash per sec
    string TemperatureToString();
};

class SolutionStats 
{
public:
    SolutionStats() { reset(); }
	
    void accepted(int gpuAbsIndex) 
    { 
        RHMINER_ASSERT(gpuAbsIndex < (int)RHMINER_ARRAY_COUNT(accepts));
        accepts[gpuAbsIndex]++;
    }

	void rejected(int gpuAbsIndex)
    {
        RHMINER_ASSERT(gpuAbsIndex < (int)RHMINER_ARRAY_COUNT(accepts));
        rejects[gpuAbsIndex]++;
    }
	
    void failed(int gpuAbsIndex) 
    {
        RHMINER_ASSERT(gpuAbsIndex < (int)RHMINER_ARRAY_COUNT(accepts));
        failures[gpuAbsIndex]++;
    }

    unsigned getAccepted(int gpuAbsIndex)
    {
        RHMINER_ASSERT(gpuAbsIndex < (int)RHMINER_ARRAY_COUNT(accepts));
        return accepts[gpuAbsIndex];
    }

    unsigned getRejected(int gpuAbsIndex)
    {
        RHMINER_ASSERT(gpuAbsIndex < (int)RHMINER_ARRAY_COUNT(accepts));
        return rejects[gpuAbsIndex];
    }

    unsigned getFailed(int gpuAbsIndex)
    {
        RHMINER_ASSERT(gpuAbsIndex < (int)RHMINER_ARRAY_COUNT(accepts));
        return failures[gpuAbsIndex];
    }
    string ToString(U64 lst);

	
    void reset() 
    { 
        memset(&accepts[0], 0, sizeof(accepts));
        memset(&rejects[0], 0, sizeof(rejects));
        memset(&failures[0], 0, sizeof(failures));
    }

    
    unsigned getTotalAccepts() { int sum = 0; for (auto x : accepts) sum += x; return sum; }
    unsigned getTotalRejects() { int sum = 0; for (auto x : rejects) sum += x; return sum; }
    unsigned getTotalFailures() { int sum = 0; for (auto x : failures) sum += x; return sum; }
    
    void        Begin() { startTime = TimeGetMicroSec(); }
    float       Elapsed() { return (TimeGetMicroSec() - startTime) / 1000 / 1000.0f;   }


	unsigned accepts[MAX_GPUS];
	unsigned rejects[MAX_GPUS];
	unsigned failures[MAX_GPUS];

    U64 startTime = 0;
};

class Miner;
class FarmFace;
class FarmFace
{
public:
	virtual ~FarmFace() = default;

	virtual void submitProof(SolutionSptr sol) = 0;
    virtual bool isMining() const = 0;
    virtual void RequestNewWork(PascalWorkSptr wp, GenericCLMiner* miner) = 0;
    virtual void ReconnectToServer(U32 minerRelIndex) =0;
};


class Miner: public Worker
{
public:
    Miner(std::string const& _name, FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex);
	virtual ~Miner() = default;

    virtual void InitFromFarm(U32 relativeIndex);
    virtual void StartWorking();
    virtual void WorkLoop();
    virtual bool WorkLoopStep() = 0;
    virtual void Kill();
    virtual void Pause();
    virtual void SetWork(PascalWorkSptr _work);
    virtual void SetWorkpackageDirty();
    virtual PlatformType GetPlatformType() { return PlatformType_None; }

    unsigned    getAbsoluteIndex() { return m_globalIndex; }
    bool        isInitializing() { return m_isInitializing.load(); }
    bool        IsInitializationDone() { return m_isInitializationDone; }

    U32         GetGlobalWorkSize()             { return m_globalWorkSize; }
    GpuManager::GPUInfos*  GetGpuInfoCache()    { return m_gpuInfoCache; }

    virtual void UpdateWorkSize(U32 absoluteVal) {};

    virtual U64     GetHashRatePerSec();
    virtual void    GetTemp(U32& temp, U32& fan);
    
protected:
    virtual bool    init(const PascalWorkSptr& work) = 0;
    virtual bool    ShouldInitialize(const PascalWorkSptr& work) = 0;

    PascalWorkSptr  GetWork() const { Guard l(m_workMutex); return m_workTemplate; }
    virtual void    AddHashCount(U64 hashes);

    std::atomic<bool> m_isInitializing;
    std::atomic<bool> m_isInitializationDone;
	U32             m_globalIndex = 0; //in all the devices list
    U32             m_relativeIndex = 0; //in farm list
	FarmFace&       m_farm;
    double          m_diff = 1.0;
    unsigned        m_globalWorkMult = 0;
    U32             m_localWorkSize = 0;
    U32             m_globalWorkSize = 0;
    unsigned        m_maxExtraTimePerWorkPackage = 61; //seconds
    U64             m_LastNewWorkStartTime = 0;
    U64             m_accumNewWorkDeltaTime = 0;
    U32             m_accumNewWorkDeltaTimeCount = 0;
    const string    c_DummyMinerName = "DummyMiner";

    U64              m_hashCount = 0;   //NOTE: CPU platform we do hash*1000 so we can have near 0 hash/sec
    U64              m_hashCountTime = 0;
    U64              m_resetHash = 0;
    
    Event           m_workReadyEvent; //only set when work is null
    PascalWorkSptr  m_workTemplate;
	mutable Mutex   m_workMutex;
    U32             m_workpackageDirty = 0;

    //temperature monitor and control
	wrap_nvml_handle *m_nvmlh = NULL;
	wrap_adl_handle *m_adlh = NULL;

    GpuManager::GPUInfos*  m_gpuInfoCache = 0;
};
typedef std::shared_ptr<Miner> MinerSptr;

