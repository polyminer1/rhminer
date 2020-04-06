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

#ifndef RH_COMPILE_CPU_ONLY
#include "RandomHashHostCudaMiner.h"
#include "MinersLib/Global.h"
#include "rhminer/ClientManager.h"

RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL("kernelactivewaiting", g_kernelactivewaiting, "Gpu", "Enable active waiting on kernel run.\nThis will raise cpu usage but bring more stability, specially when mining on multiple gpu.\nWARNING: This affect cpu mining");
RHMINER_COMMAND_LINE_DEFINE_GLOBAL_BOOL(g_kernelactivewaiting, false);

using namespace std;

extern RandomHashCUDAMiner* CreateCudaMiner_PASCALV4();

RandomHashHostCudaMiner::RandomHashHostCudaMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex) :
    RandomHashCLMiner(_farm, globalWorkMult, localWorkSize, gpuIndex),
    m_cudaMinerProxy(CreateCudaMiner_PASCALV4())
{
}

RandomHashHostCudaMiner::~RandomHashHostCudaMiner()
{
}

bool RandomHashHostCudaMiner::configureGPU()
{
    return GpuManager::SetupGPU();
}

void RandomHashHostCudaMiner::InitFromFarm(U32 relativeIndex)
{
    RandomHashCLMiner::InitFromFarm(relativeIndex);
}

bool RandomHashHostCudaMiner::init(const PascalWorkSptr& work)
{
    if ((m_globalWorkSize % m_gpuInfoCache->localWorkSize) != 0)
        RHMINER_EXIT_APP("Global thread counts must be perfect ratio of localWorkSize");

    std::lock_guard<std::mutex> g(*gs_sequentialBuildMutex);

    CudaMinerValues cudaInit;
    cudaInit.m_globalIndex = m_globalIndex;
    cudaInit.m_deviceID = m_gpuInfoCache->deviceID;
    cudaInit.m_blockSize = m_gpuInfoCache->localWorkSize;
    cudaInit.m_gridSize = m_globalWorkSize / m_gpuInfoCache->localWorkSize;
    cudaInit.m_streamCount = 1;
    cudaInit.m_globalWorkSize = m_globalWorkSize;
    cudaInit.m_outputBufferSize = GetOutputBufferSize();
    
    if (m_hashCountTime == U64_Max)
        m_hashCountTime = TimeGetMilliSec();

    CudaWorkPackage cudaWP;
    cudaWP.m_param32_1 = work->m_nonce2;
    cudaWP.m_startNonce = m_workOffset;
    cudaWP.SetHeader(work->m_fullHeader.data(), GetHeaderBufferSize());
    m_isInitialized = m_cudaMinerProxy->Init(cudaWP, cudaInit);

    return m_isInitialized;
}

PrepareWorkStatus RandomHashHostCudaMiner::PrepareWork(const PascalWorkSptr& workTempl, bool reuseCurrentWP)
{
    PrepareWorkStatus workStatus = RandomHashCLMiner::PrepareWork(workTempl, reuseCurrentWP);
     
    if (workStatus == PrepareWork_NewWork)
    {    
        PascalWorkPackage* wp = workTempl.get();
        
        m_cudaMinerProxy->SetTarget(m_currentWp->GetDeviceTargetUpperBits64());

        CudaWorkPackage cudaWP;
        cudaWP.m_param32_1 = workTempl->m_nonce2;
        cudaWP.SetHeader(workTempl->m_fullHeader.data(), GetHeaderBufferSize());
        cudaWP.m_startNonce = m_workOffset;

        m_cudaMinerProxy->PrepareWork(cudaWP);
    }

    return workStatus;
}

void RandomHashHostCudaMiner::EvalKernelResult()
{
    U32 nonces32[1024];
    U64 baseNonce64 = 0;

    U32 foundCnt = m_cudaMinerProxy->EvalKernelResult(nonces32, sizeof(nonces32), baseNonce64);

    if (foundCnt) 
    {           
        std::vector<U64> nonces;
        nonces.reserve(foundCnt);
        string tmp = " found nonce : ";
        for (U32 i = 0; i < foundCnt ; i++) 
        {
            tmp += FormatString("%u ", nonces32[i]);
            nonces.push_back(nonces32[i]);
        }
        
        m_startNonce = baseNonce64;
        SolutionSptr solPtr = MakeSubmitSolution(nonces, m_currentWp->m_nonce2, false);
        m_farm.submitProof(solPtr);
    }
}


#endif //RH_COMPILE_CPU_ONLY
