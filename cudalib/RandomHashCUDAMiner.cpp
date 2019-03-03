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
#include "RandomHashCUDAMiner.h"
#include "cuda_host_helper.h"
#include "corelib/CommonData.h"
#include "MinersLib/Pascal/PascalCommon.h"

extern bool g_kernelactivewaiting;
extern void cuda_randomhash_create(uint32_t blocks, uint32_t threadsPerBlock, uint32_t* input, U32 deviceID);
extern void cuda_RandomHash_SetTarget(uint64_t target);
extern void cuda_randomhash_init(uint32_t* input, U32 nonce2);    
extern bool g_disableCachedNonceReuse;


RandomHashCUDAMiner* CreateCudaMiner_PASCALV4()
{
    return (RandomHashCUDAMiner*)new RandomHashCUDAMiner();
}

void DestroyCudaMiner(RandomHashCUDAMiner* miner)
{
    delete miner;
} 

SynchroBuffer::SynchroBuffer()
{
	m_searchBuffer = new volatile uint32_t*;
    *m_searchBuffer = 0;
	//m_streams = new cudaStream_t;
    CUDA_SAFE_CALL(cudaStreamCreate(&m_stream));
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&m_readyEvent, cudaEventDisableTiming|cudaEventBlockingSync));
}


SynchroBuffer::~SynchroBuffer()
{
}


CudaWorkPackage& CudaWorkPackage::operator=(const CudaWorkPackage&w) 
{ 
    m_startNonce = w.m_startNonce;
    m_param32_1 = w.m_param32_1;               
    SetHeader(w.m_header, w.m_headerSize);      //NOTE: header of zero size DO NOT RESET THE CURRENT HEADER CONTENT !!!! 
    return *this;
}

RandomHashCUDAMiner::RandomHashCUDAMiner():m_outputBuffer(0)
{
}

RandomHashCUDAMiner::~RandomHashCUDAMiner()
{    
    cudaDeviceReset();
}

void RandomHashCUDAMiner::SetStartNonce(uint64_t baseNonce)
{
    m_currentWork.m_startNonce = baseNonce;
}

void RandomHashCUDAMiner::SetTarget(uint64_t target)  
{
    m_target = target;
}

bool RandomHashCUDAMiner::Init(const CudaWorkPackage& work, const CudaMinerValues& initData)
{
    m_target = 0;  
    m_currentWork = work;

    m_globalIndex =          initData.m_globalIndex;
    m_deviceID=              initData.m_deviceID;
    m_blockSize =            initData.m_blockSize;
    m_gridSize=              initData.m_gridSize;
    m_streamCount =          initData.m_streamCount;
    m_globalWorkSize =       initData.m_globalWorkSize;
    m_outputBufferSize =     initData.m_outputBufferSize;

    //FIX The stream count. No need for 2 for this algo
    m_streamCount = 1;

    CUDA_SET_DEVICE();
	CUDA_SAFE_CALL(cudaSetDeviceFlags(0));
	//CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    //CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual));
    

    if (m_outputBuffer)
    {
        if (g_kernelactivewaiting)
            cudaDeviceSynchronize();
        else
        {
            auto stream_index = m_kernelItterations % m_streamCount;
            CUDA_SAFE_CALL(cudaEventSynchronize(m_outputBuffer[stream_index].m_readyEvent));
        }

        delete[] m_outputBuffer;
        m_outputBuffer = 0;
    }
    m_outputBuffer = new SynchroBuffer[m_streamCount];
    
    cuda_randomhash_create(initData.m_blockSize, initData.m_gridSize, (U32*)work.m_header, m_globalIndex);
    cuda_randomhash_init((U32*)work.m_header, work.m_param32_1);

    m_kernelItterations = 0;

    try
    {
		//createing result buffers
		for (unsigned i = 0; i != m_streamCount; ++i)
		{
			CUDA_SAFE_CALL(cudaMallocHost((m_outputBuffer[i].m_searchBuffer), m_outputBufferSize));
		}

        return true;
	}
	catch (...)
	{
        PrintOut("RandomHashCUDAMiner::init exception\n"); 
	}
    
    return false;
}

void RandomHashCUDAMiner::ClearKernelOutputBuffer()
{
}

//must be called ONLY for new work packages
void RandomHashCUDAMiner::PrepareWork(const CudaWorkPackage& work)
{
    //mutex to avoid all N-1 GPU synking at the same time and timeouting the Nth one
    extern std::mutex*  gs_sequentialBuildMutex;
    std::lock_guard<std::mutex> g(*gs_sequentialBuildMutex);

    m_currentWork = work;
    if (m_kernelItterations)
    {
        if (g_kernelactivewaiting)
            cudaDeviceSynchronize();
        else
        {
            auto stream_index = m_kernelItterations % m_streamCount;
            CUDA_SAFE_CALL(cudaEventSynchronize(m_outputBuffer[stream_index].m_readyEvent));
        }

        cuda_RandomHash_SetTarget(m_target);
        
	    for (unsigned int i = 0; i < m_streamCount; i++)
		    *(m_outputBuffer[i].m_searchBuffer)[0] = 0;

        cuda_randomhash_init((U32*)work.m_header, work.m_param32_1);  
    }
    else
    {
        cuda_RandomHash_SetTarget(m_target);
    }
}

void cuda_randomhash_search(uint32_t blocks, uint32_t threads, cudaStream_t stream, uint32_t* input, uint32_t* output, U32 startNonce);
void RandomHashCUDAMiner::QueueKernel()
{
    RHMINER_RETURN_ON_EXIT_FLAG();

    RHMINER_ASSERT(m_globalWorkSize);

	auto stream_index = m_kernelItterations % m_streamCount;
	cudaStream_t stream = m_outputBuffer[stream_index].m_stream;
	volatile uint32_t* srchBuffer = *m_outputBuffer[stream_index].m_searchBuffer;

    U32 startnonce;
    if (g_disableCachedNonceReuse) 
    {
        startnonce = m_currentWork.m_startNonce;
    }
    else
    {
        startnonce = rand32(); 
    }

    m_lastPushTime = TimeGetMilliSec();
    cuda_randomhash_search(m_blockSize, m_gridSize, stream, (U32*)m_currentWork.m_header, (uint32_t*)srchBuffer, (U32)startnonce); 
    CUDA_SAFE_CALL(cudaEventRecord(m_outputBuffer[stream_index].m_readyEvent, 0));

    //inc stats
    m_kernelItterations++;
}

U32 RandomHashCUDAMiner::EvalKernelResult(uint32_t* outFoudnNonces, uint32_t maxCount, uint64_t& outBaseNonce)
{
	auto stream_index = m_kernelItterations % m_streamCount;
	cudaStream_t stream = m_outputBuffer[stream_index].m_stream;

	uint32_t foundCnt = 0;
	if (m_kernelItterations >= m_streamCount)
	{
        //wait for the kernel to finish
        cudaError_t err = cudaEventSynchronize(m_outputBuffer[stream_index].m_readyEvent);
        if (cudaSuccess != err) 
        {
            PrintOut("CudaError %s.\n", cudaGetErrorString(err));
            RHMINER_EXIT_APP("");
        }

        
	    volatile uint32_t* srchBuffer = *m_outputBuffer[stream_index].m_searchBuffer;

        foundCnt = srchBuffer[0];
		if (foundCnt) 
        {
			if (foundCnt > ((m_outputBufferSize/4) - 1))
				foundCnt = (m_outputBufferSize/4) - 1; 

			srchBuffer[0] = 0;

            for (U32 i = 0; i < foundCnt ; i++) 
                outFoudnNonces[i] = srchBuffer[i + 1];
            
            outBaseNonce = 0;
		}
	}

    return foundCnt;
}
        



#endif //RH_COMPILE_CPU_ONLY