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

#pragma once
#include "stdint.h"
#include <cuda_runtime.h>
#include <algorithm>

struct SynchroBuffer
{
    SynchroBuffer();
    ~SynchroBuffer();
	volatile uint32_t**     m_searchBuffer;   //m_outputBuffers;
    cudaStream_t            m_stream;
    cudaEvent_t             m_readyEvent;
};

struct CudaMinerValues
{
    CudaMinerValues():m_globalIndex(0),
                    m_deviceID(0),
                    m_blockSize(0),
                    m_gridSize(0),
                    m_streamCount(0),
                    m_globalWorkSize(0),
                    m_outputBufferSize(0){}

    uint32_t m_globalIndex;
    uint32_t m_deviceID;
    uint32_t m_blockSize;
    uint32_t m_gridSize;
    uint32_t m_streamCount;
    uint32_t m_globalWorkSize;      //Thread count
    uint32_t m_outputBufferSize;
    bool     m_isBenchmarkMode;
};


struct CudaWorkPackage: public CudaMinerValues
{
    CudaWorkPackage():m_startNonce(0), m_param32_1(0), m_headerSize(0){}

    bool IsSameHeader(const CudaWorkPackage& w) { return memcmp(w.m_header, m_header, m_headerSize) == 0;}
    
    //NOTE: header of size zero DO NOT RESET THE CURRENT HEADER CONTENT !!!! 
    void SetHeader(const uint8_t* p, unsigned size) 
    { 
        m_headerSize = RH_Min((size_t)size, sizeof(m_header)); 
        RHMINER_ASSERT(m_headerSize <= sizeof(m_header));
        memcpy(m_header, p, m_headerSize); 
    }
    
    CudaWorkPackage& operator=(const CudaWorkPackage&w);

    uint64_t     m_startNonce;       //'work offset' for all nonce32 based kernel !!!
    uint32_t     m_param32_1;        //Nonce2 for RandomHash
    uint32_t     m_headerSize;
    uint8_t      m_header[256];      
};

class RandomHashCUDAMiner
{
public:
    RandomHashCUDAMiner();
    virtual ~RandomHashCUDAMiner();
    virtual bool Init(const CudaWorkPackage& work, const CudaMinerValues& initData);
    virtual void SetTarget(uint64_t target);
    virtual void SetthreadPerHash(uint32_t tph) {};
    virtual void SetStartNonce(uint64_t baseNonce);

    virtual void QueueKernel();
    virtual void ClearKernelOutputBuffer();
    virtual void PrepareWork(const CudaWorkPackage& work);
    virtual U32  EvalKernelResult(uint32_t* outFoudnNonces, uint32_t maxCount, uint64_t& outBaseNonce);

    uint64_t        m_target;
    CudaWorkPackage m_currentWork;
    uint32_t        m_kernelItterations;

protected:
    uint32_t        m_nextGridSize;
    uint32_t        m_nextBlockSize;
    SynchroBuffer*  m_outputBuffer;

    uint64_t m_lastPushTime;
    uint32_t m_globalIndex;
    uint32_t m_deviceID;
    uint32_t m_blockSize;
    uint32_t m_gridSize;
    uint32_t m_streamCount;
    uint32_t m_globalWorkSize;      //Thread count
    uint32_t m_outputBufferSize;
};

////////////////////////////////////////////////
//
//  Cuda miner allocations
extern void DestroyCudaMiner(RandomHashCUDAMiner* miner);
extern RandomHashCUDAMiner* CreateCudaMiner_PASCALV4();
