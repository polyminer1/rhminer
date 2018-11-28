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
#include "GenericCLMiner.h"
#include "MinersLib/Global.h"
#include "corelib/PascalWork.h"

U64 c_zero = 0;

GenericCLMiner::GenericCLMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex) :
    CLMinerBase(_farm, globalWorkMult, localWorkSize, gpuIndex)
{
    string tn = FormatString("GPU%d", m_globalIndex);
}

GenericCLMiner::~GenericCLMiner()
{
    
}


bool GenericCLMiner::WorkLoopStep()
{
    //wait for work
    m_workReadyEvent.WaitUntilDone();
    RHMINER_RETURN_ON_EXIT_FLAG_EX(true); 

    PascalWorkSptr workTemplate = GetWork();
    
    //when kill() is called, we set the event but not the work, just exit then.
    if (!workTemplate.get())
        return false;

    //handle DevFee
    m_farm.ReconnectToServer(0xFFFFFFFF); 

    if (!m_gpuInfoCache->enabled)
    {
        CpuSleep(100);
        return true;
    }

    if (ShouldInitialize(workTemplate))
    {
        AutoFlagSet< std::atomic<bool> > Flag(m_isInitializing);

        m_isInitializationDone = false;
        //init a new WorkOffset
        m_workOffset = KernelOffsetManager::GetCurrentValue();
        bool res  = init(workTemplate);
        m_isInitializationDone = true;
        if (!m_isInitialized || !res)
        {
            PrintOut("Init thread failed\n");
            m_gpuInfoCache->enabled = false;
            QueueStopWorker();
            return false; 
        }
    }

    if (m_lastWorkTemplate.get() == nullptr || !m_lastWorkTemplate->IsSame(workTemplate.get()))
    {
        //new work template, we shoud PrepareWork it
        m_lastWorkTemplate = workTemplate;
    }
    else 
    {
        //nothing new, continue mining the current wp
        workTemplate = m_currentWp;
    }

    PrepareWorkStatus prepStatus = PrepareWork_Nothing;
    prepStatus = PrepareWork(workTemplate);

    //eval before running kernel so we can do some calculations while the kernel is running
    EvalKernelResult();

    if (prepStatus != PrepareWork_WaitForNewWork &&
        prepStatus != PrepareWork_Timeout)
    {
        RHMINER_RETURN_ON_EXIT_FLAG_EX(true);
        QueueKernel();

        AddHashCount(m_globalWorkSize);
    }

    //passvely wait for new job
    if (prepStatus == PrepareWork_WaitForNewWork  ||
        prepStatus == PrepareWork_Timeout)
    {
        CpuSleep(100);
    }

    // Check if we should stop.
    if (shouldStop())
    {
        // Make sure the last buffer write has finished --
        // it reads local variable.
        if (GetPlatformType() == PlatformType_OpenCL)
            RHMINER_CL_EXEC(m_queue.finish())
        else
            CpuSleep(250);
        return false; 
    }
    
    return true; 
}


bool GenericCLMiner::init(const PascalWorkSptr& work)
{
    AddPreBuildFunctor([&](string& code) 
    {
        //use AddPreBuildFunctor as pre-init !
        m_zeroBuffer.resize(GetOutputBufferSize());
        ZeroVector(m_zeroBuffer);

        m_results.resize(GetOutputMaxCount() + 1);
        ZeroVector(m_results);

        addDefinition(code, "GROUP_SIZE", (U32)m_localWorkSize);
        addDefinition(code, "MAX_OUTPUTS", GetOutputMaxCount());
    });

    // get all platforms
    try
    {
        //will call BuildKernels and fill m_kernels
        if (!CLMinerBase::init(work))
        {
            PrintOut(" Init failed\n");
            return false;
        }

#ifndef RH_COMPILE_CPU_ONLY
        // create buffers           
        U32 outSize = GetOutputBufferSize();
        U32 headSize = GetHeaderBufferSize();
        m_kernelOutput = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, outSize);
        m_kernelHeader = cl::Buffer(m_context, CL_MEM_READ_ONLY, headSize);
#endif
        //start hashrate counting
        if (m_hashCountTime == 0)
            m_hashCountTime = TimeGetMilliSec();
    }
    catch (cl::Error const& err)
    {
        RHMINER_PRINT_EXCEPTION_EX("CL Exception ",  err.what());
        return false;
    }
    return true;
}


void GenericCLMiner::KernelCallBack()
{
    
}

void GenericCLMiner::QueueKernel()
{
    m_kernelItterations++;

    {
        for(auto& kernels : m_kernels )
        {
            for(U32 i = 0; i < kernels.size(); i++)
            {
                auto& kernel = kernels[i];
                RHMINER_CL_EXEC( m_queue.enqueueNDRangeKernel(kernel.second, m_workOffset, m_globalWorkSize, m_localWorkSize, NULL, NULL/*&CBevent*/) );
            }
        } 

        m_workOffset = KernelOffsetManager::Increment(m_globalWorkSize) - m_globalWorkSize;
    }
}


bool GenericCLMiner::IsWorkStalled()
{
    if (m_LastNewWorkStartTime == 0)
        return false;

    //Verify global timeout
    S64 dt = (TimeGetMicroSec() - m_LastNewWorkStartTime )/1000/1000;
    if (dt > 0)
    {
        if (dt > MaxWorkPackageTimeout)
        {        
            PrintOut("No work for %u seconds. Reconnecing...", MaxWorkPackageTimeout);
            return true;
        }
    }
    else
    {
        //handle time change !
        m_LastNewWorkStartTime = TimeGetMicroSec();
    }

    return false;
}

PrepareWorkStatus GenericCLMiner::PrepareWork(const PascalWorkSptr& newWorkTempl, bool reuseCurrentWP)
{
    U32 isWorkpackageDirty = AtomicSet(m_workpackageDirty, 0);

    PrepareWorkStatus workStatus = PrepareWork_Nothing;
    if (m_currentWp.get() == nullptr || !m_currentWp->IsSame(newWorkTempl.get()) || reuseCurrentWP || isWorkpackageDirty)
    {
        //clone the global work package
        if (!reuseCurrentWP)
        {
            m_currentWp = PascalWorkSptr(newWorkTempl->Clone());
            m_currentWp->m_localyGenerated = true;
        }

        m_farm.RequestNewWork(m_currentWp, this);

        m_startNonce = m_currentWp->m_startNonce;
        if (m_startNonce)
            KernelOffsetManager::Reset(m_startNonce);
        
        if (GetPlatformType() != PlatformType_CPU)
        {
            m_workOffset = KernelOffsetManager::Increment(m_globalWorkSize) - m_globalWorkSize;
        }

        m_sleepWhenWorkFinished = false;
        m_lastWorkStartTimeMs = TimeGetMilliSec();

        //indicate first kernel push
        m_kernelItterations = 0;

        ClearKernelOutputBuffer();

        if (m_LastNewWorkStartTime == 0)
        {
            m_LastNewWorkStartTime = TimeGetMicroSec();
        }
        else
        {
            auto delta = TimeGetMicroSec() - m_LastNewWorkStartTime;
            m_accumNewWorkDeltaTime += delta;
            m_accumNewWorkDeltaTimeCount++;

            m_LastNewWorkStartTime = TimeGetMicroSec();
        }
        
        workStatus = PrepareWork_NewWork;
    }
    else
    {       
        // Detect stale work
        if (IsWorkStalled())
        {
            auto avgWorkTimeDelta = m_accumNewWorkDeltaTime / m_accumNewWorkDeltaTimeCount / 1000 / 1000; //sec
            PrintOut("%s timeout after %u seconds. Work is stale.", GpuManager::Gpus[m_globalIndex].gpuName.c_str(), m_maxExtraTimePerWorkPackage + avgWorkTimeDelta);
            
            //reset this warning
            m_LastNewWorkStartTime = TimeGetMicroSec();
            m_workReadyEvent.Reset();
            workStatus = PrepareWork_Timeout;

            //request reconnect
            m_farm.ReconnectToServer(m_globalIndex);
        }
    }

    return workStatus;
}

void GenericCLMiner::EvalKernelResult()
{
    //skip the first run for the kernel is not even started yet
    if (m_kernelItterations)
    {
        RHMINER_ASSERT(m_results.size() == GetOutputMaxCount()+1);
        m_results[0] = 0;
        
        RHMINER_CL_EXEC( m_queue.enqueueReadBuffer(m_kernelOutput, CL_TRUE, 0,(sizeof(U32)*m_results.size()), &m_results[0]) );
       
        //flush pending write buffers cuz they are executed with this cl call
        FreeQueuedBuffers();

        U32 count = (U32)m_results[0];
        if (count)
        {
            if (count > GetOutputMaxCount()+1 || count > m_results.size())
            {
                PrintOut("Error. To many nonces found...\n");
                count = RH_Min(GetOutputMaxCount()+1, (U32)m_results.size()-1); 
            }
            ClearKernelOutputBuffer();
           
            std::vector<U64> nonces;
            nonces.reserve(count);
            for (U32 i = 0; i < count; i++) 
                nonces.push_back((U64)m_results[i + 1]);

            SolutionSptr solPtr = MakeSubmitSolution(nonces, false);

            m_farm.submitProof(solPtr);
        }
    }
}

// Reset search buffer if any solution found.
void GenericCLMiner::ClearKernelOutputBuffer()
{
    RHMINER_CL_EXEC(m_queue.enqueueWriteBuffer(m_kernelOutput, CL_FALSE, 0, m_zeroBuffer.size(), &m_zeroBuffer[0]));
}

void GenericCLMiner::SetSearchKernelCurrentTarget(U32 paramIndex, cl::Kernel& searchKernel)
{
    cl_long upperTarget = m_currentWp->GetDeviceTargetUpperBits();

    RHMINER_CL_EXEC(searchKernel.setArg(paramIndex, upperTarget));
}

KernelCodeAndFuctions GenericCLMiner::GetKernelsCodeAndFunctions()
{
    bytes code;
    return { { code, { ""}} };
};


SolutionSptr GenericCLMiner::MakeSubmitSolution(const std::vector<U64>& nonces, bool isFromCpuMiner)
{
    PascalSolution* sol = new PascalSolution();
    sol->m_results = nonces;
    sol->m_gpuIndex = m_globalIndex;
    sol->m_work = PascalWorkSptr(m_currentWp->Clone());
    sol->m_isFromCpuMiner = isFromCpuMiner;

    return SolutionSptr(sol);
}