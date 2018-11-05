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
#include "CLMinerBase.h"
#include "MinersLib/Global.h"

#ifndef RH_COMPILE_CPU_ONLY
#include "cuda_runtime.h"
#endif

std::mutex*  gs_sequentialBuildMutex = new std::mutex;
void CLMinerBase::addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	snprintf(buf, sizeof(buf),"#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

CLMinerBase::CLMinerBase(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex):
	Miner(FormatString("%s", GpuManager::Gpus[gpuIndex].gpuName.c_str()), _farm, globalWorkMult, localWorkSize, gpuIndex)
{
    m_deviceOptions = " ";
}

CLMinerBase::~CLMinerBase()
{
}

void CLMinerBase::InitFromFarm(U32 i) 
{ 
    Miner::InitFromFarm(i);
    m_gpuInfoCache->relativeIndex = i; 
}

void CLMinerBase::SetOfflineCompileOptions(const string& deviceOptions, const string& definesCombinations)
{
    m_deviceOptions = deviceOptions;
    m_definesCombination = definesCombinations;
}

void CLMinerBase::UpdateWorkSize(U32 absoluteVal)
{
    bool forcedGWS = false;
    if (!absoluteVal)
    {
        if (m_gpuInfoCache->setGpuThreadCount != U32_Max)
        {
            m_globalWorkSize = m_gpuInfoCache->setGpuThreadCount;
            if (m_globalWorkSize < m_gpuInfoCache->maxGroupSize)
            {
                if (m_gpuInfoCache->localWorkSize >= m_globalWorkSize)
                    m_localWorkSize = 0; //calculated later in this func
                else
                    m_localWorkSize = 1;
            }
            else
            {
                if (m_gpuInfoCache->localWorkSize)
                    m_localWorkSize = m_gpuInfoCache->localWorkSize;
                else
                    m_localWorkSize = m_globalWorkSize / m_gpuInfoCache->maxGroupSize;
                
                if (m_gpuInfoCache->isOpenCL)
                    m_globalWorkSize = m_gpuInfoCache->maxGroupSize;
                else
                    m_globalWorkSize = RHMINER_FLOOR(m_globalWorkSize, 8); ////round GWS to 8
            }

            if (!m_gpuInfoCache->localWorkSize)
                m_gpuInfoCache->localWorkSize = m_localWorkSize;
            m_gpuInfoCache->setGpuThreadCount = 0;
            forcedGWS = true;
        }
        else
        {
            m_globalWorkSize = 64;
        }
    }
    else
    {
        // make sure that global work size is evenly divisible by the local workgroup size
        if (m_gpuInfoCache->isOpenCL)
            m_globalWorkSize = m_localWorkSize + absoluteVal;
        else
            m_globalWorkSize = absoluteVal;
    }

    if (m_localWorkSize == 0)
    {
        RHMINER_ASSERT(m_gpuInfoCache->maxWorkSize != 0);

        U32 localSizePreset = m_gpuInfoCache->localWorkSize;
        if (localSizePreset)
            m_localWorkSize = localSizePreset;
        else
        {
            if (m_gpuInfoCache->isOpenCL)
                m_localWorkSize = m_gpuInfoCache->maxWorkSize;
            else
            {
                //special case, we set the preset here for later use... only for nvidia
                m_localWorkSize = 1;
                m_gpuInfoCache->localWorkSize = m_localWorkSize;
            }
        }
    }

    if (forcedGWS == false)
    {
        if (GpuManager::Gpus[m_globalIndex].gpuType != GpuType_CPU)
        {
            if ((m_globalWorkSize % m_localWorkSize) != 0)
                m_globalWorkSize = ((m_globalWorkSize / m_localWorkSize) + 1) * m_localWorkSize;
        }
    }

    m_gpuInfoCache->globalWorkSize = m_globalWorkSize;
}


void  CLMinerBase::FreeQueuedBuffers()
{ 
    m_queuedBuffers.clear(); 
}

bytes* CLMinerBase::AllocQueuedBuffer(size_t size)  
{ 
    m_queuedBuffers.push_back(std::auto_ptr<bytes>(new bytes(size))); 
    return m_queuedBuffers.back().get(); 
}


bool CLMinerBase::init(const PascalWorkSptr& work)
{
    AddPreBuildFunctor([&](string& code) 
    {
        addDefinition(code, "PLATFORM", m_platformType);
        addDefinition(code, "COMPUTE_UNITS", m_deviceMaxComputeUnit);
    });

    try
    {
        std::lock_guard<std::mutex> g(*gs_sequentialBuildMutex);
        
        // get GPU device of the default platform
        const auto& devInfo = GpuManager::Gpus[m_globalIndex];

#ifndef RH_COMPILE_CPU_ONLY
        m_platformType = OPENCL_PLATFORM_UNKNOWN;
        if (devInfo.platformName.find("NVIDIA ") != std::string::npos)
        {
            m_platformType = OPENCL_PLATFORM_NVIDIA;
			m_nvmlh = wrap_nvml_create();

            cudaSetDevice(m_globalIndex);
        }
        else if (devInfo.platformName.find("AMD ") != std::string::npos)
        {
            m_platformType = OPENCL_PLATFORM_AMD;
            m_adlh = wrap_adl_create();
        }
        
        cl::Device device = GpuManager::GetDeviceFromGPUMap(devInfo);
        m_deviceOptions = "";

        clGetDeviceInfo(device(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &m_deviceMaxComputeUnit, NULL);
        clGetDeviceInfo(device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &m_deviceMaxWorkgroupSize, NULL);
        m_deviceMaxWorkItemSize = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        clGetDeviceInfo(device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &m_deviceMaxMemSize, NULL);
        clGetDeviceInfo(device(), CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &m_MemCacheLineSize, NULL);
        clGetDeviceInfo(device(), CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &m_MemCacheSize, NULL);
        clGetDeviceInfo(device(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &m_LocalMemCacheSize, NULL);

        // create context
        m_context = cl::Context(vector<cl::Device>(&device, &device + 1));
        m_queue = cl::CommandQueue(m_context, device);
#endif //RH_COMPILE_CPU_ONLY

        //set this thread at high prio
        RH_SetThreadPriority(RH_ThreadPrio_High);

        m_isInitialized = true;
    }
    catch (cl::Error const& err)
    {
        RHMINER_PRINT_EXCEPTION_EX("CL Exception", err.what());
        return false;
    }
    catch (...)
    {
        PrintOut("Unknown Exception");
    }

    return BuildKernels(work);
}

//default : wcheck the init flag to start init or not
bool CLMinerBase::ShouldInitialize(const PascalWorkSptr& work)
{
    return !m_isInitialized;
}

void CLMinerBase::ClearKernels()
{
    m_kernels.clear();
    m_lastCodeCRC.clear();
}


void ConbineAll(const vector<vector<string> > &allVecs, size_t vecIndex, string strSoFar, strings& cmdline)
{
    if (vecIndex >= allVecs.size())
    {
        cmdline.push_back(strSoFar);
        return;
    }
    for (size_t i=0; i<allVecs[vecIndex].size(); i++)
        ConbineAll(allVecs, vecIndex+1, strSoFar+allVecs[vecIndex][i], cmdline);
}


bool CLMinerBase::BuildKernels(const PascalWorkSptr& work)
{
#ifndef RH_COMPILE_CPU_ONLY
    //build programs
    KernelCodeAndFuctions codes = GetKernelsCodeAndFunctions();

    m_kernels.resize(codes.size());
    m_lastCodeCRC.resize(codes.size());
    for (int i = 0; i < codes.size(); ++i)
    {
        bytes& codeBytes = codes[i].first;
        string code = (char*)&codeBytes[0];
        strings& functions = codes[i].second;

        for(auto prebuild : m_OnPrebuildKernel)
            prebuild(code);

        string bi;
        unsigned codeCrc = RH_crc32(code.c_str());
        if (codeCrc != m_lastCodeCRC[i])
        {
            //support sequaltial init
            gs_sequentialBuildMutex->lock();
            
            PrintOut("GPU%d Building kernels %s on %s\n", m_globalIndex, functions[0].c_str(), m_gpuInfoCache->archName.c_str());

            // create miner OpenCL program
            cl::Program::Sources sources{ { code.data(), code.size() } };
            cl::Program program(m_context, sources);
            cl::Device device;

            try
            {
                m_lastCodeCRC[i] = codeCrc;
                device = GpuManager::GetDeviceFromGlobalIndex(m_globalIndex);

                if (!device())
                {
                    PrintOut("No OpenCL devices found.\n");
                    return false;
                }
                string vendorCompilerOpt;
                if (m_platformType == OPENCL_PLATFORM_AMD)
                {
                    vendorCompilerOpt = " -fno-bin-source -fno-bin-llvmir -fno-bin-amdil ";
                }

                   
                    string compileOptions = m_deviceOptions + vendorCompilerOpt;
                    //remove all debug stuff from assemblies
                    program.build({ device }, compileOptions.c_str());

                    bi = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

                    for (auto& funcName : functions)
                    {
                        cl::Kernel k = cl::Kernel(program, funcName.c_str());
                        if (!k())
                        {
                            PrintOut("Build info: Cannot make kernel %s\n", funcName.c_str());
                            throw cl::Error(-1, "Bad Kernel");
                        }
                            
                        m_kernels[i].push_back( pair<string, cl::Kernel>(funcName, k));
                    }        
            }

            catch (cl::Error const& _e)
            {
                bi = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                if (bi.length())
                {
                    const char* err = FormatString("Build (Error): %s, excep : %s\n", bi.c_str(), _e.what());
                    PrintOut("%s", err);
                }
                else
                {
                    RHMINER_PRINT_EXCEPTION_EX("OpenCL Error.", _e.what());
                }

                gs_sequentialBuildMutex->unlock();

                return false;
            }


            gs_sequentialBuildMutex->unlock();

        }
    }
#endif //RH_COMPILE_CPU_ONLY
    return true;
}

cl::Kernel  CLMinerBase::GetKernel(U32 codeID, const char* name)
{
    for (auto& k : m_kernels[codeID])
        if (k.first == name)
            return k.second;
    RHMINER_EXIT_APP("Kernel does not exists");
    return cl::Kernel();
}
        
void  CLMinerBase::RemoveKernel(U32 codeID, const char* name)
{
    for (auto it = m_kernels[codeID].begin(); it != m_kernels[codeID].end(); ++it)
        if (it->first == name)
        {
            m_kernels[codeID].erase(it);
            break;
        }
}
