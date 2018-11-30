/**
 * Gpu manager source code
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
#include "MinersLib/GpuManager.h"
#include "MinersLib/Global.h"
#include "MinersLib/Global.h"
#include "BuildInfo.h"

#ifndef RH_COMPILE_CPU_ONLY
#include "cuda_runtime.h"
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifndef _WIN32_WINNT
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#endif

#define RHMINER_MAKE_ARCH_NAME(devname, gpuToken, archname, var) if (stristr(devname.c_str(), gpuToken)) {var = string(gpuToken) + "_" + archname;}

std::vector<GpuManager::GPUInfos>  GpuManager::Gpus;
CPUInfo                            GpuManager::CpuInfos;
bool                               g_isSSE41Supported = false;

GpuManager::GpuManager()
{
}

cl::Device GpuManager::GetDeviceFromGlobalIndex(U32 globalIndex)
{
    if (globalIndex < GpuManager::Gpus.size())
        return GetDeviceFromGPUMap(GpuManager::Gpus[globalIndex]);
    else
        return cl::Device();
}


cl::Device GpuManager::GetDeviceFromGPUMap(const GPUInfos& gpuInfo)
{
    unsigned deviceId = INT_MAX;
    vector<cl::Device> devices;
    try
    {
        vector<cl::Platform> platforms = GetPlatforms();
        if (platforms.empty())
            return cl::Device();
        
        if (gpuInfo.platformID > platforms.size())
            return cl::Device();

        devices = GetDevices(platforms, gpuInfo.platformID);
        if (devices.empty() || gpuInfo.deviceID > devices.size())
            return cl::Device();

        // use selected device
        deviceId = gpuInfo.deviceID;
    }
    catch (...)
    {
    }

    if (deviceId != INT_MAX && devices.size())
        return devices[deviceId];

    return cl::Device();
}

std::vector<cl::Platform> GpuManager::GetPlatforms()
{
	vector<cl::Platform> platforms;
	try
	{
#ifndef RH_COMPILE_CPU_ONLY
		cl::Platform::get(&platforms);
#endif 
	}
	catch(...)
	{
	}
	return platforms;
}

std::vector<cl::Device> GpuManager::GetDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	try
	{
#ifndef RH_COMPILE_CPU_ONLY
    	size_t platform_num = RH_Min<size_t>(_platformId, _platforms.size() - 1);
		_platforms[platform_num].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,&devices);
#endif
	}
	catch (...)
	{
	}
	return devices;
}

#ifndef RH_COMPILE_CPU_ONLY
string GetCudaDeviceProp(int di)
{
    string str;
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, di);
    if (err == cudaSuccess)
    {
        std::stringstream strout;
        #define NV_PO_LIST_DEVICE(val) strout << "  " << #val << " : " << props.val << endl;
        NV_PO_LIST_DEVICE(totalGlobalMem);             /**< Global memory available on device in bytes */
        NV_PO_LIST_DEVICE(sharedMemPerBlock);          /**< Shared memory available per block in bytes */
        NV_PO_LIST_DEVICE(warpSize);                   /**< Warp size in threads */
        NV_PO_LIST_DEVICE(maxThreadsPerBlock);         /**< Maximum number of threads per block */
        NV_PO_LIST_DEVICE(maxThreadsDim[0]);           /**< Maximum size of each dimension of a block */
        NV_PO_LIST_DEVICE(maxThreadsDim[1]);           /**< Maximum size of each dimension of a block */
        NV_PO_LIST_DEVICE(maxThreadsDim[2]);           /**< Maximum size of each dimension of a block */
        NV_PO_LIST_DEVICE(maxGridSize[0]);             /**< Maximum size of each dimension of a grid */
        NV_PO_LIST_DEVICE(maxGridSize[1]);             /**< Maximum size of each dimension of a grid */
        NV_PO_LIST_DEVICE(maxGridSize[2]);             /**< Maximum size of each dimension of a grid */
        NV_PO_LIST_DEVICE(clockRate);                  /**< Clock frequency in kilohertz */
        NV_PO_LIST_DEVICE(totalConstMem);              /**< Constant memory available on device in bytes */
        NV_PO_LIST_DEVICE(major);                      /**< Major compute capability */
        NV_PO_LIST_DEVICE(minor);                      /**< Minor compute capability */
        NV_PO_LIST_DEVICE(pciBusID);                   /**< PCI bus ID of the device */
        NV_PO_LIST_DEVICE(pciDeviceID);                /**< PCI device ID of the device */
        NV_PO_LIST_DEVICE(pciDomainID);                /**< PCI domain ID of the device */
        NV_PO_LIST_DEVICE(deviceOverlap);              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
        NV_PO_LIST_DEVICE(multiProcessorCount);        /**< Number of multiprocessors on device */
        NV_PO_LIST_DEVICE(kernelExecTimeoutEnabled);   /**< Specified whether there is a run time limit on kernels */
        NV_PO_LIST_DEVICE(canMapHostMemory);           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
        NV_PO_LIST_DEVICE(computeMode);                /**< Compute mode (See ::cudaComputeMode) */
        NV_PO_LIST_DEVICE(concurrentKernels);          /**< Device can possibly execute multiple kernels concurrently */
        NV_PO_LIST_DEVICE(concurrentKernels);          /**< Device can possibly execute multiple kernels concurrently */
        NV_PO_LIST_DEVICE(unifiedAddressing);          /**< Device shares a unified address space with the host */
        NV_PO_LIST_DEVICE(memoryClockRate);            /**< Peak memory clock frequency in kilohertz */
        NV_PO_LIST_DEVICE(memoryBusWidth);             /**< Global memory bus width in bits */
        NV_PO_LIST_DEVICE(l2CacheSize);                /**< Size of L2 cache in bytes */
        NV_PO_LIST_DEVICE(maxThreadsPerMultiProcessor);/**< Maximum resident threads per multiprocessor */
        NV_PO_LIST_DEVICE(streamPrioritiesSupported);  /**< Device supports stream priorities */
        NV_PO_LIST_DEVICE(globalL1CacheSupported);     /**< Device supports caching globals in L1 */
        NV_PO_LIST_DEVICE(localL1CacheSupported);      /**< Device supports caching locals in L1 */
        NV_PO_LIST_DEVICE(sharedMemPerMultiprocessor); /**< Shared memory available per multiprocessor in bytes */
        NV_PO_LIST_DEVICE(managedMemory);              /**< Device supports allocating managed memory on this system */
        NV_PO_LIST_DEVICE(pageableMemoryAccess);       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
        NV_PO_LIST_DEVICE(concurrentManagedAccess);    /**< Device can coherently access managed memory concurrently with the CPU */
        
        str = FormatString("Device name %s\n", props.name);
        str += strout.str();
    }
    return str;
}
#endif //RH_COMPILE_CPU_ONLY

void GpuManager::listDevices()
{
#ifndef RH_COMPILE_CPU_ONLY
    printf("\nListing OpenCL/CUDA devices.\n");
    unsigned int i = 0;
    
    vector<cl::Platform> platforms = GetPlatforms();
    for (unsigned j = 0; j < platforms.size(); ++j)
    {
        string pname = platforms[j].getInfo<CL_PLATFORM_NAME>().c_str();
        if (stristr(pname.c_str(), "experimental"))
            continue;

        printf("Platform : #%d %s (%s %s %s)\n",i, pname.c_str(), 
            platforms[j].getInfo<CL_PLATFORM_VERSION>().c_str(),
            platforms[j].getInfo<CL_PLATFORM_VENDOR>().c_str(),
            platforms[j].getInfo<CL_PLATFORM_EXTENSIONS>().c_str());

        vector<cl::Device> devices = GetDevices(platforms, j);
        for (U32 di = 0; di < devices.size(); di++)
        {
            auto& device = devices[di];
            if (device.getInfo<CL_DEVICE_AVAILABLE>())
            {
                const char* dts = "";
                switch (device.getInfo<CL_DEVICE_TYPE>())
                {
                case CL_DEVICE_TYPE_CPU:
                    dts = "CPU";
                    break;
                case CL_DEVICE_TYPE_GPU:
                    dts = "GPU";
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    dts = "ACCELERATOR";
                    break;
                }
                printf(" Device #%d %s %s\n", di, device.getInfo<CL_DEVICE_NAME>().c_str(), dts);
              
                std::vector<size_t> sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
                printf("  CL_DEVICE_MAX_COMPUTE_UNITS %s \n",  to_string(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()).c_str());
                printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()).c_str());
                printf("  CL_DEVICE_MAX_WORK_ITEM_SIZE: %s \n",  toStringVect(sizes).c_str());
                printf("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: %s \n",  to_string(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>()).c_str());
                printf("  CL_DEVICE_ADDRESS_BITS: %s \n",  to_string(device.getInfo<CL_DEVICE_ADDRESS_BITS>()).c_str());
                printf("  CL_DEVICE_MEM_BASE_ADDR_ALIGN: %s \n",  to_string(device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>()).c_str());
                printf("  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE>()).c_str());
                printf("  CL_DEVICE_GLOBAL_MEM_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()).c_str());
                printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()).c_str());
                printf("  CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>()).c_str());
                printf("  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()).c_str());
                printf("  CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: %s \n",  to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>()).c_str());
                printf("  CL_DEVICE_LOCAL_MEM_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()).c_str());
                printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %s \n",  to_string(device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()).c_str());
                printf("  CL_DEVICE_MAX_CONSTANT_ARGS: %s \n",  to_string(device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>()).c_str());
                

                if (pname.find("NVIDIA") != string::npos)
                {
                    string strout = GetCudaDeviceProp(di);
                    printf("  %s", strout.c_str());
	            }  
            }
            else
                printf("Device OFF");

            ++i;
        }
    }
    
    //pc without AMD devices on it !?
    if (platforms.size() == 0)
    {
        int nDevices=0;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) 
        {
            printf("CUDA Device Number: #%d\n", i);
            string strout = GetCudaDeviceProp(i);
            printf("  %s", strout.c_str());
        }    
    }
#endif //RH_COMPILE_CPU_ONLY
    
    //list cpu
    for (U32 i = 0; i < GpuManager::Gpus.size(); ++i)
    {
        if (GpuManager::Gpus[i].gpuType == GpuType_CPU)
            printf("CPU : %s with %d logical cores\n", GpuManager::Gpus[i].description.c_str(), GpuManager::Gpus[i].maxCU);
    }
   
    exit(0);
}


std::vector<unsigned> GpuManager::GetEnabledGPUIndices()
{
    std::vector<unsigned> enabledGPU;
    for (auto& g : GpuManager::Gpus)
        if (g.enabled)
            enabledGPU.push_back(g.globalIndex);

    return enabledGPU;
}

U32 GpuManager::GetEnabledGPUCount()
{
    std::vector<unsigned> enabledGPU;
    U32 cnt = 0;
    for (auto& g : GpuManager::Gpus)
    {
        if (g.enabled)
            cnt++;
    }

    return cnt;
}

void GpuManager::LoadGPUMap() 
{
    std::map<string, GPUInfos> gpuIndent;

    LoadCPUInfos();  
    
#ifndef RH_COMPILE_CPU_ONLY
	vector<cl::Platform> platforms = GetPlatforms();
	for (unsigned j = 0; j < platforms.size(); ++j)
	{
        string pname = platforms[j].getInfo<CL_PLATFORM_NAME>().c_str();
        if (stristr(pname.c_str(), "experimental"))
            continue;
        
        string deviceVersion = platforms[j].getInfo<CL_PLATFORM_VERSION>().c_str();
        string deviceExt = platforms[j].getInfo<CL_PLATFORM_EXTENSIONS>().c_str();
        string platformIdent = pname;
        platformIdent += FormatString("(%s %s %s)",
            deviceVersion.c_str(),
            platforms[j].getInfo<CL_PLATFORM_VENDOR>().c_str(),
            deviceExt.c_str());
            
        bool isCPU = false;
        string deviceIdent;
		vector<cl::Device> devices = GetDevices(platforms, j);
		for (U32 di=0; di < devices.size(); di++)
		{
            auto& device = devices[di];
            string deviceName = device.getInfo<CL_DEVICE_NAME>();
            if (device.getInfo<CL_DEVICE_AVAILABLE>())
            {
                deviceIdent = deviceName;
                switch (device.getInfo<CL_DEVICE_TYPE>())
                {
                case CL_DEVICE_TYPE_CPU:
                    deviceIdent += "CPU;";
                    isCPU = true;
                    break;
                case CL_DEVICE_TYPE_GPU:
                    deviceIdent+= "GPU;";
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    deviceIdent+= "ACCELERATOR;";
                    break;
                }
                
                U64 memsize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                U64 maxAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                U32 maxCU = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                U64 maxGS = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
                std::vector<size_t> maxWS = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
                deviceIdent+= to_string(di);
                deviceIdent += ";";
                deviceIdent+= to_string(maxCU);
                deviceIdent += ";";
                deviceIdent+= to_string(memsize);
                deviceIdent += ";";
                deviceIdent+= to_string(maxAllocSize);
                deviceIdent += ";";
                deviceIdent+= to_string(maxGS);
                deviceIdent += ";";
                deviceIdent+= toStringVect(maxWS);

                string simpleDescr;
                if (pname.find("NVIDIA") != string::npos)
                    simpleDescr = deviceName;
                else
                {
                    string mbStr = FormatString("%.1f", device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 / 1024 / 1024.0f);
                    ReplaceString(mbStr, ".0", "");
                    if (pname.find("AMD") != string::npos)
                        simpleDescr = "AMD ";
                    simpleDescr += deviceName;
                    simpleDescr += " ";
                    simpleDescr += mbStr;
                    simpleDescr += "GB";
                }

                string pid = platformIdent + deviceIdent;
                bool found = false;
                //find a idnetical platform+device on another platform
                for (auto& i : gpuIndent)
                {
                    if (i.first == pid)
                    {
                        found = true;
                        break;
                    }
                }

                //detect architecture name
                string archName = "IntelGPU";
                //make a simpel device name
                if (pname.find("NVIDIA") != string::npos)
                {
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 10", "GTX1000", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 9", "GTX900", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 8", "GTX800", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 7", "GTX700", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 6", "GTX600", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 5", "GTX500", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"GTX 4", "GTX400", archName);
                }
                else if (pname.find("AMD") != string::npos)
                {
                    RHMINER_MAKE_ARCH_NAME(deviceName,"gfx901",   "GCN5", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"gfx900",   "GCN5", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Ellesmere","GCN4", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Baffin",   "GCN4", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Tonga",    "GNC3", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Fiji",     "GNC3", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"hawaii",   "GNC2", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Bonaire",  "GNC2", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Oland",    "GNC1", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Cape Verde","GNC1", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Pitcairn", "GNC1", archName);
                    RHMINER_MAKE_ARCH_NAME(deviceName,"Tahiti",   "GNC1", archName);
                }
                
                if (!found)
                {
                    GPUInfos inf;
                    inf.platformID = j;
                    inf.deviceID = di;
                    inf.description = simpleDescr;
                    inf.deviceName = deviceName;
                    inf.platformName = pname;
                    inf.archName = archName;
                    inf.deviceVersion = deviceVersion;
                    inf.deviceExtention = deviceExt;
                    inf.memorySize = memsize;
                    inf.localMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                    inf.maxAllocSize = maxAllocSize;
                    inf.maxCU = maxCU;
                    inf.maxWorkSize = (U32)maxWS[0];
                    inf.maxGroupSize = maxGS;
                    if (isCPU)
                        inf.gpuType = GpuType_CPU;
                    gpuIndent[pid] = inf;
                }
            }
		}
	}

    //pc without AMD cards
    if (!platforms.size())
    {
        int nDevices=0;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) 
        {
            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, i);
            if (err == cudaSuccess)
            {
                string pid = FormatString("NVIDIA%d", i);
                GPUInfos inf;
                inf.platformID = 0;
                inf.deviceID = i;
                inf.description = props.name;
                inf.deviceName = props.name;
                inf.platformName = "NVIDIA";
                inf.archName = "asd";
                inf.deviceVersion = FormatString("%d:%d", props.major, props.minor);
                inf.deviceExtention = "";
                inf.memorySize = props.totalGlobalMem;
                inf.localMemSize = props.sharedMemPerBlock;
                inf.maxAllocSize = props.totalGlobalMem;
                inf.maxCU = props.warpSize;
                inf.maxWorkSize = (U32)props.maxThreadsPerBlock;
                inf.maxGroupSize = props.maxGridSize[0];
                gpuIndent[pid] = inf;
            }
            else
                printf("Cuda device #%d. error %d\n", i, err);
        }    
    }
#endif //RH_COMPILE_CPU_ONLY

    //Add the cpu last
    GPUInfos CPUinf;
    CPUinf.platformID = 0xFFFFFFFF;
    CPUinf.deviceID = 0;
    CPUinf.description = CpuInfos.cpuBrandName;
    CPUinf.deviceName = "CPU";
    CPUinf.platformName = "CPU";
    CPUinf.archName = CpuInfos.cpuArchName;
    CPUinf.memorySize = CpuInfos.avaiablelMem/1024/1024;
    CPUinf.localMemSize = 0;
    CPUinf.maxAllocSize = CPUinf.memorySize;
    CPUinf.maxCU = CpuInfos.numberOfProcessors;
    CPUinf.maxWorkSize = 1;
    CPUinf.maxGroupSize = 1;
    CPUinf.gpuType = GpuType_CPU;
    gpuIndent["CPU"] = CPUinf;
    
    for(auto& g : gpuIndent)
    {
        if (stristr(g.second.platformName.c_str(), "AMD "))
        {
            GPUInfos inf = g.second;
            inf.globalIndex = (U32)GpuManager::Gpus.size();
            inf.gpuType = GpuType_AMD;
            inf.isOpenCL = true;
            inf.platformNameSmall = "AMD";
            GpuManager::Gpus.push_back(inf);
        }
    }

    for(auto& g : gpuIndent)
    {
        if (stristr(g.second.platformName.c_str(), "NVIDIA "))
        {
            GPUInfos inf = g.second;
            inf.globalIndex = (U32)GpuManager::Gpus.size();
            inf.gpuType = GpuType_NVIDIA;
            inf.platformNameSmall = "NVIDIA";
            GpuManager::Gpus.push_back(inf);
        }
    }

    for(auto& g : gpuIndent)
    {
        if (g.second.gpuType == GpuType_CPU)
        {
            GPUInfos inf = g.second;
            inf.globalIndex = (U32)GpuManager::Gpus.size();
            inf.platformNameSmall = "CPU " + inf.description.substr(0, inf.description.find(' '));
            inf.isOpenCL = false;
            GpuManager::Gpus.push_back(inf);
        }
    }
}


bool GpuManager::SetupGPU()
{
    bool atleaseone = false;
    for (auto& g : GpuManager::Gpus)
    {
        if (!g.enabled)
            continue;

        if (g.initialized)
        {
            atleaseone = true;
            continue;
        }

        atleaseone = true;
        g.initialized = true;
        if (g.gpuType == GpuType_CPU)
        {
            if (! g_testPerformance)
                PrintOutCritical("Selecting CPU (GPU%d) %s to mine on %d logical cores with %d threads\n",
                    g.globalIndex,
                    g.description.c_str(),
                    GpuManager::CpuInfos.UserSelectedCoresCount ? GpuManager::CpuInfos.UserSelectedCoresCount : GpuManager::CpuInfos.numberOfProcessors,
                    g_cpuMinerThreads);
        }
        else
        {
            if (! g_testPerformance)
                PrintOutCritical("Selecting GPU%d %s to mine with %d threads\n",
                    g.globalIndex,
                    g.description.c_str(),
                    g.setGpuThreadCount != U32_Max ? g.setGpuThreadCount : 1);
        }
    }

    return atleaseone;
}


void GpuManager::listGPU()
{
    printf("List of gpus and cpus:\n");
    for (U32 i = 0; i < GpuManager::Gpus.size(); ++i)
    {
        string gname = GpuManager::Gpus[i].gpuName;
        if (GpuManager::Gpus[i].gpuType == GpuType_CPU)
            gname = "CPU ";
        else if (gname.empty())
            gname = FormatString("GPU%d", i);

        printf("%s : %s\n", gname.c_str(), GpuManager::Gpus[i].description.c_str());
    }
}

U32 GpuManager::GetAllGpuThreadsCount(U32& enabledGpuCount)
{
    enabledGpuCount = 0;
    U32 globalWorkSizeForAllGpuMiner = 0; 
    for(auto& gpu :  GpuManager::Gpus)
    {
        if (gpu.enabled && gpu.gpuType != GpuType_CPU)
        {
            globalWorkSizeForAllGpuMiner += gpu.globalWorkSize;
            enabledGpuCount++;
        }
    }
    return globalWorkSizeForAllGpuMiner;
}


#ifdef __GNUC__

void __cpuid(int* cpuinfo, int info)
{
	__asm__ __volatile__(
		"xchg %%ebx, %%edi;"
		"cpuid;"
		"xchg %%ebx, %%edi;"
		:"=a" (cpuinfo[0]), "=D" (cpuinfo[1]), "=c" (cpuinfo[2]), "=d" (cpuinfo[3])
		:"0" (info)
	);
}

unsigned long long _xgetbv(unsigned int index)
{
	unsigned int eax, edx;
	__asm__ __volatile__(
		"xgetbv;"
		: "=a" (eax), "=d"(edx)
		: "c" (index)
	);
	return ((unsigned long long)edx << 32) | eax;
}

#endif

void GpuManager::TestExtraInstructions()
{
	int cpuinfo[4];
	__cpuid(cpuinfo, 1);

	// Check SSE, SSE2, SSE3, SSSE3, SSE4.1, and SSE4.2 support
	CpuInfos.sseSupportted = cpuinfo[3] & (1 << 25) || false;
	CpuInfos.sse2Supportted = cpuinfo[3] & (1 << 26) || false;
	CpuInfos.sse3Supportted = cpuinfo[2] & (1 << 0) || false;
	CpuInfos.ssse3Supportted = cpuinfo[2] & (1 << 9) || false;
	CpuInfos.sse4_1Supportted = cpuinfo[2] & (1 << 19) || false;
	CpuInfos.sse4_2Supportted = cpuinfo[2] & (1 << 20) || false;
    
	// ----------------------------------------------------------------------

	// Check AVX support
	// References
	// http://software.intel.com/en-us/blogs/2011/04/14/is-avx-enabled/
	// http://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/	
	CpuInfos.avxSupportted = cpuinfo[2] & (1 << 28) || false;
	bool osxsaveSupported = cpuinfo[2] & (1 << 27) || false;
	if (osxsaveSupported && CpuInfos.avxSupportted)
	{
		// _XCR_XFEATURE_ENABLED_MASK = 0
		unsigned long long xcrFeatureMask = _xgetbv(0);
		CpuInfos.avxSupportted = (xcrFeatureMask & 0x6) == 0x6;
	}

	// ----------------------------------------------------------------------
	// Check SSE4a and SSE5 support
	// Get the number of valid extended IDs
	__cpuid(cpuinfo, 0x80000000);
	int numExtendedIds = cpuinfo[0];
	if (numExtendedIds >= 0x80000001)
	{
		__cpuid(cpuinfo, 0x80000001);
		CpuInfos.sse4aSupportted = cpuinfo[2] & (1 << 6) || false;
		CpuInfos.sse5Supportted = cpuinfo[2] & (1 << 11) || false;
	}
    
    g_isSSE41Supported = CpuInfos.sse4_1Supportted;
    PrintOutSilent("SSe3   supported : %s\n", CpuInfos.sse3Supportted ? "Yes" : "No");
    PrintOutSilent("SSe4.1 supported : %s\n", CpuInfos.sse4_1Supportted ? "Yes" : "No");
    PrintOutSilent("avx    supported : %s\n", CpuInfos.avxSupportted ? "Yes" : "No");	
}

void GpuManager::LoadCPUInfos()
{
#ifdef _WIN32_WINNT
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo); 
    CpuInfos.numberOfProcessors = siSysInfo.dwNumberOfProcessors;
    CpuInfos.activeProcessorMask = (size_t)siSysInfo.dwActiveProcessorMask;
    CpuInfos.allocationGranularity = siSysInfo.dwAllocationGranularity;
    CpuInfos.cpuArchName = "CPU";
    if (siSysInfo.wProcessorArchitecture == 9)
        CpuInfos.cpuArchName = "AMD64_CPU";
    if (siSysInfo.wProcessorArchitecture == 6)
        CpuInfos.cpuArchName = "IA64_CPU";
    if (siSysInfo.wProcessorArchitecture == 0)
        CpuInfos.cpuArchName = "INTEL_CPU";
    
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof (statex);
    if (GlobalMemoryStatusEx(&statex))
        CpuInfos.avaiablelMem = statex.ullAvailPhys;
    else
        CpuInfos.avaiablelMem = 0;

    int CPUInfo[4] = {-1};
    unsigned   nExIds, i =  0;
    char CPUBrandString[0x40];
    // Get the information associated with each extended ID.
    __cpuid(CPUInfo, 0x80000000);
    nExIds = CPUInfo[0];
    for (i=0x80000000; i<=nExIds; ++i)
    {
        __cpuid(CPUInfo, i);
        // Interpret CPU brand string
        if  (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if  (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if  (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }

    //remove double spaces    
    string brand = CPUBrandString;
#else
    CpuInfos.numberOfProcessors = sysconf(_SC_NPROCESSORS_ONLN);
    CpuInfos.allocationGranularity = 1024*64;
    CpuInfos.avaiablelMem =  sysconf(_SC_PAGESIZE) * sysconf(_SC_AVPHYS_PAGES);
    CpuInfos.cpuBrandName = "x64";
    
    string line;
    string brand;
    ifstream finfo("/proc/cpuinfo");
    while(getline(finfo,line)) {
        stringstream str(line);
        string itype;
        string info;
        if ( getline( str, itype, ':' ) && getline(str,info) && itype.substr(0,10) == "model name" ) 
	{
        	brand = info;
            break;
        }
    }    

#endif
    brand = TrimString(brand);
    char spaces[] = "                 ";
    for (int i = 8; i >= 2; i--)
    {
        spaces[i] = 0;
        ReplaceStringALL(brand, spaces, " ");
        spaces[i] = ' ';
    }
    //string includes manufacturer, model and clockspeed
    CpuInfos.cpuBrandName = brand;

    TestExtraInstructions();
}
