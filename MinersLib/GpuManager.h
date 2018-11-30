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

#pragma once

#include "corelib/Common.h"
#include "corelib/PascalWork.h"

#ifndef RH_COMPILE_CPU_ONLY

#define __CL_ENABLE_EXCEPTIONS true
#include "MinersLib/cl.hpp"

#else

//openCL stubs
namespace cl
{
    class Device {};
    class Platform {};
    class CommandQueue {};
    class Buffer {};
    class Context {};
    class Kernel {};
    class Error : public RH_Exception
    {
        public: explicit Error(const char* const& msg): RH_Exception(msg)  {}
    };
}
typedef S32 cl_int;
typedef U32 cl_uint;
typedef S64 cl_long;
typedef U64 cl_ulong;

#endif

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#endif
        
enum GpuType {  GpuType_AMD           = 0x1,
                GpuType_NVIDIA        = 0x2,
                GpuType_CPU           = 0x4};

struct CPUInfo
{
    string  cpuArchName;
    string  cpuBrandName;
    bool    sseSupportted = false;
    bool    sse2Supportted = false;
    bool    sse3Supportted = false;
    bool    ssse3Supportted = false;
    bool    sse4_1Supportted = false;
    bool    sse4_2Supportted = false;
    bool    sse4aSupportted = false;
    bool    sse5Supportted = false;
    bool    avxSupportted = false;
    U64     avaiablelMem;
    U32     numberOfProcessors;
    size_t  activeProcessorMask;
    U32     allocationGranularity;
    U64     UserSelectedCores = 0x0;  //mask used by SetProcessAffinityMask when application starts
    U64     UserSelectedCoresCount = 0;
};


class GpuManager
{
public:
    GpuManager();
    struct GPUInfos
    {
        GPUInfos() = default;
        //descr
        U32                 platformID = 0;
        U32                 deviceID = 0;       //NVIDIA + openCL
        U64                 memorySize = 0;
        U64                 localMemSize = 0; 
        U64                 maxAllocSize = 0;
        U32                 maxCU = 0; //Compute units count
        U32                 maxWorkSize = 0;    //CUDA : MaxThreadsPerBlock
        U64                 maxGroupSize = 0;   //CUDA : Max Grid dSize 
        string              description;
        string              deviceName;
        string              platformName;
        string              archName;
        string              platformNameSmall;
        string              deviceVersion;
        string              deviceExtention;
        string              gpuName; //set by the farm
        U32                 globalIndex = 0;        // in GpuManager::Gpus
        U32                 relativeIndex = 0;      // farm index
        U32                 gpuType = 0;            //1 GpuType_AMD, 2 GpuType_NVIDIA, 4 GpuType_CPU
        bool                isOpenCL = false;
        bool                initialized = false;
        //runtime variables
        bool                enabled = false;
        U32                 localWorkSize = 0;        //CUDA: blockSize
        U32                 globalWorkSize = 0;                 //a cache of m_globalWorkSize
        U32                 setGpuThreadCount = U32_Max;
        U32                 GPUFrequencyMHZ = 0;
        U32                 MEMFrequencyMHZ = 0;
        U32                 MaxFANSpeed = 0;
    };

    static std::vector<GPUInfos>    Gpus;
    static CPUInfo                  CpuInfos;

    static void                     LogDriverInfo(bool silent);
	static void                     listDevices();
    static void                     listGPU();
    static bool                     SetupGPU();
    static U32                      GetEnabledGPUCount();
    static std::vector<unsigned>    GetEnabledGPUIndices();
    static U32                      GetAllGpuThreadsCount(U32& enabledGpuCount);
    static void                     LoadGPUMap();
    static void                     LoadCPUInfos();
    static void                     TestExtraInstructions();
    
    static std::vector<cl::Device>  GetDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId);
    static std::vector<cl::Platform>GetPlatforms();
    static cl::Device               GetDeviceFromGPUMap(const GPUInfos& gpuInfo);
    static cl::Device               GetDeviceFromGlobalIndex(U32 globalIndex);

};

