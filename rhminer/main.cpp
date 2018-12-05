/**
 * rhminer code
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
#include "BuildInfo.h"
#include "ClientManager.h"
#include "MinersLib/Global.h"
#include "MinersLib/GenericMinerClient.h"
#include <atomic>

#ifndef RH_COMPILE_CPU_ONLY
#include "cuda_runtime.h"
#endif

RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("v", g_logVerbosity, "General", "Log verbosity. From 0 to 3. 0 no log, 1 normal log, 2 include warnings. 3 network (only in log file). Default is 1",0, 3);
RHMINER_COMMAND_LINE_DEFINE_GLOBAL_INT(g_logVerbosity, 1)
bool g_ExitApplication = false;

void DisplayHelp(CmdLineManager& cmdline)
{ 
    cmdline.List();
    exit(0);
}

using namespace std;
using namespace boost::algorithm;

#ifdef _WIN32_WINNT
void HandleExit();
BOOL WINAPI ConsoleHandler(DWORD signal);
long   __stdcall   GlobalExpCallback(_EXCEPTION_POINTERS*   excp);
#endif

bool g_appActive = true;
 
int main(int argc, char** argv)
{
    ////////////////////////////////////////////////////////////////////
    //
    //      App header
    //
#ifndef RH_COMPILE_CPU_ONLY 
    printf("\n  rhminer v%s beta for CPU and NVIDIA GPUs by polyminer1 (https://github.com/polyminer1/rhminer)\n", RH_PROJECT_VERSION);
    printf("  Build %s (CUDA SDK %d.%d) %s %s\n\n", RH_BUILD_TYPE, CUDART_VERSION/1000, (CUDART_VERSION % 1000)/10, __DATE__, __TIME__);
#else
    printf("\n  rhminer v%s beta for CPU by polyminer1 (https://github.com/polyminer1/rhminer)\n", RH_PROJECT_VERSION);
    printf("  Build %s %s %s\n\n", RH_BUILD_TYPE, __DATE__, __TIME__);
#endif    

	printf("  Donations : Pascal account 529692-23 \n");
    printf("  Donations : Bitcoin address 19GfXGpRJfwcHPx2Nf8wHgMps8Eat1o4Jp \n\n");

#ifdef _WIN32_WINNT
    std::atexit(HandleExit);
    if (!SetConsoleCtrlHandler(ConsoleHandler, TRUE)) 
    {
        printf("\nError: Could not set control handler"); 
        return 14454;
    }

    SetUnhandledExceptionFilter(GlobalExpCallback);

    // Initialize Winsock
    WSAData wsa_data;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (iResult != 0) {
        printf("WSAStartup() failed with Error. %d\n", iResult);
        return 1;
    }
#endif

    // Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");
    rand32_reseed((U32)(TimeGetMilliSec())^0xF5E8A1C4);

    //Preparse log file name cuz we need it prior init
    for (int i = 0; i < argc; i++)
    {
        if (stristr(argv[i], "logfilename") && i+1 < argc)
        {
            SetLogFileName(argv[i + 1]);
            break;
        }
    }

    GlobalMiningPreset::I().Initialize(argv, argc);

    bool displayHelp = false;
    CmdLineManager::GlobalOptions().RegisterFlag("h",           "General", "Display Help", [&]() { displayHelp = true; });
    CmdLineManager::GlobalOptions().RegisterFlag("help",        "General", "Display Help", [&]() { displayHelp = true; });
    CmdLineManager::GlobalOptions().RegisterFlag("?",           "General", "Display Help", [&]() { displayHelp = true; });

    setThreadName("Log");

    //set the coin count right in GpuManager::Gpus
    GpuManager::LoadGPUMap();

    //DISPLAY HELP
    CmdLineManager::GlobalOptions().Parse(argc, argv, true);
    if (displayHelp || argc == 1)
        DisplayHelp(CmdLineManager::GlobalOptions()); //exit app

    KernelOffsetManager::Reset(0);

#ifdef _WIN32_WINNT
    if (g_setProcessPrio == 0)
    {
        BOOL res = SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN);
        if (!res)
        {
            PrintOut("Error. %d Cannot set priority to background mode. Using IDLE.\n", GetLastError());
            SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
            g_setProcessPrio = 1;
        }
        else
        {
            HWND hcon = GetConsoleWindow();
            if (hcon) 
                ShowWindow(hcon, SW_HIDE);
        }
    }
    else if (g_setProcessPrio == 1)
        SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    else if (g_setProcessPrio == 2)
        //Force relax mode 
        g_setProcessPrio = 1;
    else
    {
        //High priority mode : Miner threads and stratum have High priority
    }
#endif

    ClientManager::I().Initialize();    

    while(g_appActive)
    {
        CpuSleep(200);
        //do stuffs !
    }

    CpuSleep(200);
    
#ifdef _WIN32_WINNT
    if (g_setProcessPrio == 0)
        SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_END);
#endif

	return 0;
}

#ifdef _WIN32_WINNT
void HandleExit()
{
    g_ExitApplication = true;
    if (g_setProcessPrio == 0)
        SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_END);

    ClientManager::I().Shutdown();
}

bool isCtrlC = false;
BOOL WINAPI ConsoleHandler(DWORD signal) 
{ 
     if ((signal == CTRL_C_EVENT ||
        signal == CTRL_BREAK_EVENT ||
        signal == CTRL_CLOSE_EVENT) && !isCtrlC)
    {
        isCtrlC = true;
        if (g_setProcessPrio == 0)
            SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_END);

        exit(0);
    }

    return TRUE;
}

long  __stdcall GlobalExpCallback(_EXCEPTION_POINTERS* excp)
{
    //printf("Error. Global exception 0x%X\n", excp->ExceptionRecord->ExceptionCode);
    return EXCEPTION_EXECUTE_HANDLER;
}

#endif
