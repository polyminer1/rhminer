/**
 * Various utility functions
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
#include <random>
#include "utils.h"
#include <sys/stat.h>

#ifdef _WIN32_WINNT
#include <io.h>
#include <direct.h>
#include <time.h>

#else //_WIN32_WINNT

#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdarg.h>
#include <mm_malloc.h>

#if !defined(_WIN32_WINNT)
#define OutputDebugStringA(...) 
#endif

#define _vsnprintf vsnprintf

#endif //_WIN32_WINNT

//#define RH_OUTPUT_TO_DEBUGGER

extern int g_logVerbosity;

#define STR_FORMAT_TEXT_BUFFER_COUNT    (64+8)
#define STR_FORMAT_TEXT_BUFFER_SIZE     4096

extern char const* getThreadName();
typedef std::chrono::system_clock Clock;

string TrimZeros(const string& str, bool tailing, bool heading)
{
    const char* head = str.c_str();
    if (heading)
    {
        while(*head == '0' && *head != 0)
        {
            if (*(head + 1) == '.')
                break;
            head++;
        }
    }

    string rstring(head);
    if (tailing) 
    {
        char* foot = (char*)rstring.c_str() + rstring.length()-1;
        while(*foot== '0' && foot != str.c_str())
        { 
            *foot = 0;
            foot--;
        }
        if (*foot == '.')
            *foot = 0;
    }

    return rstring;
}

string TrimString(const string& str)
{
    const char* head = str.c_str();
    while((*head == ' ') || ((*head == '\t') && (*head != 0)))
        head++;

    string rstring(head);
    char* foot = (char*)rstring.c_str() + rstring.length()-1;
    while((*foot== ' ') || ((*foot== '\t') && (foot != str.c_str())))
    { 
        *foot = 0;
        foot--;
    }

    return rstring;
}


char* stristr(const char *string, const char *Pattern)
{
      char *pptr, *sptr, *start;
      size_t  slen, plen;

      for (start = (char *)string,
           pptr  = (char *)Pattern,
           slen  = (size_t)strlen(string),
           plen  = (size_t)strlen(Pattern);

           /* while string length not shorter than pattern length */
           slen >= plen;

           start++, slen--)
      {
            /* find start of pattern in string */
            while (toupper(*start) != toupper(*Pattern))
            {
                  start++;
                  slen--;

                  /* if pattern longer than string */

                  if (slen < plen)
                        return(NULL);
            }

            sptr = start;
            pptr = (char *)Pattern;

            while (toupper(*sptr) == toupper(*pptr))
            {
                  sptr++;
                  pptr++;

                  /* if end of pattern then pattern was found */

                  if ('\0' == *pptr)
                        return (start);
            }
      }
      return(NULL);
}

bool ReplaceString(string& str, const char* toFind, const char* toReplace)
{
    const char* pos = stristr(str.c_str(), toFind);
    if (pos != 0)
    {
        string right(pos+strlen(toFind));
        string left(str.c_str(), (size_t)(pos-str.c_str()));
        str = left;
        str += toReplace;
        str += right;
        return true;
    }
    return false;
}



std::mutex* g_GlobalOutputMutex = 0;
std::mutex* GlobalOutputMutex()
{
    if (!g_GlobalOutputMutex)
        g_GlobalOutputMutex = new std::mutex;

    return g_GlobalOutputMutex;
}

const char* GetOutputDecoration(const char* szBuffer)
{
    if (*szBuffer != '~')
    {
        const char* str;
        char const* tname = getThreadName();
#ifdef _DEBUG
        str = FormatString("%-5s %llu  %s", tname, TimeGetMilliSec(), szBuffer);
#else
        char tstr[64];
        GetSysTimeStrF(tstr, sizeof(tstr), "%H:%M:%S");
        str = FormatString("%-5s %-9s  %s", tname, tstr, szBuffer);
#endif

        return str;
    }
    else
        return szBuffer+1;
}



FILE* Logfile = 0;
char LogfileFileName[256] = {0};
void SetLogFileName(const char* fn)
{
    if (Logfile)
        fclose(Logfile);
    
    strncpy(LogfileFileName, fn, sizeof(LogfileFileName) - 1);

    Logfile = fopen(fn,"w");
    
    if (!Logfile)
    {
        LogfileFileName[0] = 0;
        printf("ERROR: Cannot create log file %s\n", fn);
    }
}

void _PrintToLog(const char* szBuffer)
{
    if (Logfile)
    {
        fprintf(Logfile, "%s", szBuffer);
        fflush(Logfile);
    }
}


#ifndef RHMINER_RELEASE
void DebugOut(const char *szFormat, ...)
{
    va_list argList;
    char szBuffer[RHMINER_KB(4)];

    va_start(argList, szFormat);
    _vsnprintf(szBuffer, sizeof(szBuffer), szFormat, argList);
    va_end(argList);

    GlobalOutputMutex()->lock();
    const char* str = GetOutputDecoration(szBuffer);
  #if defined(_WIN32_WINNT)
    OutputDebugStringA(str);
  #else
    printf(str);
  #endif
    _PrintToLog(str);
    GlobalOutputMutex()->unlock();
}
#endif


void PrintOutWarning(const char *szFormat, ...) 
{
    va_list argList;
    char szBuffer[RHMINER_KB(12)];

    va_start(argList, szFormat);
    _vsnprintf(szBuffer, sizeof(szBuffer), szFormat, argList);
    va_end(argList);

    GlobalOutputMutex()->lock();
    const char* str = GetOutputDecoration(szBuffer);
    if (g_logVerbosity > 1)
    {
        printf("%s", str);
#if defined(RHMINER_DEBUG) || defined(RH_OUTPUT_TO_DEBUGGER)
        OutputDebugStringA(str);
#endif
    }

    _PrintToLog(str);
    GlobalOutputMutex()->unlock();
}

void PrintOutCritical(const char *szFormat, ...) 
{
    va_list argList;
    char szBuffer[RHMINER_KB(12)];

    va_start(argList, szFormat);
    _vsnprintf(szBuffer, sizeof(szBuffer), szFormat, argList);
    va_end(argList);

    GlobalOutputMutex()->lock();
    const char* str = GetOutputDecoration(szBuffer);
    printf("%s", str);
#if defined(RHMINER_DEBUG) || defined(RH_OUTPUT_TO_DEBUGGER)
    OutputDebugStringA(str);
#endif
    _PrintToLog(str);
    GlobalOutputMutex()->unlock();
}

void PrintOut(const char *szFormat, ...)
{
    va_list argList;
    char szBuffer[RHMINER_KB(12)];

    va_start(argList, szFormat);
    _vsnprintf(szBuffer, sizeof(szBuffer), szFormat, argList);
    va_end(argList);

    GlobalOutputMutex()->lock();
    const char* str = GetOutputDecoration(szBuffer);
    if (g_logVerbosity > 0)
    {
        printf("%s", str);
#if defined(RHMINER_DEBUG) || defined(RH_OUTPUT_TO_DEBUGGER)
        OutputDebugStringA(str);
#endif
    }
    _PrintToLog(str);
    GlobalOutputMutex()->unlock();
}

void PrintOutSilent(const char *szFormat, ...)
{
    if (g_logVerbosity > 2)
    {
        va_list argList;
        char szBuffer[RHMINER_KB(12)];

        va_start(argList, szFormat);
        _vsnprintf(szBuffer, sizeof(szBuffer), szFormat, argList);
        va_end(argList);

        GlobalOutputMutex()->lock();
        const char* str = GetOutputDecoration(szBuffer);
        _PrintToLog(str);
#if defined(RHMINER_DEBUG) || defined(RH_OUTPUT_TO_DEBUGGER)
        OutputDebugStringA(str);
#endif
        GlobalOutputMutex()->unlock();
    }
}

U32 AtomicAdd(U32& x, U32 val)
{
#ifdef _WIN32_WINNT
    return InterlockedAdd((LONG*)&x, val);
#else
    return __sync_add_and_fetch(&x, val);
#endif
}
U64 AtomicAdd(U64& x, U64 val)
{
#ifdef _WIN32_WINNT
    return (U64)InterlockedAdd64((LONG64*)&x, val);
#else
    return __sync_add_and_fetch(&x, val);
#endif
}
U32 AtomicIncrement(U32& x)
{
#ifdef _WIN32_WINNT
    return InterlockedIncrement((LONG*)&x);
#else
    return __sync_add_and_fetch(&x, 1);
#endif
}
U64 AtomicIncrement(U64& x)
{
#ifdef _WIN32_WINNT
    return (U64)InterlockedIncrement((volatile unsigned __int64*)&x);
#else
    return __sync_add_and_fetch(&x, 1);
#endif
}
U32 AtomicDecrement(U32& x)
{
#ifdef _WIN32_WINNT
    return InterlockedDecrement((LONG*)&x);
#else
    return __sync_sub_and_fetch(&x, 1);
#endif
}
U64 AtomicDecrement(U64& x)
{
#ifdef _WIN32_WINNT
    return (U64)InterlockedDecrement((volatile unsigned __int64*)&x);
#else
    return __sync_sub_and_fetch(&x, 1);
#endif
}
U32 AtomicSet(U32& x, U32 val)
{
#ifdef _WIN32_WINNT
    return InterlockedExchange((LONG*)&x, val);
#else
    return __sync_lock_test_and_set(&x, val);
#endif
}
U64 AtomicSet(U64& x, U64 val)
{
#ifdef _WIN32_WINNT
    return (U64)InterlockedExchange64((LONG64*)&x, val);
#else
    return __sync_lock_test_and_set(&x, val);
#endif
}
U32 AtomicGet(U32& x)
{
#ifdef _WIN32_WINNT
    return InterlockedAdd((LONG*)&x, 0);
#else
    return __sync_add_and_fetch(&x, 0);
#endif
}
U64 AtomicGet(U64& x)
{
#ifdef _WIN32_WINNT
    return (U64)InterlockedAdd64((LONG64*)&x, 0);
#else
    return __sync_add_and_fetch(&x, 0);
#endif
}
void RH_SetThreadPriority(RH_ThreadPrio prio)
{ 
#ifdef _WIN32_WINNT
    switch(prio)
    {
        case RH_ThreadPrio_Normal:  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL); break;
        case RH_ThreadPrio_Low:     SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL); break;
        case RH_ThreadPrio_High:    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); break;
        case RH_ThreadPrio_RT:      SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL); break;        
    }
#else
    switch(prio)
    {
        case RH_ThreadPrio_Normal:  setpriority(PRIO_PROCESS, 0, 0); break;
        case RH_ThreadPrio_Low:     setpriority(PRIO_PROCESS, 0, -5); break;
        case RH_ThreadPrio_High:    setpriority(PRIO_PROCESS, 0, 20); break;
        case RH_ThreadPrio_RT:      setpriority(PRIO_PROCESS, 0, 25); break;
    }
#endif
}

void SetThreadPriority_EXT(void* threadNativeHandle)
{
#if 0 && defined(__linux__)
    //Use pthread
    int policy;
    pthread_t threadID = (pthread_t)threadNativeHandle;

    sched_param param;
    int retcode = pthread_getschedparam(threadID, &policy, &param);
    RHMINER_ASSERT(ertCode == 0);

    policy = SCHED_FIFO;
    switch(prio)
    {
        case RH_ThreadPrio_Normal:  param.sched_priority = 0; break;
        case RH_ThreadPrio_Low:     param.sched_priority = 8; break;
        case RH_ThreadPrio_High:    param.sched_priority = -8; break;
    }    

    retcode = pthread_setschedparam(threadID, policy, &param);
    RHMINER_ASSERT(ertCode == 0);  
#endif
}

//--------------------------------
void GetSysTimeStrF(char* buf, size_t buffSize, const char* frmt, bool addMillisec)
{
    auto now = Clock::now();
    auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto fraction = now - seconds;
    time_t cnow = Clock::to_time_t(now);
    std::tm * ptm = localtime(&cnow);
    strftime(buf, buffSize, frmt, ptm);

    if (addMillisec)
    {
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(fraction);
        strcat(buf, FormatString(".%04d", milliseconds.count()));
    }
}

//-------------------------

#ifdef _WIN32_WINNT
U64 TimeGetMicroSec() 
{
    LARGE_INTEGER Counter;
    LARGE_INTEGER Freq;
    U64 Result = 0;
    QueryPerformanceFrequency(&Freq); 
    if(Freq.QuadPart)
    {
        QueryPerformanceCounter(&Counter);
        Result = ((1000 * Counter.QuadPart) / (Freq.QuadPart / 1000));
    }

    return(Result);
}
#else
auto begening = std::chrono::steady_clock::now();
U64 TimeGetMicroSec() 
{ 
    auto n = std::chrono::steady_clock::now(); 
    return std::chrono::duration_cast<std::chrono::microseconds>(n-begening).count();
}
#endif

void GetSysTimeStr(char* buf, size_t buffSize)
{
    bool addMili = false;
    //addMili = true;

    GetSysTimeStrF(buf, buffSize, "%H:%M:%S", addMili);
}

int ToIntX(const string& s) 
{ 
    try 
    { 
        const char* str = s.c_str();
        if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X'))
            str = str + 2;
        return strtol(str, 0, 16); 
    } 
    catch (...) 
    { 
        return 0; 
    } 
}

unsigned ToUIntX(const string& s) 
{ 
    try 
    {  
        const char* str = s.c_str();
        if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X'))
            str = str + 2;
        return strtoul(str, 0, 16); 
    } 
    catch (...) 
    { 
        return 0; 
    }
}

S64 ToInt64X(const string& s) 
{ 
    try 
    { 
        const char* str = s.c_str();
        if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X'))
            str = str + 2;
        return strtoll(str, 0, 16); 
    } 
    catch (...) 
    { 
        return 0; 
    } 
}

U64 ToUInt64X(const string& s) 
{ 
    try 
    {  
        const char* str = s.c_str();
        if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X'))
            str = str + 2;
        return strtoull(str, 0, 16); 
    } 
    catch (...) 
    { 
        return 0; 
    }
}

const char* HashrateToString(float hashrate)
{
    /*
    if (hashrate > (1000.0f*1000.0f))
        return FormatString("%.2f MH/S", hashrate/(1000.0f*1000.0f));
    else if (hashrate > 1000.0f)
        return FormatString("%.3f KH/S", hashrate/1000.0f);
    else*/
        return FormatString("%d H/S", (U32)hashrate);

}


const char* DiffToStr(float diff) 
{
    string s = FormatString("%.9f", diff);
    return FormatString("%s", TrimZeros(s, true, true).c_str());
}

const char* SecondsToStr(U64 sec) 
{
    int m = int(sec / 60) % 60;
    int s = int(sec) % 60;
    int h = int((sec / 60) / 60);
    return FormatString("%02d:%02d:%02d", h,m,s);
}



const char* FormatString(const char * pFormat, ... )
{
    static char BufferList[STR_FORMAT_TEXT_BUFFER_COUNT][STR_FORMAT_TEXT_BUFFER_SIZE];
    static U32 BufferIndex=0;

    unsigned index = AtomicIncrement(BufferIndex);
    index = index % STR_FORMAT_TEXT_BUFFER_COUNT;

	const size_t BufferSize=sizeof(BufferList[0])/sizeof(BufferList[0][0])-1;
	char *CurrentBuffer=BufferList[index];

	va_list Args;
	va_start(Args,pFormat);
	_vsnprintf(CurrentBuffer,BufferSize,pFormat,Args);
	va_end(Args);

	CurrentBuffer[BufferSize]='\0';		// Fail safe termination

	return CurrentBuffer;
}



void ReplaceStringALL(string& str, const char* toFind, const char* toReplace)
{
    const char* pos = stristr(str.c_str(), toFind);
    const char* lastPos = str.c_str();
    string right;
    string left;
    string newStr;
    newStr.reserve(str.length()*2);
    size_t lenToFind = strlen(toFind);
    if (pos)
    {
        while(pos)
        {
            newStr.append(lastPos, (size_t)(pos-lastPos));
            newStr.append(toReplace);
            lastPos = pos+lenToFind;
            pos = stristr(pos+lenToFind, toFind);
        }
        newStr.append(lastPos);
        str = newStr;
    }    
}

std::vector<string> GetTokens(const string& data, const char* delimiter)
{
    std::vector<string> tokens;
    string str = data;
    char* token = strtok((char*)str.c_str(), delimiter);
    while (token != NULL)
    {
        tokens.push_back(token);
        token = strtok(NULL, delimiter);
    }
    
    return tokens;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Event

Event::Event(bool initiallySet, bool autoReset)
{
#ifdef RH_USE_WINDOWS_EVENTS
    evhHandle = CreateEvent(NULL, !autoReset, initiallySet, NULL);
#else
    m_isDone = initiallySet;
#endif
}


Event::~Event()
{

#ifdef RH_USE_WINDOWS_EVENTS
    ::CloseHandle(evhHandle);
#else
    m_isDone = true;
#endif
}



void Event::SetDone()
{
#ifdef RH_USE_WINDOWS_EVENTS
    ::SetEvent(evhHandle);
#else
    m_isDone = true;
    m_condition.notify_all();
#endif
}


void Event::Reset()
{
#ifdef RH_USE_WINDOWS_EVENTS
    ::ResetEvent(evhHandle);
#else
    m_isDone = false;
    m_condition.notify_all();
#endif
}


void Event::WaitUntilDone()
{
#ifdef RH_USE_WINDOWS_EVENTS
    ::WaitForSingleObject(evhHandle, INFINITE);
#else
    std::unique_lock<std::mutex> lock( m_mutex );
    while( !m_isDone)
    {
        m_condition.wait( lock );
    }
#endif
}

//----------------------------------------------------------------------
// CRC32
U32 RH_xcrc32(const unsigned char *buf, int len, U32 init)
{
    U32 crc = init;
    while (len--)
    {
        crc = (crc << 8) ^ crc32_table[((crc >> 24) ^ *buf) & 255];
        buf++;
    }
    return crc;
}
/////////////////////////////////////////////////////////////////////////////

U64 rand64()
{
    static std::mt19937_64 s_gen(std::random_device{}());
    return std::uniform_int_distribution<U64>{}(s_gen);
}

static std::mt19937 s_gen(std::random_device{}());

void rand32_reseed(U32 _seed)
{
    s_gen.seed(_seed);
}

U32 rand32()
{
    return std::uniform_int_distribution<U32>{}(s_gen);
}


#define le64toh(x) (x)
/* Converts a little endian 256 bit value to a double */
double le256todouble(const void *target)
{
    U64 *data64;
    double dcut64;

    data64 = (U64 *)((unsigned char *)target + 24);
    dcut64 = le64toh(*data64) * bits192;

    data64 = (U64 *)((unsigned char *)target + 16);
    dcut64 += le64toh(*data64) * bits128;

    data64 = (U64 *)((unsigned char *)target + 8);
    dcut64 += le64toh(*data64) * bits64;

    data64 = (U64 *)target;
    dcut64 += le64toh(*data64);

    return dcut64;
}


///////////////////////////////////////////////////////
// memory
//

void* RH_SysAlloc(size_t s)
{
#ifdef _WIN32_WINNT
    return _aligned_malloc(s, 4096);
#else
    return _mm_malloc( s, 4096 );
#endif
}

void RH_SysFree(void* ptr)
{
#ifdef _WIN32_WINNT
    _aligned_free(ptr);
#else
    _mm_free(ptr);
#endif    
}

