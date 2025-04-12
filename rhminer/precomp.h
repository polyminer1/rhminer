#pragma once
#define BOOST_TIMER_ENABLE_DEPRECATED
#//#define _WINSOCKAPI_    // stops windows.h including winsock.h
#ifdef _WIN32_WINNT
    #include <winsock2.h>
    #include <windows.h>
#else
    //#define __LITTLE_ENDIAN 1234
    #ifdef __linux__
    //This is an aberation, we need to include little_endian.h or linux and boost core headers dont compile ! I am speechless.
    #include <linux/byteorder/little_endian.h>
    #include <endian.h>
    #else
		#include <sys/mman.h>
		#include <sys/time.h>
		#include <sys/param.h> /* need this to define BSD */
		#include <unistd.h>
		#include <fcntl.h>
    #endif
    #include <stdlib.h>
#endif

#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <assert.h>
#include <signal.h>

#include "corelib/rh_endian.h"
#include "corelib/utils.h"
#include "corelib/Common.h"
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/timer.hpp>
#include <boost/bind.hpp>
#include <boost/optional.hpp>

#if defined __CUDACC__
#error dont compile cuda with boost
#endif
