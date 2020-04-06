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

#pragma once

// Visual Studio doesn't support the inline keyword in C mode
#if defined(_MSC_VER) && !defined(__cplusplus)
#define inline __inline
#endif

// pretend restrict is a standard keyword
#if defined(_MSC_VER)
#define restrict __restrict
#else
#define restrict __restrict__
#endif


//----------------------------------------------------------------------------
#include <exception>

#if __cplusplus >= 201103L
#define RH_OVERRIDE override
#define RH_NOEXCEPT noexcept
#elif defined(_MSC_VER) && _MSC_VER > 1600 && _MSC_VER < 1900
#define RH_OVERRIDE override
#define RH_NOEXCEPT throw()
#elif defined(_MSC_VER) && _MSC_VER >= 1900
#define RH_OVERRIDE override
#define RH_NOEXCEPT noexcept
#else
#define RH_OVERRIDE
#define RH_NOEXCEPT throw()
#endif

#if defined(_WIN32)    || defined(_WIN64) || defined(__TOS_WIN__) || defined(__WINDOWS__)
    #define RH_OS_NAME "Windows"
#elif defined(sun) || defined(__sun) || defined(__SVR4) || defined(__svr4__)
    #define RH_OS_NAME "Solaris"
#else
    #if defined(__linux__)
        #define RH_OS_NAME "Linux"
    #elif defined(BSD)
        #if defined(MACOS_X) || (defined(__APPLE__) & defined(__MACH__))
            #define RH_OS_NAME "MacOS"
        #elif defined(macintosh) || defined(Macintosh)
            #define RH_OS_NAME "Macintosh"
        #elif defined(__OpenBSD__)
            #define RH_OS_NAME "OpenBSD"
        #else
            #define RH_OS_NAME "BSD"
        #endif
    #else
        #define RH_OS_NAME "OS"
    #endif
#endif


#define RH_BUILD_TYPE RHMINER_STRINGIZE(RH_ARCH)

class RH_Exception : public std::exception 
{
public:
    explicit RH_Exception(const char* const& msg): msg_(msg)  {}
    ~RH_Exception() RH_NOEXCEPT {}
    virtual char const* what() const RH_NOEXCEPT RH_OVERRIDE  {return msg_; }

protected:
  const char* msg_;
};
