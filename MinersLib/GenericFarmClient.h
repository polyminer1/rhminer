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
#include "corelib/PascalWork.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4290)
#endif

//SOLO interface
class FarmFace;
class GenericFarmClient
{
public:
    GenericFarmClient(FarmFace* farm, string farmURL, U32 port, string farmFailOverURL, string email)
    {
        m_farm = farm;
        m_farmURL = farmURL;
        m_port = port;
        m_farmFailOverURL = farmFailOverURL;
        m_email = email;
    }

    virtual PascalWorkSptr getWork()
    {
    	return PascalWorkSptr();
    }

    virtual bool submitWork(SolutionSptr sol)
    {
        return false;
    }

    virtual Json::Value awaitNewWork()
    {
    	return Json::Value();
    }
    
    virtual bool progress()
    {
    	return false;
    }

protected:
    FarmFace*   m_farm;
    string      m_farmURL;
    U32         m_port;
    string      m_farmFailOverURL;
    string      m_email;
};

typedef std::shared_ptr<GenericFarmClient> GenericFarmClientSptr;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
