/**
 * Generic command line parser
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
#ifdef _WIN32_WINNT
#include <json/json.h>
#else
#include <jsoncpp/json/json.h>
#endif


#include "corelib/basetypes.h" 
#include <condition_variable>
#include "FixedHash.h"
#include "utils.h"


//Enable C3 on cpu only. To bootstrap the chain.
//#define RH_FORCE_PASCAL_V3_ON_CPU

struct PascalSolution;

struct PascalWorkPackage
{
    PascalWorkPackage();
    explicit PascalWorkPackage(bool isSolo) { m_isSolo = isSolo; }
    explicit PascalWorkPackage(const PascalWorkPackage& c);
    ~PascalWorkPackage();
    PascalWorkPackage* Clone();

    void    Init(const string& job, const h256& prevHash, const string& coinbase1, const string& coinbase2, const string& nTime, bool cleanWork, const string& nonce1, U32 nonce2Size, U64 extranonce);
    bool    Eval(PascalSolution* solPtr);
    void    ComputeWorkDiff(double& diff);
    bool    IsSame(PascalWorkPackage* work);
    U32     GetDeviceTargetUpperBits();
    U64     GetDeviceTargetUpperBits64();
    void    UpdateHeader();
    void    ComputeTargetBoundary();
    bool    IsEmpty();
    h256    RebuildNonce(U64 nonce);
    string  ComputePayload();

    static void     ComputeTargetBoundary(h256& boundary, double& diff, double diffMultiplyer);

    string          m_jobID;
    string          m_ntime;   
    h256            m_prev_hash;
    U64             m_startNonce = 0; 
    mutable h256    m_boundary;                
    mutable h256    m_deviceBoundary;
    mutable h256    m_soloTargetPow;
    double          m_workDiff = 1.0;          
    double          m_deviceDiff = 1.0;        
    bool            m_localyGenerated = false; 
    U32             m_nonce2 = U32_Max;
    bytes           m_fullHeader;
    string          m_coinbase1;
    string          m_coinbase2;
    string          m_nonce1;
    unsigned        m_nonce2Size = 0; 
    bool            m_clean = false;
    U64             m_nonce2_64 = 0;
    
    //solo stuff
    bool            m_isSolo = false;
};
typedef std::shared_ptr<PascalWorkPackage> PascalWorkSptr;

struct PascalSolution
{
    U64                 GetCurrentEvaluatingNonce();
    void                SetCurrentEvaluatingNonceIndex(U32 i);
    bool                Eval() { return m_work->Eval(this);  }
        
    h256                m_calcHash;
    std::vector<U64>    m_results; 
    U32                 m_gpuIndex = 0;
    bool                m_isFromCpuMiner = false;
    PascalWorkSptr      m_work;

private:
    U32                 _eval_current_result_index = 0; //index used to travers m_results and eval each value

};
typedef std::shared_ptr<PascalSolution> SolutionSptr;

