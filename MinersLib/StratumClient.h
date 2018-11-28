/**
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

#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <random>
#include "MinersLib/Farm.h"
#include "corelib/PascalWork.h"
#include "MinersLib/Miner.h"
#include "rhminer/CommandLineManager.h"
#include "MinersLib/Global.h"
#include "BuildInfo.h"
    
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL("dar", g_DisableAutoReconnect, "Network", "Disable auto-reconnect on connection lost. Note : The miner will exit uppon loosing connection. ")
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("maxsubmiterrors", g_maxConsecutiveSubmitErrors, "General", "Stop the miner when a number of consecutive submit errors occured. Default is 10 consecutive errors. This is usefull when mining into local wallet.", 3, 65535);
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL("forcesequentialnonce", g_forceSequentialNonce, "Debug", "(For debugging purpose) Force search nonce to be sequential, starting at 0. WARNING: This will gerate alot of uncle and refused solutions.")
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL("disablecachednoncereuse", g_disableCachedNonceReuse, "Debug", "(For debugging purpose) Disable RandomHash cached nonce reuse. This will lower hashrate substantially.")
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_STRING("extrapayload", g_extraPayload, "General", "An extra payload to be added when submiting solution to local wallet.")
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("apiport", g_apiPort, "General", "Tcp port of the remote api. Default port is 71111. Set to 0 to disable server", 0, 32768)
RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT("worktimeout", g_workTimeout, "General", "No new work timeout. Default is 60 seconds", 0, 1000)

using namespace std;
//using namespace boost::asio;
using boost::asio::ip::tcp;

struct StratumInit
{
    StratumInit(Farm* _f, 
        string const & _host, 
        string const & _port,
        string const & _user,
        string const & _pass,
        int const & _retries,
        string const & _email,
        bool _soloOvertStratum):f(_f),
                        host(_host),
                        port(_port),
                        user(_user),
                        pass(_pass),
                        retries(_retries),
                        email(_email),
                        soloOverStratum(_soloOvertStratum){}

    Farm*       f;
    string      host;
    string      port;
    string      user;
    string      pass;
    int         retries;
    int         worktimeout;
    string      email;
    bool        soloOverStratum;
};


class StratumClient : public Worker
{
public:
    StratumClient(const StratumInit& initData);
    ~StratumClient();

	void SetFailover(string const & host, string const & port, string const & user, string const & pass);
    ServerCredential GetFailover() { return m_failover;}

	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected && m_authorized; }
	
    Farm*   getFarm() { return m_farm; }
    void    GetUserPW(string& user, string& pw);
    int	    GetMaxREconectRetry() {return m_maxRetries;}

    void    CloseConnection() { Disconnect(); }
    void    ReconnectToServer() { Reconnect(); }
    void    SetDevFeeCredentials(const string& param);

    U32     GetDropConnectionCount() { return m_dropConnectionCount; }
    U64     GetLastDropConnectionTime() {return m_dropConnectionLastTime; }
    U32     ReceivedWorkCount() { return m_receivedWorkCount; }

    bool IsStarted() { return m_started; }
    bool IsWorkTimedOut() { return m_connected && m_running && m_lastReceivedCommandTime && (TimeGetMilliSec() - m_lastReceivedCommandTime) > g_workTimeout*1000; }

    virtual void StartWorking();
    virtual void Connect();
    virtual void Disconnect();
    virtual void StopFarming();
    virtual void Reconnect(U32 preSleepMS = 0);
    virtual void Write(tcp::socket& socket, boost::asio::streambuf& buff);

    bool                GetCurrentWorkInfo(h256& header);
    ServerCredential*   GetCurrentCred();
    float               GetDiff() { return (float)m_nextWorkDifficulty; }
    
    virtual bool    Submit(SolutionSptr sol);
    virtual void    InitializeWP(PascalWorkSptr wp);
    bool            HandleMiningSubmitResponceResult(Json::Value& responseObject, string& errorStr, U64 lastMethodCallTime);

    //CPU mining stuff
    U64             GetLastSubmitTime() { return m_lastSubmitTime;}
    bool            IsSoloMining();

protected:
    //interface
    virtual void    PrepareWorkInternal(PascalWorkSptr wp);
    virtual void    WorkLoop() override;
    virtual void    Preconnect();
    virtual string  ReadLineFromServer();
    virtual void    ProcessReponse(Json::Value& responseObject);

    //Stratum protocol
    virtual void OnPostConnect();
    virtual void RequestCleanNonce2();
    virtual bool ProcessMiningNotify(Json::Value& arrayParam);
    virtual void ProcessSetDiff(Json::Value& responseObject);
    virtual void WriteGenericAnwser(const string& line, U32 workID);
    virtual void ProcessExtranonce(Json::Value& responseObject);

    virtual void CallMiningSubscribeMethod();
    virtual void CallSubmit(SolutionSptr solution);
    virtual void RespondExtraNonceSubscribe(const string& method, Json::Value& responseObjectd, U64 extraParam);
    virtual void RespondAuthorize(Json::Value& responseObject, U64 extraParam);
    virtual void RespondMiningSubmit(Json::Value& responseObject, U64 extraParam, U64 lastMethodCallTime);
    virtual void RespondSubscribe(Json::Value& responseObjec, U64 extraParam);
    
    virtual void SendWorkToMiners(PascalWorkSptr wp);
    virtual bool ValidateNewWork(PascalWorkSptr& work);
    
    U32 GetNewNonce2();

    virtual PascalWorkSptr  InstanciateWorkPackage(PascalWorkSptr* cloneFrom = 0);
    virtual U32             GetDefaultNonce2Size() { return 4; }
    virtual void            CallJsonMethod(string methodName, string params, U64 gpuIndexOrRigID = 0, string additionalCallParams = "", bool dontPutID = false);
    void                    MiningNotify(Json::Value& responseObject);
    void                    SetStratumDiff(float stratDiff);
    void                    PrintDonationMsg();

protected:
	ServerCredential*   m_active;
	ServerCredential    m_primary;
	ServerCredential    m_failover;
    ServerCredential    m_devFee;
    bool    m_devFeeConnectionMode = false;
    string  m_userAgent = RH_PROJECT_NAME "/" RH_PROJECT_VERSION "/" RH_BUILD_TYPE;

	bool m_authorized;
	bool m_connected;
	bool m_running = true;
    h256 m_sessionRandomID;

    U32 m_sleepBeforeConnectMS = 0;
	int	m_retries = 0;
	int	m_maxRetries;  //set by -retries

    U64 m_lastReceivedCommandTime = 0;
    string m_lastReceivedMiningNotify;
	string m_lastReceivedLine;
    U32   m_processedResponsesCount = 0;
	Farm* m_farm;
    std::mutex m_reconnMutex;
	mutex m_currentMutex;
	
    boost::asio::io_service m_io_service;
	tcp::socket             m_socket;
	boost::asio::streambuf  m_requestBuffer;
	boost::asio::streambuf  m_responseBuffer;

	string      m_email;
    string      m_lastActiveHost;
    bool        m_soloMining = false;
    U32         m_workID; //  work_id++
    atomic_bool m_started;

    //stats
    U32         m_receivedWorkCount = 0;
    U32         m_dropConnectionCount = 0;
    U64         m_dropConnectionLastTime = 0;
    

    //Json call managments
    std::mutex m_CallJsonMethodMutex;
    struct MethodInfos
    {
        MethodInfos() = default;
        MethodInfos(string  n, U64 i) :metName(n), gpuIndex(i) { callTime = TimeGetMilliSec(); }
        string  metName;
        U64     gpuIndex = 0;
        U64     callTime = 0;
    };
    map<unsigned, MethodInfos> m_lastCalleJSondMethod; // ID -> methodCalled | extra

    struct PastWorkInfo
    {
        h256    hash;
        U64     recvTime;
        U32     unixTime;
    };
    std::vector<PastWorkInfo>   m_workBacklog;
    std::mutex                  m_workBacklogMutex;
    static const int            MaxBackLogCount = 6;
    
    std::mutex      m_currentWorkMutex;
    PascalWorkSptr  m_current; //these are clones of the one sent to the miners
    PascalWorkSptr  m_previous;//these are clones of the one sent to the miners

    string          m_jsonrpcVersion;
    string          m_sessionID; // pool->sessionid
    string          m_nonce1;
    unsigned        m_nonce2Size = 0;

    double          m_nextWorkDifficulty = 0.0f;
    U64             m_extraNonce = 0;
    U32             m_nonce2 = 0;
    U64             m_lastNonce2CleanTime = 0; 
    U64             m_lastSubmitTime = 0;
    std::mutex      m_stsMutex;
    U64             m_submittedTimestamp = 0;
    std::mt19937    m_nonce2Rand;

    //solo mining stuff
    U32         m_lastBlock = 0;
    U64         m_lastTimestamp = 0;
    U64         m_cleanTime = 0;
    U32         m_soloJobId = 0;

    virtual void ProcessMiningNotifySolo(Json::Value& arrayParam);
    virtual void RespondMiningSubmitSolo(Json::Value& stratumData, U32 gpuIndex);
};
typedef std::shared_ptr<StratumClient> StratumClientSptr;

////////////////////////////////// Utils

inline bool JsonGetSafeBool(Json::Value& responseObject, const char* field)
{
    try
    {
        return responseObject.get(field, false).asBool();
    }
    catch (...)
    {
        return false;
    }
}

inline string JsonGetSafeString(Json::Value& responseObject, const char* field)
{
    try
    {
        return responseObject.get(field, "").asString();
    }
    catch (...)
    {
        return "";
    }
}

inline Json::Value JsonGetSafeArray(Json::Value& responseObject, const char* field)
{
    try
    {
         return responseObject.get(field, Json::Value::null);
    }
    catch (...)
    {
        return Json::Value::null;
    }
}

