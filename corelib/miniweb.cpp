/**
 * MiniWeb
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
#include "MinersLib/Global.h"
#include "corelib/utils.h"
#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>


#if !defined(RH_SCREEN_SAVER_MODE)

extern string g_apiPW;
using boost::asio::ip::tcp;
typedef boost::shared_ptr<tcp::socket> socket_ptr;
static string g_webdata = "{}";
static string g_ethMandata = "{}";
std::mutex* g_miniwebMutex = new std::mutex;
extern int g_apiPort;

void SetMiniWebData(const string& data, const string& ethManData)
{
    g_miniwebMutex->lock();
    g_webdata = data;
    g_ethMandata = ethManData;
    g_miniwebMutex->unlock();
}

bool IsValidPassword(const string& inPW)
{
    if (g_apiPW.length())
        return inPW == g_apiPW;
    else
        return false;
}

void ProcessControlgpu(Json::Value responseObject, socket_ptr sock)
{
    try
    {
        Json::Value params = responseObject.get("params", Json::Value::null);
        if (!params.isArray())
            PrintOut("Remote Api Error. Wrong parameter\n");
        else
        {
            string g = params.get((Json::Value::ArrayIndex)0, "0").asString();
            string s = params.get((Json::Value::ArrayIndex)1, "0").asString();
            U32 gpu = ToUInt(g);
            U32 state = ToUInt(s);

            if (gpu < GpuManager::Gpus.size())
            {
                PrintOut("api.control_gpu %s %s\n", (state == 1) ? "enabling" : "disabling", GpuManager::Gpus[gpu].gpuName.c_str());

                if (state == 1)
                    GpuManager::Gpus[gpu].enabled = true;
                if (state == 0)
                    GpuManager::Gpus[gpu].enabled = false;
            }
            else if ((int)gpu == -1)
            {
                PrintOut("api.control_gpu %s all GPU\n", (state == 1) ? "enabling" : "disabling");
                for (auto& g : GpuManager::Gpus)
                {
                    if (state == 1)
                        g.enabled = true;
                    if (state == 0)
                        g.enabled = false;
                }
            }
            else
                PrintOutCritical("api.control_gpu gpu ID invalid.\n");
        }
    }
    catch (...)
    {
        PrintOut("Error parsing api.control_gpu\n");
    }
}

void ProcessMinerFile(Json::Value responseObject, socket_ptr sock)
{
    try
    {
        Json::Value params = responseObject.get("params", Json::Value::null);
        if (!params.isArray())
            PrintOut("Remote Api Error. Wrong parameter\n");
        else
        {
            string fn = params.get((Json::Value::ArrayIndex)0, "").asString();
            string fc = params.get((Json::Value::ArrayIndex)1, "0").asString();
            Json::Value forceRestart = responseObject.get("forcerestart", "").asString();

            if (__stricmp(fn.c_str(), "config.txt") == 0 ||
                __stricmp(fn.c_str(), "config.xml") == 0)
            {
                bytes data = fromHex(fc);

                char basePath[1024];
                __getcwd(basePath, sizeof(basePath));
 
                strncat(basePath, __path_separator, sizeof(basePath)-1);

                strncat(basePath, fn.c_str(), sizeof(basePath)-1);
                FILE* f = fopen(basePath, "wb");
                if (f)
                {
                    PrintOut("Remote API Received %s\n", basePath);
                    fwrite(&data[0], data.size(), 1, f);
                    fclose(f);

                    GlobalMiningPreset::I().SetLastConfigFile(basePath);

                    //override command lines !
                    if (forceRestart == "1" || forceRestart == "true")
                    {
                        CmdLineManager::GlobalOptions().OverrideArgs(FormatString("-configfile %s", fn.c_str()));
                        GlobalMiningPreset::I().SetRestart(GlobalMiningPreset::eExternalRestart);
                    }
                }
                else
                    PrintOut("Error cannot create %s\n", basePath);
            }
        }
    }
    catch (...)
    {
        PrintOut("Error parsing api.control_gpu\n");
    }
}

void ProcessJsonApi(Json::Value responseObject, socket_ptr sock)
{
    try
    {
        string psw = responseObject.get("psw", "").asString();
        string method = responseObject.get("method", Json::Value::null).asString();
        if (method.length())
        {
            if (method == "miner_getstat1")
            {
                string webData;
                g_miniwebMutex->lock();
                webData = g_ethMandata;
                g_miniwebMutex->unlock();
                webData += "\n";
                boost::asio::write(*sock, boost::asio::buffer(webData, webData.length()));
            }
            else if (method == "miner_restart")
            {
                if (IsValidPassword(psw))
                    GlobalMiningPreset::I().SetRestart(GlobalMiningPreset::eExternalRestart);
                else
                    PrintOut("Remote Api Error. Wrong password\n");
            }
            else if (method == "miner_reboot")
            {
                if (IsValidPassword(psw))
                    GlobalMiningPreset::I().RequestReboot();
                else
                    PrintOut("Remote Api Error. Wrong password\n");
            }
            else if (method == "miner_file")
            {
                if (IsValidPassword(psw))
                {
                    ProcessMinerFile(responseObject, sock);
                }
                else
                    PrintOut("Remote Api Error. Wrong password\n");
            }
            else if (method == "control_gpu")
            {
                if (IsValidPassword(psw))
                {
                    ProcessControlgpu(responseObject, sock);
                }
                else
                    PrintOut("Remote Api Error. Wrong password\n");
            }

        }
        else
            PrintOut("Remote Api Error. unsupported method\n");
    }
    catch (...)
    {
        PrintOut("Remote Api Error. Json error\n");
    }
}


void MiniWeb_Connection(socket_ptr sock)
{
    try
    {
        const int MaxBuffer = 1024*16;
        char data[MaxBuffer];
        *data = 0;
        CpuSleep(100);
        boost::system::error_code error;
        size_t length = sock->read_some(boost::asio::buffer(data), error);
        if (length == MaxBuffer)
            length--;

        data[MaxBuffer - 1] = 0;
        data[length + 1] = 0;

        PrintOutSilent("Remote API:  %s\n", data);

        bool validJson = false;
        Json::Value responseObject;
        try
        {
            Json::Reader reader;
            validJson = reader.parse(data, responseObject);
        }
        catch (...)
        {
            validJson = false;
        }

        if (validJson)
        {
            ProcessJsonApi(responseObject, sock);
        }
        else
        {
            string webData;
            g_miniwebMutex->lock();
            webData = g_webdata;
            g_miniwebMutex->unlock();
            webData += "\n";
            boost::asio::write(*sock, boost::asio::buffer(webData, webData.length()));
        }
    }
    catch (std::exception& e)
    {
        PrintOut("Exception in MiniWeb thread: %s\n", e.what());
    }
}

U32 shutdownMiniWeb = false;
tcp::acceptor* acceptor = 0;
void MiniWeb_Server(boost::asio::io_service& io_service, unsigned short port)
{
    acceptor = new tcp::acceptor(io_service, tcp::endpoint(tcp::v4(), port));

    PrintOut("Remote API started on port %d\n", port);
    for (;;)
    {
        socket_ptr sock(new tcp::socket(io_service));
        acceptor->accept(*sock);
        if (AtomicGet(shutdownMiniWeb))
            break;

        boost::thread t(boost::bind(MiniWeb_Connection, sock));
        t.join();
    }
}

void StartMiniWeb()
{
    std::thread* f = new std::thread([&]()
    {
        while(1)
        {
            try
            {
                boost::asio::io_service io_service;
                MiniWeb_Server(io_service, g_apiPort);
                if (AtomicGet(shutdownMiniWeb))
                    return;
            }
            catch (std::exception& e)
            {
                PrintOut("Exception in miniweb: %s\n", e.what());
                return;
            }
            CpuSleep(1000);
        }
    });
}


void StopMiniWeb()
{
    AtomicSet(shutdownMiniWeb, 1);
    if (acceptor)
    {
        try
        {
            acceptor->cancel();
            delete acceptor;
        }
        catch (...)
        {
        }
    }
}

#else
void StartMiniWeb() {}
void SetMiniWebData(const string& data, const string& ethManData){}
void StopMiniWeb() {}
#endif