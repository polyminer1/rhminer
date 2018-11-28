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
#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

using boost::asio::ip::tcp;
typedef boost::shared_ptr<tcp::socket> socket_ptr;
static string g_webdata = "{}";
std::mutex* g_miniwebMutex = new std::mutex;

void SetMiniWebData(const string& data)
{
    g_miniwebMutex->lock();
    g_webdata = data;
    g_miniwebMutex->unlock();
}

void MiniWeb_Connection(socket_ptr sock)
{
    try
    {
    	const int MaxBuffer = 512;
        char data[MaxBuffer];
        *data = 0;
        CpuSleep(100);
        boost::system::error_code error;
        size_t length = sock->read_some(boost::asio::buffer(data), error);
		data[MaxBuffer-1] = 0;
        
        PrintOutSilent("api : %s\n", data);

        //if (stristr(data, "get"))
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
        PrintOut("MiniWeb: Exception in webthread: %s\n", e.what());
    }
}

void MiniWeb_Server(boost::asio::io_service& io_service, unsigned short port)
{
    tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
    PrintOut("MiniWeb: Webserver started on port %d\n", port);
    for (;;)
    {
        socket_ptr sock(new tcp::socket(io_service));
        a.accept(*sock);
        boost::thread t(boost::bind(MiniWeb_Connection, sock));
        t.join();
    }
}

extern int g_apiPort;
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
            }
            catch (std::exception& e)
            {
                PrintOut("MiniWeb: Exception in miniweb: %s\n", e.what());
                return;
            }
            CpuSleep(1000);
        }
    });
}
