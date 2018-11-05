/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Worker.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * @author Polyminer1 <https://github.com/polyminer1>
 * @date 2018
 */


#include "precomp.h"
#include "corelib/Worker.h"
#include "corelib/Log.h"

extern bool g_appActive;

using namespace std;

//MICRO_RTTI_DEFINE(Worker);
void Worker::StartWorking()
{
    RHMINER_RETURN_ON_EXIT_FLAG();

	Guard l(m_workMutex);
	if (m_work)
	{
		WorkerState ex = WorkerState::Stopped;
		m_state.compare_exchange_strong(ex, WorkerState::Starting);
	}
	else
	{
		m_state = WorkerState::Starting;
		m_work.reset(new thread([&]()
		{
            setThreadName(m_name.c_str());
			while (m_state != WorkerState::Killing)
			{
				WorkerState ex = WorkerState::Starting;
				m_state.compare_exchange_strong(ex, WorkerState::Started);
				try
				{
					WorkLoop();
				}
				catch (std::exception const& _e) 
				{
					RHMINER_PRINT_EXCEPTION_EX("Worker Exception ", _e.what());
                    QueueStopWorker();
				}

				ex = m_state.exchange(WorkerState::Stopped);
				if (ex == WorkerState::Killing || ex == WorkerState::Starting)
					m_state.exchange(ex);

				while (m_state == WorkerState::Stopped)
                {
					this_thread::sleep_for(chrono::milliseconds(20));
                }
			}
		}));
	}
	while (m_state == WorkerState::Starting)
		this_thread::sleep_for(chrono::microseconds(20));
}

void Worker::StopWorking()
{
	DEV_GUARDED(m_workMutex)
	if (m_work)
	{
		WorkerState ex = WorkerState::Started;
		m_state.compare_exchange_strong(ex, WorkerState::Stopping);

		while (m_state != WorkerState::Stopped)
			this_thread::sleep_for(chrono::microseconds(20));
	}
}

void Worker::Kill()
{
    DEV_GUARDED(m_workMutex)
	if (m_work)
	{
		m_state.exchange(WorkerState::Killing);
        {
		    m_work->join();
		    m_work.reset();
        }
	}
}

Worker::~Worker()
{
    Kill();
}
