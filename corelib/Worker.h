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

#pragma once

#include <cassert>
#include "Guards.h"
#include "corelib/basetypes.h"

enum class WorkerState
{
	Starting,
	Started,
	Stopping,
	Stopped,
	Killing
};

class Worker
{
public:
	Worker(std::string const& _name): m_name(_name) {}

	Worker(Worker const&) = delete;
	Worker& operator=(Worker const&) = delete;

	virtual ~Worker();

	/// Starts worker thread; causes startedWorking() to be called.
	virtual void StartWorking();
    virtual void Kill();
	
	/// Stop worker thread; causes call to StopWorking().
	void StopWorking();
    void QueueStopWorker() { m_state.exchange(WorkerState::Stopping); }
    bool isStopped() { return m_state == WorkerState::Stopped; }
    bool isStarted() { return m_state == WorkerState::Started; }

	bool shouldStop() const { return m_state != WorkerState::Started; }

    std::thread::id GetWorkerTID() { return m_work->get_id(); }

protected:
    //the entire work loop is to be defined by derivators
    virtual void WorkLoop() { RHMINER_EXIT_APP(""); }

	std::string m_name;

	mutable Mutex m_workMutex;						///< Lock for the network existance.
	std::unique_ptr<std::thread> m_work;		///< The network thread.
	std::atomic<WorkerState> m_state = {WorkerState::Starting};
};

