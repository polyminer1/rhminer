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


#include "precomp.h"
#include "RandomHashCLMiner.h"
#include "MinersLib/Global.h"


constexpr size_t c_maxSearchResults = 1;

RandomHashCLMiner::RandomHashCLMiner(FarmFace& _farm, unsigned globalWorkMult, unsigned localWorkSize, U32 gpuIndex) :
    GenericCLMiner(_farm, globalWorkMult, localWorkSize, gpuIndex)
{
}

bool RandomHashCLMiner::init(const PascalWorkSptr& work)
{
    AddPreBuildFunctor([&](string& code) 
    {
        addDefinition(code, "PAS_FAST", 1);
    });

    auto res = GenericCLMiner::init(work);
    return res;
}

void RandomHashCLMiner::QueueKernel()
{
    GenericCLMiner::QueueKernel();
}
/*
SolutionSptr RandomHashCLMiner::MakeSubmitSolution(const std::vector<U64>& nonces, bool isFromCpuMiner)
{
    PascalSolution* sol = new PascalSolution();
    sol->m_results = nonces;
    sol->m_gpuIndex = m_globalIndex;
    sol->m_work = PascalWorkSptr(m_currentWp->Clone());
    sol->m_isFromCpuMiner = isFromCpuMiner;

    return SolutionSptr(sol);
}

*/
bool RandomHashCLMiner::configureGPU()
{
    return GpuManager::SetupGPU();
}



