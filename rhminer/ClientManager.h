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

#include "precomp.h"
#include "BuildInfo.h"
#include "MinersLib/Global.h"
#include "MinersLib/GenericMinerClient.h"
#include "MinersLib/StratumClient.h"

struct ActiveClientsData
{
    std::shared_ptr<GenericMinerClient> client;
    StratumClientSptr            stratum;
};

class ClientManager
{
public:
    static ClientManager& I();
    enum eShutdownMode { eShutdownLite, eShutdownRestart, eShutdownFull };

    void Initialize();
    void Shutdown(eShutdownMode esmode);

    ActiveClientsData     ActiveClients;
};
