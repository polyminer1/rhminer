# rhminer update and bugfix history

Nov 13 2008
* Fixed processproprity bug on windows
* Clarify and update documentation
* change -disabledevfee to -devfee so good souls can donate more than 1%
* fixed miniweb exception when running more than one instance
* change agent name and display title so it contain cuda arch
* output available logical core on CPU, when calling -completelist
* fixed reconnect bug when wallet crashes

Nov 20 2008
* Graceful handling of recurrent assert (m_fullHeader.size() == PascalHeaderSize)
* Add work timeout so miner exits if pool or wallet is stalled. see -worktimeout option.
* better handling of wallet/pool disconnection
* Removed temperature output. (Will be done later on.)
* Corrected share/block output stats depending if you mining locally or on pool
* Devfee watchdog timer to assert devfee time in all possible situations
* Updated windows/linux script example. Now linux script works properly.