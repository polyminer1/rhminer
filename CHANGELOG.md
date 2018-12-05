# rhminer update and bugfix history

Version 1.1 - 30 Nov 2018
* Optimization up to 7% faster on cuda (Tested on gtx 1070 8gb)
* Optimization up to 5% faster on cpu (Tested on Xenon and i5 7th)
* Fixed "unknown error" when disconnected from pool

Version 1.0 - 30 Nov 2018
* Fixed network error with coinotron and f2pool where miner was in limbo after pool disconnected.
* Fixed miner not starting on cpu without SSe4.1
* Fixed error where miners was rejecting shares due to wrong work package data.
* Fixed 'Exit on too many rejected shares' that was not working properly
* Added SSe detection
* Fixed connection error when coinotron and f2pool randomly closing socket.
* Minor memory optimizations

Version 0.9.4 - Nov 27 2018
* Fixed wrong hashrate on linux. (a thread-concurency bug made H/S show-up higer on linux.) 
* Minor code optimizations (up to 5% on certain system, more when not mining on all cores)
* Fix failover server feature
* Fixed "no work timeout" that was exiting application instead of reconnecting to server.
* Fixed testperformance feature so it show pure hash performance instead of raw simulation of solo/pool mining.
* Fixed some typos
* Fixed processpriority 2 that made Windows GUI lags. Performances may reduce, over time, if this processpriority is chosen. NOTE: New default processpriority is now 3.
* Added processpriority 3. All miner threads are on high priority. Give a gain on certain system. Can create GUI lags.
* Added "zero speed" watchdog. Miner will exit when speed is at zero for 60 sec
* Added More stable hashrate
* Added Faster work transfer time to miner threads
* Added Faster share submition time
* Change Speed output to show SMA20 of past raw speed readings


Version 0.9.3 - Nov 20 2018
* Graceful handling of recurrent assert (m_fullHeader.size() == PascalHeaderSize)
* Add work timeout so miner exits if pool or wallet is stalled. see -worktimeout option.
* better handling of wallet/pool disconnection
* Removed temperature output. (Will be done later on.)
* Corrected share/block output stats depending if you mining locally or on pool
* Devfee watchdog timer to assert devfee time in all possible situations
* Updated windows/linux script example. Now linux script works properly.

Version 0.9.2 - Nov 13 2018
* Fixed processproprity bug on windows
* Clarify and update documentation
* change -disabledevfee to -devfee so good souls can donate more than 1%
* fixed miniweb exception when running more than one instance
* change agent name and display title so it contain cuda arch
* output available logical core on CPU, when calling -completelist
* fixed reconnect bug when wallet crashes

