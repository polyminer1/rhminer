# rhminer update and bugfix history

Version 1.5.3 - 8 Jul 2019
* Added MacOS support and binaries. (Tested on High Sierra and El Capitan)
* Added internal cpu throttling option to slow down mining.
* Fixed: No more dulicate shares
* Fixed: EthMan API : sent config.txt was ignored when restarting miner on Linux.

Version 1.5.2 - 16 May 2019
* Fixed Xml parsing of config file. Some options where skipped.


Version 1.5.1 - 14 May 2019
* Support for CUDA 10.1 with drivers 419.x(Windows) and 418.x(Linux)
* Added suppot for Nvidia Turing architecture (GTX 20xx) Untested.
* Fixed : Cannot Disable gpu mining when using config.txton gpu binaries. 
  Now you can simply empty the value of "gpu" and "gputhreads" to disable gpu mining on gpu binaries. Ex: "gpu":"" and "gputhreads":""
* Little performance boost on linux binaries (CPU and GPU)
* Added missing screensaver source files
* change line ending in many files


Version 1.5 - 2 April 2019
* Added ETHman and AwesomeMiner support (miner_getstat1, miner_restart, miner_reboot, miner_file, control_gpu)
  It is strongly recommented to NOT put rhminer in a loop in a script if you intend to use EthMan or AwseomeMiner.
  The restart option will close the instance and restart a new one. So if rhminer is in a loop, it will start another new instance everytime.
* Added Pascal Coin ScreenSaver V1.0 - Windows
* Now starting rhminer without parametres will automatically use config.txt.
* Added support Maxwell architecture 5.3 (Tegra Tegra (Jetson) TX1 / Tegra X1) on windows
* Re-added Pascal architecture 6.0 (Quadro GP100, Tesla P100, DGX-1) that was mistakenly removed in 1.4
* Re-added Pascal architecture 6.2 ( Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2) that was mistakenly removed in 1.4


Version 1.4 - 3 march 2019
*Major gpu optimizations:
    Gtx 1060 3gb 280 threads +196%
    Gtx 1070 8gb 384 threads +218%
    gtx 950 2gb 140 threads +226% [linux]
* Fix : On linux, rhminer stall for ever when network is lost (cable unplug or switch/router crashes). 
* Fix : On very rare occasions, rhminer would stall at startup.
* Fix : Opensource pool bug where work-ID stuck to the same ID, creating many invalid shares.
* Fix : Duplicate/stale shares was high on cpu with with core count. 
* Add -configfile option with a config.xml example
* Add extra option (+) to -cputhreads to enable mining on more threads that system's core/hyperthreads count (ex: on a i5 with 4 core -threadcount +7 will mine on 7 threads instead of 4)

Version 1.3 - 11 Jan 2019
*Major CPU optimizations. Up to 260% on some CPU
* +155% on Xeon E52665
* +260% on Intel i5 2th gen
* +195% on Intel i5 4th gen 
* +250% on Intel Core 2 
* New commandline option -sseboost. This option will give up to 10% on some cpu. BUT can also make it slower on other. TEST it before using it, to assert it gives a boost to your cpu.

Version 1.2 - 13 dec 2018
* Major optimization on cuda miner (+32% on gtx 1070 8gb, +37% on gtx 1060 3gb, +40% on gtx 950)
* Simple optimization on Linux. Up to 5% depending on the cpu

Version 1.1.1 - 6 dec 2018
* Critical Fix for memory corruption on Linux
* Fixed cross server shares submission

Version 1.1 - 5 dec 2018
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

