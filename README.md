# rhminer 

Miner for RandomHash2 POW algorithm.<br>
Support Intel/AMD 64 bit CPU.<br>
Support stratum and solo mining<br>
Works on Windows 7,10, Ubuntu 16/18, and MacOS
Also comes as a ScreenSaver on windows.

## Download prebuilt binaries
**Current version is 2.3** <br>
Support Windows7, Windows 10, Ubuntu 16/18, MacOS <br>
There is one prebuilt binariy per OS <br>
https://github.com/polyminer1/rhminer/releases/<br>

## Alternative download site : 
* Windows binaries https://mega.nz/#F!DqpAjCJQ!Q12a_YRlu_CWA92kIglKug
* Linux binaries https://mega.nz/#F!Dz4ElAwK!gbWbU4OpmEf6YnOCLIKfSQ


## Mining locally/Solo
To mine locally/solo you'll need the official PascalCoin wallet https://github.com/PascalCoin/PascalCoin/releases <br>
In order to mine locally with rhminer, **You need to set a miner name smaller than 26 characters and mine into a private key with encryption type secp256k1**<br>
The best way to assure you're mining in a secp256k1 private key is to create one and select it in the option **"Always mine with this key"**.<br>
**Do not use the "Use random existing key" option** because if one of your key is not a secp256k1 key, the miner will exit when. Plus when there is to much keys in the wallet it gives out errors, sometimes, when submiting nonces<br>
To ensure your miner name is correct, go to Project menu, then Options and set a miner name smaller than 26 characters<br>
To get the number of logical cores, on you system, simply run rhminer with the -completelist option. The last line is the cpu description with the amount of logical core. Ex :<br>
```
C:\rhminer>rhminer -completelist

  rhminer v1.5.3 beta for CPU by polyminer1 (https://github.com/polyminer1/rhminer)
  Buid CPU Nov 19 2018 20:04:01

  Donations : Pascal account 529692-23
  Donations : Bitcoin address 19GfXGpRJfwcHPx2Nf8wHgMps8Eat1o4Jp

CPU : Intel(R) Core(TM) i5-4460 CPU @ 3.20GHz with 4 logical cores
```
This tells you what is the ideal maximum number of threads for cpu mining (-cputhreads) <br>

```
Solo mining examples:  rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4009 -cpu -cputhreads 1 -extrapayload HelloWorld

```
Note2: It is not recommended to mine using a laptop.

## Tested on 
CPU: I3, I5, Core2, Xeon, Macbook <br>

## Performances for V2.1

| Cpu/GPU                               |  OS       | Threads   | Speed in H/s  | Extra infos   |
| --------------------------------------|-----------|----------:|--------------:|---------------|
|  i7-8750H CPU @ 2.20GHz               | win10     |   12      |   94,222      |               |
|  i5-3337U CPU @ 1.80GHz               | Win10     |    4      |   23,450      |               |
|  Ryzen 1800x @ 3.67 GHz               | Win10     |   15      |  150,000      |               |

## Performances for V2.0

| Cpu/GPU                               |  OS       | Threads   | Speed in H/s  | Extra infos   |
| --------------------------------------|-----------|----------:|--------------:|---------------|
|  i7-8750H CPU @ 2.20GHz               | win10     |   12      |   56,742      |               |
|  i5-3330S 2.7ghz                      | Win10     |  Max      |   22,230      |               |
|  i5-3337U CPU @ 1.80GHz               | Win10     |    4      |   14,539      |               |
|  Ryzen 1800x @ 3.67GHz                | Win10     |   16      |   96,000      |               |
|  Core(TM) 2 QuadCore Q6600 @ 2.40GHz  | Win7      |    2      |    6,835      |               |


**NOTE: I do not recommend to overclock your cpu. If you do it, it's at your own risk.**


note: raw is for raw performance on all hyper-threads. This does not represent real life performance.

## ScreenSaver
To download the screensaver go to release section here https://github.com/polyminer1/rhminer/releases and download **PascalCoinScreenSaver.zip** <br>
To install PascalCoin ScreenSaver simply right-click on file PascalCoinScreenSaver.scr and click "install" from the menu. <br>
Then you can configure it. <br>
For Laptop users it is *STROGLY* recommented to set only 1 thread in the scrensaver's config. <br>
 <br>
To set a mining password open regedit.exe and append your password command line, followed by a space, to the string located here : **Computer\HKEY_CURRENT_USER\Software\PascalCoin\ScreenSaver\extra** <br>
EX: -pw MyEmail@email.com <br>
Dont forget to put a space at the end. <br>

## Troubleshoot
On Windows 7/8/10, if you get the missing OpenCL.dll error you need to download it into rhminer's folder. (hint: You can safely get one with the Intel SDK on Intel's opencl website)


## Command line options
```
General options:
  -maxsubmiterrors      Stop the miner when a number of consecutive submit errors occured.
                        Default is 10 consecutive errors.
                        This is usefull when mining into local wallet.
  -extrapayload         An extra payload to be added when submiting solution to local wallet.
  -apiport              Tcp port of the remote api.
                        Default port is 7111.
                        Set to 0 to disable server.
                        Port is read-only by default. See API.txt for more informations
  -apipw                Api password for non read-only (miner_restart, miner_reboot, control_gpu, ..).
                        Default password is empty (read-only mode).
                        Note: must match ethman password
  -worktimeout          No new work timeout. Default is 60 seconds
  -displayspeedtimeout  Display mining speeds every x seconds.
                        Default is 10
  -logfilename          Set the name of the log's filename.
                        Note: the log file will be overwritten every time you start rhminer
  -configfile           Xml config file containing all config options.
                        All other command line options are ignored if config file given.
  -processpriority      On windows only. Set miner's process priority.
                        0=Background Process, 1=Low Priority, 2=Normal Priority, 3=High Priority.
                        Default is 3.
                        NOTE:Background Proces mode will make the console disapear from the desktop and taskbar. WARNING: Changing this value will affect GPU mining.
  -v                    Log verbosity. From 0 to 3.
                        0 no log, 1 normal log, 2 include warnings. 3 network and silent logs.
                        Default is 1
  -list                 List all gpu in the system
  -completelist         Exhaustive list of all devices in the system
  -diff                 Set local difficulyu. ex: -diff 999
  -processorsaffinity   On windows only. Force miner to only run on selected logical core processors.
                        ex: -processorsaffinity 0,3 will make the miner run only on logical core #0 and #3.
                        WARNING: Changing this value will affect GPU mining.
  -h                    Display Help
  -help                 Display Help
  -?                    Display Help

Optimizations options:
  -memoryboost          This option will enable some memory optimizations that could make the miner slower on some cpu.
                        Test it with -testperformance before using it.
                        1 to enable boost. 0 to disable boost.
                        Enabled, by default, on cpu with hyperthreading.
  -sseboost             This option will enable some sse4 optimizations.
                        It could make the miner slower on some cpu.
                        Test it with -testperformance before using it.
                        1 to enable SSe4.1 optimizations. 0 to disable.
                        Disabled by default. 
  -cputhrottling        Slow down mining by internally throttling the cpu. 
                        This is usefull to prevent virtual computer provider throttling vCpu when mining softwares are detected.
                        Min-Max are 0 and 99.

Gpu options:
  -cpu                  Enable the use of CPU to mine.
                        ex '-cpu -cputhreads 4' will enable mining on cpu while gpu mining.
  -cputhreads           Number of CPU miner threads when mining with CPU. ex: -cpu -cputhreads 4.
                        NOTE: adding + before thread count will disable the maximum thread count safety of one thread per core/hyperthread.
                        Use this option at your own risk.

Network options:
  -s                    Stratum/wallet server address:port.
                        NOTE: You can also use http://address to connect to local wallet.
  -su                   Stratum user
  -pw                   Stratum password
  -fo                   Failover address:port for stratum or local wallet
  -fou                  Failover user for stratum of a local wallet
  -fop                  Failover password for stratum or local wallet
  -r                    Retries connection count for stratum or local wallet
  -dar                  Disable auto-reconnect on connection lost.
                        Note : The miner will exit uppon loosing connection. 

Debug options:
  -testperformance      Run performance test for an amount of seconds
  -testperformancethreads Amount of threads to use for performance test

```

## Examples
```
With config file:
 First use : Edit config.txt and set "s", "su" and desired "cputhreads" or "gputhreads"
 
 Mining with default config.txt : rhminer.exe
 Mining with specific config file : rhminer.exe -configfile {config file pathname}

 With command line:
 Mining solo on cpu          : rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4009 -cpu -cputhreads 4 -extrapayload HelloWorld
 Mining solo on cpu          : rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4009 -cpu -cputhreads 4 -extrapayload HelloWorld
 Mining on a pool            : rhminer.exe -v 2 -r 20 -s stratum+tcp://somepool.com:1379 -su MyUsername -extrapayload Rig1
```

## Api access
Default port is 7111.  <br>
Api supports EthMan api format and structure with some limitations. <br>
- At the exception of 'miner_getstat1', the rest of the api needs a pw that match what was passed to rhminer (see -apipw with rhminer and 'Password' field in EthMan)
- For security reasons, method "miner_file" ONLY accept config.txt and config.xml.
  - The config file must be under 8K
  - The config file must be an rhminer compatible config file containing xml data
  - With parameter 'forcerestart' the miner will restart uppon reception of the config file, no mather what was the command line given to rhminer orignialy.
- miner_getstat2 return same as miner_getstat1
- Fan and temperature data are all zero
<br>
To change miner's config remotly, using EthMan, you send a config.txt first then send a restart command to the miner. <br>
    
Just sending empty string will return mining status in json format like that:
```
{
	"infos": [
        {
			"name": "CPU",
			"threads": 2,
			"speed": 266,
			"accepted": 3,
			"rejected": 0,
			"temp": 0,
			"fan": 0
		}
	],
	"speed": 380,
	"accepted": 4,
	"rejected": 0,
	"failed": 0,
	"uptime": 91,
	"extrapayload": "",
	"stratum.server": "localhost:4109",
	"stratum.user": "",
	"diff": 0.00000049
}
```
For more details and informations see https://github.com/polyminer1/rhminer/blob/master/Release/API.txt <br>

## Developer Donation
Default donation is 1%. <br>
Donation is hardcoded in the binaries downloadable on gitgub. That is to recoup the 6 month it toke to R&D, develop, stabilize and optimize this miner and for the upcoming bug fixes and many upcoming optimizations. <br>
To disable donation download and compile locally, then use the -devfee option with chosen donation percentage. 0 will disable the donation. <br>

For direct donations:
  * Pascal wallet 529692-23


## Contacts
Discord user ID : polyminer1#8454
Discord channel : https://discord.gg/RVcEpF9 (PascalCoin discord server) <br>
Discord channel : https://discord.gg/Egz2bdS (polyminer1 discord server) <br>
Bitcointalk : https://bitcointalk.org/index.php?topic=5065304.0 <br>
Twitter https://twitter.com/polyminer1 <br>

 