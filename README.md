# rhminer 

RandomHash miner for the PascalCoin blockchain.<br>
Support Intel/AMD 64 bit CPU and NVidia GPU.<br>
Support stratum and solo mining<br>
Works on Windows 7,10 and Ubuntu 18

## Download prebuilt binaries
**Current version is 1.1** <br>

There is one prebuilt binariy per OS and CUDA architectures. <br>
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

  rhminer v1.1 beta for CPU by polyminer1 (https://github.com/polyminer1/rhminer)
  Buid CPU Nov 19 2018 20:04:01

  Donations : Pascal account 529692-23
  Donations : Bitcoin address 19GfXGpRJfwcHPx2Nf8wHgMps8Eat1o4Jp

CPU : Intel(R) Core(TM) i5-4460 CPU @ 3.20GHz with 4 logical cores
```
This tells you what is the ideal maximum number of threads for cpu mining (-cputhreads) <br>

```
Solo mining examples:

For Test net solo mining :  rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4109 -cpu -cputhreads 1 -gpu 0 -gputhreads 100 -extrapayload HelloWorld
For Main net solo mining :  rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4009 -cpu -cputhreads 1 -gpu 0 -gputhreads 100 -extrapayload HelloWorld

NOTE: remove -gpu 0 -gputhreads 100 if you dont have a gpu
```
Note2: It is not recommended to mine using a laptop.

## Supported Cuda architecture
* Kepler  GTX  700 series, Tesla K40/K80
* Maxwell GTX  900 series, Quadro M series, GTX Titan X
* Pascal  GTX 1000 series, Titan Xp, Tesla P40, Tesla P4,GP100/Tesla P100 – DGX-1
* Volta   GTX 1100 series (GV104), Tesla V100

## Gpu mining
To mine using gpu you must provide the gpu numbers and the amount of threads for each gpu.<br>
If you only have one gpu, use **-gpu 0** and **-gputhreads {amount of threads}**<br>
If you have more than one gpus, you can see their number by executing the miner with the *list* option :<br> 
``` 
C:>rhminer -list 

  rhminer v0.9 beta for CPU and NVIDIA GPUs by polyminer1 (http://github.com/polyminer1)
  NVIDIA CUDA SDK 9.2

  Donations : Pascal account 529692-23
  Donations : Bitcoin address 19GfXGpRJfwcHPx2Nf8wHgMps8Eat1o4Jp

List of gpus and cpus:
GPU0 : GeForce GTX 1060 3GB
GPU1 : GeForce GTX 1060 3GB
GPU2 : GeForce GTX 950 2GB
CPU  : Intel(R) Core(TM) i5-4460 CPU @ 3.20GHz
```
Then select the ones you want to mine with like this : <br> 
```
rhminer -s http://localhost:4009 -gpu 1,2 -gputhreads 262,102
```

## Ideal CUDA threads count
To find the ideal maximum amount of threads, start with 75% of the memory divided by 8.8. <br>
For a GTX 1060 3GB that is 3000 * 0.75 / 8.8 = 255 threads. <br>
Then run 2 minutes and if everything is stable, raise by say 32 until you get no crashes after 2 min.<br>
To help you in that process, look for the log line that say "CUDA: Using " when the miner starts. It  will indicate how much memory you take and how much is left depending on your selected thread count.<br>
**ALLWAYS** let at lease 150 meg of free memory, for internal OS operations, or you have stability issues.<br>


## Tested on 
CPU: I3, I5, Core2, Xeon, Athlon <br>
GPU: GTX 950, GTX 1060, GTX 1070 <br>
CUDA: Linux CUDA 9.1, Windows CUDA 9.2 <br>

## Upcoming
* CPU optimization using SSe4 and AVX
* Adding more stability to GPU mining
* GPU optimizations
* curl Api for stats and control

## Performances

| Cpu/GPU                            |  OS        | Threads          | Speed in H/s |
| -----------------------------------|------------|-----------------:|-------------:|
| Gtx 1060 3gb                       | Windows 10 | 262              | 109          |  
| Gtx 1060 6gb (mobile)              | Windows 10 | 570              | 199          |
| Gtx 1080 11gb                      | Ubuntu     | 400              | 400          |
| Gtx 950 2gb                        | Windows 7  | 140              | 52           |
| i7-8750H CPU @ 3.90Hz              | Ubuntu 18  | 11               | 1217         |
| Ryzen 1800X @ 4GHz                 | Windows 10 | 15               | 1241         |
| Ryzen 7 2700x 4.1GHz               | Ubuntu     | ?                | 1372         |
| Xeon x5650@ 2.67GHz                | Linux      | 24               | 716          |
| Xeon E5 2690 @ 2.4 GHz.            | Ubuntu 18  | 24               | 1347         |
| Xeon Platinum 8168@ 2.70GHz        | Ubuntu 18  | 32               | 3085         | 
| i7 7700K  @ 4.2Ghz                 | Windows 10 | 8                | 945          |
| i7-7500U CPU @ 2.70GHz             | Linux      | 400              | 339 raw      |
| i7 4770k @ 4.5GHz. Dual chan. DDR3 | Ubuntu     | 8/10             | 1100         |
| i7-3615QM CPU @ 2.30GHz            | Linux      | 8                | 675 raw      |
| i7 2600K  @ 3.4Ghz                 | Windows 10 | 8                | 506          |
| i7-4558U CPU @ 2.80GHz             | Linux Mint | 4                | 342          |
| i5-4460 CPU @ 3.20GHz              | Windows 10 | 4                | 511          |
| i5-3337U CPU @ 1.80GHz             | Windows 10 | 4                | 245          |
| i5-2400 CPU @ 3.10GHz              | Ubuntu 18  | 4                | 303          |
| i3-4030U CPU @ 1.90GHz             | Linux      | 400              | 201 raw      |
| Core(TM) 2 Duo 6300 @ 1.86GHz      | Windows 7  | 2                | 92           |
| Core(TM) 2 QuadCore Q6600@2.40GHz  | windows 7  | 4                | 163          |

note: raw is for raw performance on all hyper-threads. This does not represent real life performance.

## Build instructions (Windows)                      
Install VisualStudio 2017 with chose Platform Toolset v140 <br>
Install the lastest NVIDIA Display Driver <br>
Install the CUDA Toolkit 9.2 (or more) <br>
Install boost_1_64_0 and make libs using bjam (https://www.boost.org/doc/libs/1_64_0/more/getting_started/windows.html) <br>
Open solution, select target and compile <br>
Run <br>

## Build Linux (Ubuntu)
sudo apt-get install git build-essential cmake  <br>
install CUDA ToolKit <br>
Install and compile boost_1_64_0 <br>
Install jsoncpp <br>
git clone https://github.com/polyminer1/rhminer.git <br>
cd rhminer <br>
mkdir build <br>
cd build <br>
*To build for CUDA Pascal :* cmake -DRH_CPU_ONLY=OFF -DRH_DEBUG_TARGET=OFF -DRH_CUDA_ARCH=Pascal --target all ..  <br>
*To build for CPU only    :* cmake -DRH_CPU_ONLY=ON -DRH_DEBUG_TARGET=OFF --target all ..  <br>
make all <br>


## Stability issues
Thre are some limitations on nvidia gpu to consider.

First, the kernel is not 100% stable in all settings. This mean you'll have to experiment to find the stable sweet spot in term of gputhreads. Maximum thread count does not mean maximum speed. Sometimes lower thread count will give you more stability and more speed also.

On multiple gpu rigs, it's NOT recommended to mine CPU at the same time. You'll have more kernel timeout error because the driver will lack cpu time.<br>
Also, it is recommented, on multiple GPU rigs, to run the miner in a loop in a batch file !

## Troubleshoot
On Windows 7/8/10, if you get the missing OpenCL.dll error you need to download it into rhminer's folder. (hint: You can safely get one with the Intel SDK on Intel's opencl website)


## Command line options
```
General options:
General options:
  -maxsubmiterrors      Stop the miner when a number of consecutive submit errors occured. Default is 10 consecutive errors. This is usefull when mining into local wallet.
  -extrapayload         An extra payload to be added when submiting solution to local wallet.
  -apiport              Tcp port of the remote api. Default port is 71111. Set to 0 to disable server
  -worktimeout          No new work timeout. Default is 60 seconds
  -displayspeedtimeout  Display mining speeds every x seconds. Default is 10 
  -logfilename          Set the name of the log's filename. Note: the log file will be overwritten every time you start rhminer
  -processpriority      On windows only. Set miner's process priority. 0=Background Process, 1=Low Priority, 2=Normal Priority, 3=High Priority. Default is 3. WARNING: Changing this value will affect GPU mining.
  -v                    Log verbosity. From 0 to 3. 0 no log, 1 normal log, 2 include warnings. 3 network (only in log file). Default is 1
  -devfee               Set devfee raward percentage. To disable devfee, simply put 0 here. But, before disabling developer fees, consider that it takes time and energy to maintain, develop and optimize this software. Your help is very appreciated.
  -list                 List all gpu in the system
  -completelist         Exhaustive list of all devices in the system
  -diff                 Set local difficulyu. ex: -diff 0.832
  -processorsaffinity   On windows only. Force miner to only run on selected logical core processors. ex: -processorsaffinity 0,3 will make the miner run only on logical core #0 and #3. WARNING: Changing this value will affect GPU mining.
  -h                    Display Help

Gpu options:
  -cpu                  Enable the use of CPU to mine. ex '-cpu -cputhreads 4' will enable mining on cpu while gpu mining.
  -cputhreads           Number of CPU miner threads when mining with CPU. ex: -cpu -cputhreads 4

Network options:
  -dar                  Disable auto-reconnect on connection lost. Note : The miner will exit uppon loosing connection. 
  -s                    Stratum/wallet server address:port. NOTE: You can also use http://address.xyz to connect to local wallet.
  -su                   Stratum user
  -pw                   Stratum password
  -fo                   Failover address:port for stratum or local wallet
  -fou                  Failover user for stratum of a local wallet
  -fop                  Failover password for stratum or local wallet
  -r                    Retries connection count for stratum or local wallet

Debug options:
  -testperformance      Run performance test for an amount of seconds
```

## Examples
```
 Mining solo on cpu          : rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4009 -cpu -cputhreads 4 -extrapayload HelloWorld
 Mining solo on cpu and gpu  : rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4009 -cpu -cputhreads 4 -gpu 0 -gputhreads 262 -extrapayload HelloWorld
 Mining on a pool with 6 gpu : rhminer.exe -v 2 -r 20 -s stratum+tcp://somepool.com:1379 -su MyUsername -gpu 0,1,2,3,4,5 -gputhreads 400,512,512,512,210,512 -extrapayload Rig1
```

## Api access
Default port is 7111. Just sending empty string will return mining status in json format like that:
```
{
	"infos": [
        {
			"name": "GPU2",
			"threads": 262,
			"speed": 114,
			"accepted": 1,
			"rejected": 0,
			"temp": 0,
			"fan": 0
		}, 
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

  
## Developer Donation
Default donation is 1%. <br>
Donation is hardcoded in the binaries downloadable on gitgub. That is to recoup the 6 month it toke to R&D, develop, stabilize and optimize this miner and for the upcoming bug fixes and many upcoming optimizations. <br>
To disable donation download and compile locally, then use the -devfee option with chosen donation percentage. 0 will disable the donation. <br>

For direct donations:
  * Pascal wallet 529692-23
  * Bitcoin address 19GfXGpRJfwcHPx2Nf8wHgMps8Eat1o4Jp


## Contacts
Discord channel : https://discord.gg/RVcEpF9 (PascalCoin discord server) <br>
Discord channel : https://discord.gg/Egz2bdS (polyminer1 discord server) <br>
Bitcointalk : https://bitcointalk.org/index.php?topic=5065304.0 <br>
Twitter https://twitter.com/polyminer1 <br>

 