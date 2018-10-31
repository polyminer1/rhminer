# rhminer 

RandomHash miner for the PascalCoin blockchain.<br>
Support Intel/AMD 64 bit CPU and NVidia GPU.<br>
Support stratum and solo mining<br>

## Download prebuilt binaries
There is one prebuilt binariy per OS and CUDA architectures. <br>
https://github.com/polyminer1/rhminer/blob/master/Release

## Supported Cuda architecture
* Kepler  GTX  700 series, Tesla K40/K80
* Maxwell GTX  900 series, Quadro M series, GTX Titan X
* Pascal  GTX 1000 series, Titan Xp, Tesla P40, Tesla P4,GP100/Tesla P100 – DGX-1
* Volta   GTX 1100 series (GV104), Tesla V100

## Tested on 
CPU: I3, I5, Core2, Xeon, Athlon <br>
GPU: GTX 950, GTX 1060, GTX 1070 <br>
CUDA: Linux CUDA 9.1, Windows CUDA 9.2 <br>

## Upcoming
* CPU optimization using SSe4 and AVX
* Adding more stability to GPU mining
* GPU optimizations
* curl Api for stats and control


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
```
To build for CUDA Pascal : cmake -DRH_CPU_ONLY=OFF -DRH_DEBUG_TARGET=OFF -DRH_CUDA_ARCH=Pascal --target all ..
To build for CPU only    : cmake -DRH_CPU_ONLY=ON -DRH_DEBUG_TARGET=OFF --target all ..
```
make all <br>


## Stability issues
Thre are some limitations on nvidia gpu to consider.

First, the kernel is not 100% stable in all settings. This mean you'll have to experiment to find the stable sweet spot in term of gputhreads. Maximum thread count does not mean maximum speed. Sometimes lower thread count will give you more stability and more speed also.

On multiple gpu rigs, it's NOT recommended to mine CPU at the same time. You'll have more kernel timeout error because the driver will lack cpu time.<br>
Also, it is recommented, on multiple GPU rigs, to run the miner in a loop in a batch file !

Best way to find stability sweetspot's thread count is to divide total memory mb by 4.4 and then by 2<br>
Exemple: On GTX 1060 3gb, 3000 / 8.8 /2  = 170 gpu threads. Use that as a starting point for the -gputhreads parameter.

## Troubleshoot
On Windows 7/8/10, if you get the missing OpenCL.dll error you need to download it into rhminer's folder. (hint: You can safely get one with the Intel SDK on Intel's opencl website)


## Command line options
```
General options:
  -list                 List all gpu in the system
  -diff                 Set local difficulyu. ex: -diff 0.832
  -logfilename          Set the name of the log's filename. Note: the log file will be overwritten every time you start rhminer
  -extrapayload         An extra payload to be added when submiting solution to local wallet.
  -displayspeedtimeout  Display mining speeds every x seconds. Default is 10
  -processpriority      Set miner's process priority. 0=Background Process, 1=Low Priority, 2=Normal Priority. Default is 2. WARNING: Changing this value will affect GPU mining.
  -v                    Log verbosity. From 0 to 3. 0 no log, 1 normal log, 2 include warnings. Default is 1
  -disabledevfee        Before disabling developer fees, consider that it takes time and energy to maintain, develop and optimize this software.
  -completelist         Exhaustive list of all devices in the system
  -processorsaffinity   Force miner to only run on selected core processors. ex: -processorsaffinity 0,3 will make the miner run only on core #0 and #3. WARNING: Changing this value will affect GPU mining.
  -maxsubmiterrors      Stop the miner when a number of consecutive submit errors occured. Default is 10 consecutive errors. This is usefull when mining into local wallet.

Gpu options:
  -cpu                  Enable the use of CPU to mine. ex '-cpu -cputhread 4' will enable mining on cpu while gpu mining.
  -cputhreads           Number of CPU miner threads when mining with CPU. ex: -cpu -cputhread 4
  -cpurounds            Number of round per CPU cpu threads. Default is 50. ex: -cpu -cpurounds do 100. Each cpu thread will 100 hashes at a time
  -gputhreads           Cuda thread count. ex: -gputhreads  100 launche 100 threads on selected gpu
  -gpu                  Enable indiviaual GPU by their index. GPU not in the list will be disabled. ex: -gpu 0,3,4.
  -kernelactivewayting  Enable active wayting on kernel run. This will raise cpu usage but bring more stability, specially when mining on multiple gpu. WARNING: This affect cpu mining

Network options:
  -dar                  Disable auto-reconnect on connection lost. Note : The miner will exit uppon loosing connection.
  -s                    Stratum server or wallet address:port. NOTE: You can also use http://address.xyz to connect to local wallet.
  -su                   Stratum user
  -pw                   Stratum password
  -fo                   Failover address:port for stratum or local wallet
  -fou                  Failover user for stratum of a local wallet
  -fop                  Failover password for stratum or local wallet
  -r                    Retries connection count for stratum or local wallet
```

## Examples
```
 Mining solo with on cpu          : rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4109 -cpu -cputhreads 4 -extrapayload HelloWorld
 Mining solo with on cpu and gpu  : rhminer.exe -v 2 -r 20 -s http://127.0.0.1:4109 -cpu -cputhreads 4 -gpu 0 -gputhreads 262 -extrapayload HelloWorld
 Mining on pool a pool with 6 gpu : rhminer.exe -v 2 -r 20 -s stratum+tcp://somepool.com:2222 -su MyUsername -gpu 0,1,2,3,4,5 -gputhreads 400,512,512,512,210,512 -extrapayload Rig1
```
  
  
## Donations
Default dev donation is 1%. That is to recoup the 6 month it toke to develop this miner, and to support bug fixes and many upcoming optimizations. To disable donation use disabledevfee option.

  * Pascal address 529692-23
  * Bitcoin address 19GfXGpRJfwcHPx2Nf8wHgMps8Eat1o4Jp


## Release checksums MD5
```
Linux:
-------------------------------
MD5 for rhminer for CPU is :
5092420042198639106c227557c40b38
-------------------------------
MD5 for rhminer for Kepler is :
711acdf4c57e879dd7c5bb05f545d747
-------------------------------
MD5 for rhminer for Maxwell is :
346ab4b200077d948de786c6a758ca97
-------------------------------
MD5 for rhminer for Pascal is :
623ab16821c84c234df97f55b359b4f3
-------------------------------
MD5 for rhminer for Volta is :
d8a6c2615c8b815b524df4af216994cd

Windows:
-------------------------------- 
MD5 for rhminer.exe CPU is  
d6483f7a81c23ef62529681da98bf463
-------------------------------- 
MD5 for rhminer.exe Kepler is  
a769de7d37bac88ec11954342123874c
-------------------------------- 
MD5 for rhminer.exe Maxwell is  
bfbc6a38969cfc3d717e74502fcd5268
-------------------------------- 
MD5 for rhminer.exe Pascal is  
e770e60ce12bdca47165af6c831753c4
-------------------------------- 
MD5 for rhminer.exe Volta is  
cf154b44ee517543dd074bd074864dc6
-------------------------------- 

```
