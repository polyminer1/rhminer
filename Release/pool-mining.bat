a:
REM Mining on somepool.org with gpu 0 thru 5 with selective threadcount.
rhminer.exe -v 2 -r 20 -s stratum+tcp://somepool.com:1379 -su MyPoolUsername -gpu 0,1,2,3,4,5 -gputhreads 400,512,512,512,210,512
goto 1

