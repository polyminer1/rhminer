#!/bin/bash
while true
do
    ./rhminer -s stratum+tcp://fastpool.xyz:10098 -su 1300378-87.0.Donations -cpu -cputhreads 4 -r 40
    sleep 5s
done




