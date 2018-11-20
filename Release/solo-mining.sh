#!/bin/bash
while true
do
    ./rhminer -s http://localhost:4009 -cpu -cputhreads 4 -r 5
    sleep 5s
done




