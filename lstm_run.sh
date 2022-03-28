#!/bin/bash

for i in {10..100..10}
    do
    for j in 15 30 45 60
        do 
	echo "doing $i $j"
        python3 LSTM\ script.py $j $i
        done
    done
