#!/bin/bash

for i in {1..7..1}
    do
    for j in 15 30 45 60
        do 
	    v=$((2 ** $i)) 
	    echo "doing $v $j"
	    python3 GRU\ script.py $j $v
        done
    done
