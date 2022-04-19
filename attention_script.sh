#!/bin/bash

for i in {3..8..1}
    do
    for j in {10..50..10}
        do 
	    v=$((2 ** $i)) 
	    python3 ML\ script.py AttentionBiLSTM $v $j
        done
    done
