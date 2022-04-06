#!/bin/bash

for i in {1..10..2}
    do
    
        python3 GRU\ Dense.py $i
        python3 LSTM\ Dense.py $i
    done

for i in {10..50..10}
    do
    
        python3 Dense\ GRU.py $i
        python3 Dense\ LSTM.py $i
    done
