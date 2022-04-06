#!/bin/bash

for i in {2..5}
    do
    
        python3 Stack\ Dense\ GRU.py $i
        python3 Stacked\ Dense\ LSTM.py $i
    done

for i in {1..5}
    do
    
        python3 Conv\ GRU.py $i
        python3 Conv\ LSTM.py $i
    done