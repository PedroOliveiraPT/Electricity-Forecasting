#!/bin/bash

for i in {10..90..10}
    do
    
        #python3 LSTM\ script.py $i
        python3 GRU\ script.py $i
    done
