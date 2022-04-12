#!/bin/bash

for i in {10..100..10}
    do
        python3 ML\ Script.py SimpleLSTM $i
        python3 ML\ Script.py SimpleGRU $i
    done

for i in {10..70..10}
    do
        python3 ML\ Script.py DropoutLSTM $i
        python3 ML\ Script.py DropoutGRU $i
    done

for i in {1..10..2}
    do
    
        python3 ML\ Script.py LSTMDense $i
        python3 ML\ Script.py GRUDense $i
    done

for i in {10..50..10}
    do
    
        python3 ML\ Script.py DenseLSTM $i
        python3 ML\ Script.py DenseGRU $i
    done

for i in {2..5}
    do
    
        python3 ML\ Script.py StackedLSTM $i
        python3 ML\ Script.py StackedGRU $i
    done

for i in {1..5}
    do
    
        python3 ML\ Script.py ConvLSTM $i
        python3 ML\ Script.py ConvGRU $i
    done

