#!/bin/bash

for i in {200..1000..200}
    do  
        python3 XGBoost.py $i
    done