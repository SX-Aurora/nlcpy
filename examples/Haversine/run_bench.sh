#!/bin/sh

EXE=eval.py

# Number of threads on VE
VE_NT=8

# Benchmark parameters
MIN_SCALE=$((2**6))
MAX_SCALE=$(( $MIN_SCALE*$((2**10)) ))

if [ ! -d result ]; then
    mkdir result    
fi

sh clean.sh

for ((i=$MIN_SCALE; i <= $MAX_SCALE; i*=2)); do
    script -qc "VE_OMP_NUM_THREADS=$VE_NT python $EXE -scale $i --nlcpy --numpy" /dev/null
done

python gen_graph.py -inplace 1

