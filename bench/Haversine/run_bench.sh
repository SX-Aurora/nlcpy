#!/bin/sh

if [ $# -eq 0 ]; then
    TARG="--numpy --nlcpy"
elif [ $# -eq 1 ]; then
    if [ $1 = "numpy" ]; then
        TARG="--numpy"
    elif [ $1 = "nlcpy" ]; then
        TARG="--nlcpy"
    else
        echo "error: detected unknown target"
        exit 1
    fi
elif [ $# -eq 2 ]; then
    if [ $1 = "numpy" ] && [ $2 = "nlcpy" ]; then
        TARG="--numpy -- nlcpy"
    elif [ $1 = "nlcpy" ] && [ $2 = "numpy" ]; then
        TARG="--numpy --nlcpy"
    else
        echo "error: detected unknown target"
        exit 1
    fi
else
    echo "error: too many arguments"
    exit 1
fi


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
    script -qc "VE_OMP_NUM_THREADS=$VE_NT python $EXE -scale $i $TARG" /dev/null
done
