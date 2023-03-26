#!/bin/sh

if [ $# -eq 0 ]; then
    TARG="--numpy --nlcpy --nlcpy-jit"
elif [ $# -eq 1 ]; then
    if [ $1 = "numpy" ]; then
        TARG="--numpy"
    elif [ $1 = "nlcpy" ]; then
        TARG="--nlcpy"
    elif [ $1 = "nlcpy-jit" ]; then
        TARG="--nlcpy-jit"
    else
        echo "error: detected unknown target"
        exit 1
    fi
elif [ $# -eq 2 ]; then
    if [ $1 = "numpy" ] && [ $2 = "nlcpy" ]; then
        TARG="--numpy --nlcpy"
    elif [ $1 = "nlcpy" ] && [ $2 = "numpy" ]; then
        TARG="--nlcpy --numpy"
    elif [ $1 = "numpy" ] && [ $2 = "nlcpy-jit" ]; then
        TARG="--numpy --nlcpy-jit"
    elif [ $1 = "nlcpy-jit" ] && [ $2 = "numpy" ]; then
        TARG="--nlcpy-jit --numpy"
    elif [ $1 = "nlcpy" ] && [ $2 = "nlcpy-jit" ]; then
        TARG="--nlcpy --nlcpy-jit"
    elif [ $1 = "nlcpy-jit" ] && [ $2 = "nlcpy" ]; then
        TARG="--nlcpy-jit --nlcpy"
    else
        echo "error: detected unknown target"
        exit 1
    fi
elif [ $# -eq 3 ]; then
    if [ $1 = "numpy"] && [ $2 = "nlcpy"] && [ $3 = "nlcpy-jit" ]; then
        TARG="--numpy --nlcpy --nlcpy-jit"
    elif [ $1 = "numpy" ] && [ $2 = "nlcpy-jit" ] && [ $3 = "nlcpy" ]; then
        TARG="--numpy --nlcpy-jit --nlcpy"
    elif [ $1 = "nlcpy" ] && [ $2 = "numpy" ] && [ $3 = "nlcpy-jit" ]; then
        TARG="--nlcpy --numpy --nlcpy-jit"
    elif [ $1 = "nlcpy" ] && [ $2 = "nlcpy-jit" ] && [ $3 = "numpy" ]; then
        TARG="--nlcpy --nlcpy-jit --numpy"
    elif [ $1 = "nlcpy-jit" ] && [ $2 = "numpy" ] && [ $3 = "nlcpy" ]; then
        TARG="--nlcpy-jit --numpy --nlcpy"
    elif [ $1 = "nlcpy-jit" ] && [ $2 = "nlcpy" ] && [ $3 = "numpy" ]; then
        TARG="--nlcpy-jit --nlcpy --numpy"
    else
        echo "error: detected unknown target"
        exit 1
    fi
else
    echo "error: too many arguments"
    exit 1
fi


EXE=eval.py

# Benchmark parameters
MIN_SCALE=$((2**6))
#MAX_SCALE=$(( $MIN_SCALE*$((2**10)) ))
MAX_SCALE=$(( $MIN_SCALE*$((2**12)) ))


if [ ! -d result ]; then
    mkdir result
fi

sh clean.sh

for ((i=$MIN_SCALE; i <= $MAX_SCALE; i*=2)); do
    script -qc "python $EXE -scale $i $TARG" /dev/null
done
