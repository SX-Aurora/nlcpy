#!/bin/sh

EXE=eval.py

SHAPES=(xa xya xyza)

# Number of threads on VE
VE_NT=8

# Benchmark parameters
NX=1000
NY=1000
NZ=1000
IT=10
N_MIN=1
N_MAX=4

if [ ! -d result ]; then
    mkdir result    
fi

sh clean.sh

for SHAPE in ${SHAPES[@]}; do
    for ((i=$N_MIN; i <= $N_MAX; i+=1)); do
        script -qc "VE_OMP_NUM_THREADS=$VE_NT python $EXE -nx $NX -ny $NY -nz $NZ -n $i -it $IT -stencil_shape $SHAPE --numba --nlcpy_naive --nlcpy_sca" /dev/null
    done
done

python gen_graph.py
