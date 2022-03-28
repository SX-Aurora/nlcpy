#!/bin/bash


#PYTHONPATH=./ python tests/exe_all_ut.py
if [ ! -e tests/NEC_internal/result/ut.log ]; then
    mkdir -p tests/NEC_internal/result
    touch tests/NEC_internal/result/ut.log
fi

STR="Error at"
PYTHONPATH=`pwd` python -u tests/NEC_internal/exe_all_ut.py 2>&1 |tee >(grep "$STR" > tests/NEC_internal/result/ncc_intrinsic-check.log) |grep -v "$STR" |& tee tests/NEC_internal/result/ut.log

