#!/bin/sh

if [ ! -d result ]; then
    mkdir result    
fi

python eval.py

python extract_bad_op.py


