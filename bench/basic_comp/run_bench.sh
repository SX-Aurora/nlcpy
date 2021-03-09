#!/bin/sh

if [ ! -d result ]; then
    mkdir result
fi

python eval.py

python gen_graph.py
