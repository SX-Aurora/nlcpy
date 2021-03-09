#!/bin/sh

if [ $# -gt 1 ]; then
    echo "Too many arguments."
    exit 1
fi

if [[ $1 = "small" ]]; then
    echo "Start small tests."
    cd tests/pytest && pytest --test=small -x
elif [[ $1 = "full" ]]; then
    echo "Start full tests."
    cd tests/pytest && pytest --test=full -x
else
    echo "Start no ufunc tests."
    cd tests/pytest && pytest -x
fi
