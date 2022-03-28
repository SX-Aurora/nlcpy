#!/bin/sh

BASEDIR=$1

cd ${BASEDIR}

find . -type d -name __pycache__ | xargs rm -rf

FILES=`find . -type f -name "*.pyx"`
for f_pyx in $FILES; do
    dir=`dirname $f_pyx`
    f_prefix=$dir/`basename $f_pyx .pyx`
    rm -rf $f_prefix\.*\.so $f_prefix\.cpp $f_prefix\.c
done
