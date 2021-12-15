#!/bin/sh

cd nlcpy/ve_kernel
FILES=`find . -name "*.m4" | xargs -I {} basename {}`

# echo $FILES
ext="macros.m4"

for f_m4 in $FILES; do
    if [ $f_m4 != "macros.m4" ]; then
        f_base=`basename $f_m4 .m4`
        m4 $f_m4 > $f_base
    fi
done
