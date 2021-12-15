#!/bin/sh

DIRS=`find . -name "clean.sh" | xargs -I {} dirname {}`
echo $FILES

for d in $DIRS; do
    (cd $d; sh clean.sh)
done
