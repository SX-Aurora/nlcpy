#!/bin/bash

set +H

file=$1
CURDIR=$(cd $(dirname $0); pwd)
template=$(cat ${CURDIR}/dox_template | perl -pe 's/\n/#dox_n#/g')

dox=${file}.dox
cp ${file} ${dox}
grep -P "def ((?!_).+)(:|,)$" $file | while IFS= read func;
do
 indent=$(echo "$func"|perl -lne 'print $1 if /^(\s*)def/')
 ins_var=$(echo "$template"|sed -e "s/#dox_n#/#dox_n#${indent}/g" -e 's/\s*$//')
 func=${func/\*/\\\*}
 cat ${dox} | sed "s/^\(\s*\)\(${func}\)$/${indent}${ins_var}\1\2/g" > ${dox}.tmp
 mv ${dox}.tmp ${dox}
done

sed -i 's/#dox_n#/\
/g' ${dox}

set -H
