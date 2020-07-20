#!/usr/bin/bash
FILE=make.dep
for i in *.[ch] *.master
do 
(echo $i:; grep -h "#include" $i  |grep "\"" |perl -pe s"/(.*)\"(.*)\"(.*)/\2/" )| perl -pe 's/\n/ /g' ; echo 
done > $FILE
