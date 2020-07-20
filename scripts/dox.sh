#!/bin/bash

BASEDIR=$1

cd ${BASEDIR}

CURDIR=${BASEDIR}
DOXYGENDIR=${CURDIR}/doc
DOXTOOLDIR=${DOXYGENDIR}/scripts
DOXEXE=doxygen

TARGET=doxygen
DOXYFILE=doxyfile

repl_forlist () {
for i in "${ref_list[@]}";do
  data=(${i[@]})
  before=${data[0]}
  list=${data[1]}
  grep -ril "${before}" ${TARGET}/$1/ | grep -E "group__[a-zA-Z\_\-]+.html$" | while read file;do
    sed -i "s/${before}/${list}/g" $file
  done
done
}

repl_fordetail () {
for i in "${ref_list[@]}";do
  data=(${i[@]})
  before=${data[0]}
  detail=${data[2]}
  grep -ril "${before}" ${TARGET}/$1/ | while read file;do
    sed -i "s/\([^\.]\)${before}/\1${detail}/g" $file;
  done
done
}

rm -rf ${DOXYGENDIR}/${TARGET}/*
mkdir -p ${DOXYGENDIR}/${TARGET}/en

echo "Start conversion dox to pydoc .."

cd ${DOXYGENDIR} && ([ -e ./src/nlcpy ] && rm -r ./src/nlcpy
mkdir -p ./src/nlcpy; cd $_

find ${CURDIR}/nlcpy/ -maxdepth 2 -mindepth 2 -name "*.dox" | while read -r file;
do
 tmpmod=`dirname $file`
 mod=${tmpmod##*/}
 [ ! -e ${mod} ] && mkdir -p $mod
 dox=`basename $file`
 cp ${file} ${mod}/${dox%.*} && 
 sed -i -e 's/cdef class\|cpdef\sclass/class/g' -e 's/cdef\|cpdef/def/g' -e 's/@raise/@li/g' -e 's/@sa\(.*\)\".*\"$/@sa\1/g' ${mod}/${dox%.*} &&
 sed -i '/@pydoc\|@endpydoc\|@rename\|@pydocmove/d' ${mod}/${dox%.*}
 perl ${DOXTOOLDIR}/doxycommand_to_docstring.pl ${file} ${DOXTOOLDIR}/func_reform_list 
done
# for ufunc
grep @fn ${DOXYGENDIR}/src/ufuncs.dox | sed -e s"/\(.*@fn\s*\)\(\w*\)\(\s*(.*\)/ufuncs.\2 \2 nlcpy.\2/" >> ${DOXTOOLDIR}/func_reform_list

# get reform list
OLDIFS=$IFS
IFS=$'\n'
ref_list=(`cat ${DOXTOOLDIR}/func_reform_list`)
IFS=$OLDIFS

cd $DOXYGENDIR

${DOXEXE} ${DOXYGENDIR}/${DOXYFILE}.en #> /dev/null

repl_forlist "en"

repl_fordetail "en"

ls ${TARGET}/en/*.html | while read file; do sed -i -e 's/def\&#/\&#/g' -e 's/<dt>Attention<\/dt>/<dt>Restrictions<\/dt>/g' -e '/a href=\"pages.html\"/d' ${file}
sed -i -e '/sync_off.png/d' -e '/sync_on.png/d' ${TARGET}/en/navtree.js
perl -0pi -e 's/<td class="paramtype">&#160;<\/td>\n          <td class="paramname"><em>self<\/em>, <\/td>\n        <\/tr>\n        <tr>\n          <td class="paramkey"><\/td>\n          <td><\/td>\n          //mg' ${file}
perl -0pi -e 's/<td class="paramtype">&#160;<\/td>\n          <td class="paramname"><em>self<\/em><\/td>//mg' ${file}
done

)
cd $CURDIR

cp ${DOXYGENDIR}/common/* ${DOXYGENDIR}/${TARGET}/en/.
echo "Done."
