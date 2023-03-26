#!/bin/sh

### usage ###
# VH only
# $ sh build_inplace.sh --targ vh
#
# VE1 and VE3
# $ sh build_inplace.sh --targ ve1,ve3
#
# VH and VE1 and VE3
# $ sh build_inplace.sh --targ vh,ve1,ve3
#############

python setup.py build_ext -i $@
