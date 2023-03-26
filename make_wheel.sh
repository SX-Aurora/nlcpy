#!/bin/sh

### usage ###
# VH only
# $ sh make_wheel.sh --targ vh
#
# VE1 and VE3
# $ sh make_wheel.sh --targ ve1,ve3
#
# VH and VE1 and VE3
# $ sh make_wheel.sh --targ vh,ve1,ve3
#############

python setup.py bdist_wheel $@
