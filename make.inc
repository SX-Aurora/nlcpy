PYTHON=$(shell echo ${PYTHONPATH} | sed -e "s/\([^:]*\)\(.*\)/\1/")

# BASEDIR is a relative path to the base directory of source tree, e.g. ../..
# It must be set in Makefile (but not make.inc files).

SRCDIR=$(BASEDIR)/nlcpy/ve_kernel
TOOLDIR=$(BASEDIR)/scripts
DISTDIR=$(BASEDIR)/dist
OBJDIR_COMMON=$(BASEDIR)/obj/common
OBJDIR_NO_FAST_MATH=$(BASEDIR)/obj/no_fast_math
OBJDIR_FAST_MATH=$(BASEDIR)/obj/fast_math
LIBDIR=$(BASEDIR)/nlcpy/lib

CPIO = find . -name CVS -prune -o -print | cpio -padlum
