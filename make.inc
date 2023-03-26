PYTHON=$(shell echo ${PYTHONPATH} | sed -e "s/\([^:]*\)\(.*\)/\1/")

# BASEDIR is a relative path to the base directory of source tree, e.g. ../..
# It must be set in Makefile (but not make.inc files).

SRCDIR=$(BASEDIR)/nlcpy/ve_kernel
INCDIR=$(BASEDIR)/nlcpy/include
TOOLDIR=$(BASEDIR)/scripts
DISTDIR=$(BASEDIR)/dist
ifeq ($(ARCH),ve1)
OBJDIR_COMMON=$(BASEDIR)/obj_ve1/common
OBJDIR_NO_FAST_MATH=$(BASEDIR)/obj_ve1/no_fast_math
OBJDIR_FAST_MATH=$(BASEDIR)/obj_ve1/fast_math
LIBDIR=$(BASEDIR)/nlcpy_ve1_kernel
BUILD_VER_FILE=$(LIBDIR)/build_info.py
else ifeq ($(ARCH),ve3)
OBJDIR_COMMON=$(BASEDIR)/obj_ve3/common
OBJDIR_NO_FAST_MATH=$(BASEDIR)/obj_ve3/no_fast_math
OBJDIR_FAST_MATH=$(BASEDIR)/obj_ve3/fast_math
LIBDIR=$(BASEDIR)/nlcpy_ve3_kernel
BUILD_VER_FILE=$(LIBDIR)/build_info.py
endif


CPIO = find . | cpio -padlum
CPIO_INC = find . -name "*.h" | cpio -padlum
