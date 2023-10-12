BASEDIR = $(CURDIR)

include make.inc

.PHONY: all arg_cehck nlcpy_ve_common nlcpy_ve_no_fast_math nlcpy_ve_fast_math FORCE

ifeq ($(JOBS),)
JOBS:=$(shell grep -c ^processor /proc/cpuinfo 2>/dev/null)
JOBS:=$(shell if [ $(JOBS) -le 16 ]; then echo "8"; elif [ $(JOBS) -le 32 ]; then echo "16"; else echo "32"; fi)
endif

ifeq ($(FTRACE),yes)
else
FTRACE=no
endif

ifeq ($(DEBUG),yes)
else
DEBUG=no
endif

all: make.dep nlcpy_ve_common nlcpy_ve_no_fast_math nlcpy_ve_fast_math embed_build_info

arg_check:
ifeq ($(ARCH),ve1)
else ifeq ($(ARCH),ve3)
else
	$(error ARCH is not defined or has an invalid value: $(ARCH))
endif

make.dep: arg_check $(OBJDIR_COMMON) $(OBJDIR_NO_FAST_MATH) $(OBJDIR_FAST_MATH)
	cd $(OBJDIR_COMMON) && sh $(TOOLDIR)/make_dep.sh
	cd $(OBJDIR_NO_FAST_MATH) && sh $(TOOLDIR)/make_dep.sh
	cd $(OBJDIR_FAST_MATH) && sh $(TOOLDIR)/make_dep.sh

nlcpy_ve_common: arg_check
	cd $(OBJDIR_COMMON) && make -f Makefile perl
	cd $(OBJDIR_COMMON) && make -j$(JOBS) -f Makefile COMMON=yes FTRACE=$(FTRACE) ARCH=$(ARCH) DEBUG=$(DEBUG)

nlcpy_ve_no_fast_math: arg_check nlcpy_ve_common
	cd $(OBJDIR_NO_FAST_MATH) && make -f Makefile perl
	cd $(OBJDIR_NO_FAST_MATH) && make -j$(JOBS) -f Makefile FTRACE=$(FTRACE) ARCH=$(ARCH) DEBUG=$(DEBUG)

nlcpy_ve_fast_math: arg_check nlcpy_ve_common
	cd $(OBJDIR_FAST_MATH) && make -f Makefile FAST_MATH=yes perl
	cd $(OBJDIR_FAST_MATH) && make -j$(JOBS) -f Makefile FAST_MATH=yes FTRACE=$(FTRACE) ARCH=$(ARCH) DEBUG=$(DEBUG)

embed_build_info: arg_check nlcpy_ve_common
	cd $(OBJDIR_COMMON) && make -f Makefile embed_build_info

$(OBJDIR_COMMON): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@
	cd $(INCDIR) && $(CPIO_INC) $@

$(OBJDIR_NO_FAST_MATH): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@
	cd $(INCDIR) && $(CPIO_INC) $@

$(OBJDIR_FAST_MATH): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@
	cd $(INCDIR) && $(CPIO_INC) $@

clean:
	sh scripts/clean.sh $(BASEDIR)
	rm -rf obj_ve1/
	rm -rf obj_ve3/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .tox
	rm -rf nlcpy_ve1_kernel/*.so
	rm -rf nlcpy_ve3_kernel/*.so

clean_cython:
	sh scripts/clean.sh $(BASEDIR)

clean_coverage:
	rm -rf .coverage .coverage.* htmlcov
