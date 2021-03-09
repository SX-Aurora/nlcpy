BASEDIR = $(CURDIR)

include make.inc

.PHONY: all nlcpy_ve_no_fast_math nlcpy_ve_fast_math doc FORCE

JOBS:=$(shell grep -c ^processor /proc/cpuinfo 2>/dev/null)
JOBS:=$(shell if [ $(JOBS) -le 16 ]; then echo "8"; else echo "16"; fi)

all: make.dep nlcpy_ve_no_fast_math nlcpy_ve_fast_math
#all: make.dep nlcpy_ve_no_fast_math

make.dep: $(OBJDIR_NO_FAST_MATH) $(OBJDIR_FAST_MATH)
	cd $(OBJDIR_NO_FAST_MATH) && sh $(TOOLDIR)/make_dep.sh
	cd $(OBJDIR_FAST_MATH) && sh $(TOOLDIR)/make_dep.sh

nlcpy_ve_no_fast_math:
	cd $(OBJDIR_NO_FAST_MATH) && make -f Makefile perl
	cd $(OBJDIR_NO_FAST_MATH) && make -j$(JOBS) -f Makefile
#	cd $(OBJDIR_NO_FAST_MATH) && make -j$(JOBS) -f Makefile FTRACE=yes # requirement veos >= 2.4.1

nlcpy_ve_fast_math:
	cd $(OBJDIR_FAST_MATH) && make -f Makefile FAST_MATH=yes perl
	cd $(OBJDIR_FAST_MATH) && make -j$(JOBS) -f Makefile FAST_MATH=yes
#	cd $(OBJDIR_FAST_MATH) && make -j$(JOBS) -f Makefile FAST_MATH=yes FTRACE=yes # requirement veos >= 2.4.1


$(OBJDIR_NO_FAST_MATH): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@

$(OBJDIR_FAST_MATH): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@

clean:
	sh scripts/clean.sh $(BASEDIR)
