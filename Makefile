BASEDIR = $(CURDIR)

include make.inc

.PHONY: all nlcpy_ve_common nlcpy_ve_no_fast_math nlcpy_ve_fast_math FORCE

JOBS:=$(shell grep -c ^processor /proc/cpuinfo 2>/dev/null)
JOBS:=$(shell if [ $(JOBS) -le 16 ]; then echo "8"; elif [ $(JOBS) -le 32 ]; then echo "16"; else echo "32"; fi)

all: make.dep nlcpy_ve_common nlcpy_ve_no_fast_math nlcpy_ve_fast_math
	cp $(SRCDIR)/*.h $(BASEDIR)/nlcpy/include/

make.dep: $(OBJDIR_COMMON) $(OBJDIR_NO_FAST_MATH) $(OBJDIR_FAST_MATH)
	cd $(OBJDIR_COMMON) && sh $(TOOLDIR)/make_dep.sh
	cd $(OBJDIR_NO_FAST_MATH) && sh $(TOOLDIR)/make_dep.sh
	cd $(OBJDIR_FAST_MATH) && sh $(TOOLDIR)/make_dep.sh

nlcpy_ve_common:
	cd $(OBJDIR_COMMON) && make -f Makefile perl
	cd $(OBJDIR_COMMON) && make -j$(JOBS) -f Makefile COMMON=yes
#	cd $(OBJDIR_COMMON) && make -j$(JOBS) -f Makefile COMMON=yes FTRACE=yes

nlcpy_ve_no_fast_math: nlcpy_ve_common
	cd $(OBJDIR_NO_FAST_MATH) && make -f Makefile perl
	cd $(OBJDIR_NO_FAST_MATH) && make -j$(JOBS) -f Makefile
#	cd $(OBJDIR_NO_FAST_MATH) && make -j$(JOBS) -f Makefile FTRACE=yes

nlcpy_ve_fast_math: nlcpy_ve_common
	cd $(OBJDIR_FAST_MATH) && make -f Makefile FAST_MATH=yes perl
	cd $(OBJDIR_FAST_MATH) && make -j$(JOBS) -f Makefile FAST_MATH=yes
#	cd $(OBJDIR_FAST_MATH) && make -j$(JOBS) -f Makefile FAST_MATH=yes FTRACE=yes


$(OBJDIR_COMMON): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@

$(OBJDIR_NO_FAST_MATH): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@

$(OBJDIR_FAST_MATH): FORCE
	mkdir -p $@
	cd $(SRCDIR) && $(CPIO) $@

clean:
	sh scripts/clean.sh $(BASEDIR)
