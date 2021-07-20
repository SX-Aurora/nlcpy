.PHONY:nlcpy

all:nlcpy

BASEDIR = ..

# Parameters passed into the spec file
PLAT           = el8
ARCH           = x86_64
VERSION_PYTHON = python3.6
VERSION_NLCPY  = 2.0.0
VERSION_RPM    = 1
RELEASE        = 1
VERSION_NUMPY  = 1.19.5

DISTDIR  = $(BASEDIR)/dist
DOCDIR   = $(BASEDIR)/doc/build/html
RPMDIR   = $(DISTDIR)/rpm/RPMBUILD
SPECS    = $(RPMDIR)/SPECS
RPMS     = $(RPMDIR)/RPMS
SOURCES  = $(RPMDIR)/SOURCES

# convert 'python3.x -> python3x'
$(eval REQUIRES_PYTHON = $(shell echo $(VERSION_PYTHON) | sed -e s"/\.//"))

PRODUCT        = nec-$(REQUIRES_PYTHON)-nlcpy-ve-$(VERSION_NLCPY)
REQUIRES_NUMPY = nec-$(REQUIRES_PYTHON)-numpy-$(VERSION_NUMPY)-$(VERSION_NLCPY)
SRC_NLCPY      = nec-$(REQUIRES_PYTHON)-nlcpy-ve-$(VERSION_NLCPY)-$(VERSION_RPM)-$(RELEASE)
RPM_NLCPY      = $(SRC_NLCPY).$(PLAT).$(ARCH).rpm

# Options of rpmbuild command
$(eval SPECDEF  =--define "_topdir         `cd ..;pwd`")
$(eval SPECDEF +=--define "python_version  $(VERSION_PYTHON)")
$(eval SPECDEF +=--define "nlcpy_version   $(VERSION_NLCPY)")
$(eval SPECDEF +=--define "rpm_version     $(VERSION_RPM)")
$(eval SPECDEF +=--define "nlcpy_platform  $(PLAT)")
$(eval SPECDEF +=--define "nlcpy_arch      $(ARCH)")
$(eval SPECDEF +=--define "rpm_release     $(RELEASE)")
$(eval SPECDEF +=--define "product         $(PRODUCT)")
$(eval SPECDEF +=--define "requires_numpy  $(REQUIRES_NUMPY)")

nlcpy: $(RPMS)/$(ARCH)/$(RPM_NLCPY)

# Creates Specfiles
$(SPECS)/$(RPM_NLCPY:%.rpm=%.spec):
	mkdir -p $(SPECS) $(RPMS) $(SOURCES)
	cp -p SPECS/nec-python3-nlcpy-ve $@

# Creates RPM files
$(RPMS)/$(ARCH)/$(RPM_NLCPY): $(SPECS)/$(RPM_NLCPY:%.rpm=%.spec)
	$(eval SRC    = $(SRC_NLCPY))
	$(eval SRCDIR = $(SOURCES)/$(SRC))
	rm -rf $(SRCDIR)
	mkdir -p $(SRCDIR)
	mkdir -p $(SRCDIR)/doc
	cp -p $(DISTDIR)/nlcpy*.whl $(SRCDIR)/
	cd $(SRCDIR) && unzip nlcpy*.whl && rm -f nlcpy*.whl
	cp -r $(DOCDIR)/* $(SRCDIR)/doc/
	cd $(SOURCES) && tar zcvf $(SRC).tar.gz $(SRC)
	cd $(SPECS) && rpmbuild --bb $(SPECDEF) $(^F)

clean:
	rm -rf $(RPMDIR)
