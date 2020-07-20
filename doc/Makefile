FILEDIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BASEDIR:=$(shell dirname $(FILEDIR))

SRC=$(shell find $(BASEDIR)/nlcpy -type f -name '*.dox')
TARGETS:=$(SRC:.dox= )

define DOX
$(1): $(1).dox
	@perl $(FILEDIR)scripts/doxycommand_to_docstring.pl $(1).dox > $(1)
	@echo "$(notdir $(1)) was generated from $(notdir $(1)).dox"
endef

.PHONY:all
all: $(TARGETS)
$(foreach src,$(TARGETS),$(eval $(call DOX,$(src))))

