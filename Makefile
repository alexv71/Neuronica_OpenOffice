#**************************************************************
# Copyright 2016 Aleksandr Voishchev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http ://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#************************************************************

# Builds the Neuronica Extension (realizes some Neural Networks algorithms).

PRJ=../..
SETTINGS=$(PRJ)/settings

include $(SETTINGS)/settings.mk
include $(SETTINGS)/std.mk
include $(SETTINGS)/dk.mk

# Define non-platform/compiler specific settings
SAMPLE_NAME=NeuronicaComponent
SAMPLE_INC_OUT=$(OUT_INC)/$(SAMPLE_NAME)
SAMPLE_GEN_OUT=$(OUT_MISC)/$(SAMPLE_NAME)
SAMPLE_SLO_OUT=$(OUT_SLO)/$(SAMPLE_NAME)
SAMPLE_OBJ_OUT=$(OUT_OBJ)/$(SAMPLE_NAME)

COMP_NAME=Neuronica
COMP_IMPL_NAME=$(COMP_NAME).uno.$(SHAREDLIB_EXT)

COMP_RDB_NAME = $(COMP_NAME).uno.rdb
COMP_RDB = $(SAMPLE_GEN_OUT)/$(COMP_RDB_NAME)
COMP_PACKAGE = $(OUT_BIN)/$(COMP_NAME).$(UNOOXT_EXT)
COMP_PACKAGE_URL = $(subst \\,\,"$(COMP_PACKAGE_DIR)$(PS)$(COMP_NAME).$(UNOOXT_EXT)")
COMP_UNOPKG_MANIFEST = $(SAMPLE_GEN_OUT)/$(COMP_NAME)/META-INF/manifest.xml
#COMP_MAPFILE = $(SAMPLE_GEN_OUT)/$(COMP_NAME).uno.map
COMP_COMPONENTS = $(SAMPLE_GEN_OUT)/$(COMP_NAME).components

COMP_REGISTERFLAG = $(SAMPLE_GEN_OUT)/devguide_$(COMP_NAME)_register_component.flag
COMP_TYPEFLAG = $(SAMPLE_GEN_OUT)/devguide_$(COMP_NAME)_types.flag

COMP_LIBRARY_FILES=\
	Neuronica/dialog.xlb\
	Neuronica/script.xlb\
	Neuronica/BackPropagation.xba\
	Neuronica/DannysLib.xba\
	Neuronica/dlgBackPropagation.xdl\
	Neuronica/dlgParamBProp.xdl\
	Neuronica/dlgParamRProp.xdl\
	Neuronica/dlgParamSCG.xdl

COMP_MISC_FILES=\
	Addons.xcu\
	description.xml

COMP_DESC_FILES=\
	desc/en.txt

IDLFILES = neuronica.idl

CXXFILES = backprop_cmpnt.cxx neuralnet.cxx

SLOFILES = $(patsubst %.cxx,$(SAMPLE_SLO_OUT)/%.$(OBJ_EXT),$(CXXFILES))

GENURDFILES = $(patsubst %.idl,$(SAMPLE_GEN_OUT)/%.urd,$(IDLFILES))

TYPELIST=-Tneuronica.XBackProp  \
	-Tneuronica.BackProp
 
# Targets
.PHONY: ALL
ALL : \
	$(SAMPLE_NAME)

include $(SETTINGS)/stdtarget.mk

$(SAMPLE_GEN_OUT)/%.urd : %.idl
	-$(MKDIR) $(subst /,$(PS),$(@D))
	$(IDLC) -I. -I$(IDL_DIR) -O$(SAMPLE_GEN_OUT) $<

$(SAMPLE_GEN_OUT)/%.rdb : $(GENURDFILES)
	-$(DEL) $(subst \\,\,$(subst /,$(PS),$@))
	-$(MKDIR) $(subst /,$(PS),$(@D))
	$(REGMERGE) $@ /UCR $(GENURDFILES)

$(COMP_TYPEFLAG) : $(COMP_RDB) $(SDKTYPEFLAG)
	-$(DEL) $(subst \\,\,$(subst /,$(PS),$@))
	-$(MKDIR) $(subst /,$(PS),$(@D))
	$(CPPUMAKER) -Gc -BUCR -O$(SAMPLE_INC_OUT) $(TYPESLIST) $(COMP_RDB) -X$(URE_TYPES) -X$(OFFICE_TYPES)
	echo flagged > $@

$(SAMPLE_SLO_OUT)/%.$(OBJ_EXT) : %.cxx $(COMP_TYPEFLAG)
	-$(MKDIR) $(subst /,$(PS),$(@D))
	$(CC) $(CC_FLAGS) $(CC_INCLUDES) -I$(SAMPLE_INC_OUT) $(CC_DEFINES) $(CC_OUTPUT_SWITCH)$(subst /,$(PS),$@) $<

ifeq "$(OS)" "WIN"
$(SHAREDLIB_OUT)/%.$(SHAREDLIB_EXT) : $(SLOFILES)
	-$(MKDIR) $(subst /,$(PS),$(@D))
	-$(MKDIR) $(subst /,$(PS),$(SAMPLE_GEN_OUT))
	$(LINK) $(COMP_LINK_FLAGS) /OUT:$@ \
	/MAP:$(SAMPLE_GEN_OUT)/$(subst $(SHAREDLIB_EXT),map,$(@F)) $(SLOFILES) \
	$(CPPUHELPERLIB) $(CPPULIB) $(SALLIB) msvcrt.lib kernel32.lib 
	$(LINK_MANIFEST)
else
#$(SHAREDLIB_OUT)/%.$(SHAREDLIB_EXT) : $(SLOFILES) $(COMP_MAPFILE)
$(SHAREDLIB_OUT)/%.$(SHAREDLIB_EXT) : $(SLOFILES)
	-$(MKDIR) $(subst /,$(PS),$(@D)) && $(DEL) $(subst \\,\,$(subst /,$(PS),$@))
	$(LINK) $(COMP_LINK_FLAGS) $(LINK_LIBS) -o $@ $(SLOFILES) \
	$(CPPUHELPERLIB) $(CPPULIB) $(SALLIB) $(STC++LIB) $(CPPUHELPERDYLIB) $(CPPUDYLIB) $(SALDYLIB)
ifeq "$(OS)" "MACOSX"
	$(INSTALL_NAME_URELIBS)  $@
endif
endif	

# rule for component package manifest
$(SAMPLE_GEN_OUT)/%/manifest.xml :
	-$(MKDIR) $(subst /,$(PS),$(@D))
	@echo $(OSEP)?xml version="$(QM)1.0$(QM)" encoding="$(QM)UTF-8$(QM)"?$(CSEP) > $@
	@echo $(OSEP)!DOCTYPE manifest:manifest PUBLIC "$(QM)-//OpenOffice.org//DTD Manifest 1.0//EN$(QM)" "$(QM)Manifest.dtd$(QM)"$(CSEP) >> $@
	@echo $(OSEP)manifest:manifest xmlns:manifest="$(QM)http://openoffice.org/2001/manifest$(QM)"$(CSEP) >> $@
	@echo $(SQM)  $(SQM)$(OSEP)manifest:file-entry manifest:media-type="$(QM)application/vnd.sun.star.uno-typelibrary;type=RDB$(QM)" >> $@
	@echo $(SQM)                       $(SQM)manifest:full-path="$(QM)$(subst /META-INF,,$(subst $(SAMPLE_GEN_OUT)/,,$(@D))).uno.rdb$(QM)"/$(CSEP) >> $@
	@echo $(SQM)  $(SQM)$(OSEP)manifest:file-entry manifest:media-type="$(QM)application/vnd.sun.star.uno-components;platform=$(UNOPKG_PLATFORM)$(QM)">> $@
	@echo $(SQM)                       $(SQM)manifest:full-path="$(QM)$(COMP_NAME).components$(QM)"/$(CSEP)>> $@
	@echo $(SQM)  $(SQM)$(OSEP)manifest:file-entry manifest:media-type="$(QM)application/vnd.sun.star.basic-library$(QM)">> $@
	@echo $(SQM)                       $(SQM)manifest:full-path="$(QM)Neuronica/$(QM)"/$(CSEP)>> $@
	@echo $(SQM)  $(SQM)$(OSEP)manifest:file-entry manifest:media-type="$(QM)application/vnd.sun.star.configuration-data$(QM)">> $@
	@echo $(SQM)                       $(SQM)manifest:full-path="$(QM)Addons.xcu$(QM)"/$(CSEP)>> $@
	@echo $(OSEP)/manifest:manifest$(CSEP) >> $@

$(COMP_COMPONENTS) :
	-$(MKDIR) $(subst /,$(PS),$(@D))
	@echo $(OSEP)?xml version="$(QM)1.0$(QM)" encoding="$(QM)UTF-8$(QM)"?$(CSEP) > $@
	@echo $(OSEP)components xmlns="$(QM)http://openoffice.org/2010/uno-components$(QM)"$(CSEP) >> $@
	@echo $(SQM)  $(SQM)$(OSEP)component loader="$(QM)com.sun.star.loader.SharedLibrary$(QM)" uri="$(QM)$(UNOPKG_PLATFORM)/$(COMP_IMPL_NAME)$(QM)"$(CSEP) >> $@
#	@echo $(SQM)    $(SQM)$(OSEP)implementation name="$(QM)my_module.my_sc_implementation.MyService1$(QM)"$(CSEP) >> $@
#	@echo $(SQM)      $(SQM)$(OSEP)service name="$(QM)my_module.MyService1$(QM)"/$(CSEP) >> $@
#	@echo $(SQM)    $(SQM)$(OSEP)/implementation$(CSEP) >> $@
	@echo $(SQM)    $(SQM)$(OSEP)implementation name="$(QM)neuronica.my_sc_implementation.BackProp$(QM)"$(CSEP) >> $@
	@echo $(SQM)      $(SQM)$(OSEP)service name="$(QM)neuronica.BackProp$(QM)"/$(CSEP) >> $@
	@echo $(SQM)    $(SQM)$(OSEP)/implementation$(CSEP) >> $@
	@echo $(SQM)  $(SQM)$(OSEP)/component$(CSEP) >> $@
	@echo $(OSEP)/components$(CSEP) >> $@

$(COMP_PACKAGE) : $(SHAREDLIB_OUT)/$(COMP_IMPL_NAME) $(COMP_RDB) $(COMP_UNOPKG_MANIFEST) $(COMP_COMPONENTS) $(COMP_LIBRARY_FILES) $(COMP_MISC_FILES) $(COMP_DESC_FILES)
	-$(DEL) $(subst \\,\,$(subst /,$(PS),$@))
	-$(MKDIR) $(subst /,$(PS),$(@D))
	$(SDK_ZIP) $@ $(COMP_LIBRARY_FILES)
	$(SDK_ZIP) $@ $(COMP_MISC_FILES)
	$(SDK_ZIP) $@ $(COMP_DESC_FILES)
	-$(MKDIR) $(subst /,$(PS),$(SAMPLE_GEN_OUT)/$(UNOPKG_PLATFORM))	 
	$(COPY) $(subst /,$(PS),$<) $(subst /,$(PS),$(SAMPLE_GEN_OUT)/$(UNOPKG_PLATFORM))
	cd $(subst /,$(PS),$(SAMPLE_GEN_OUT)) && $(SDK_ZIP) ../../bin/$(@F) $(COMP_NAME).components
	cd $(subst /,$(PS),$(SAMPLE_GEN_OUT)) && $(SDK_ZIP) -u ../../bin/$(@F) $(COMP_RDB_NAME) $(UNOPKG_PLATFORM)/$(<F)
	cd $(subst /,$(PS),$(SAMPLE_GEN_OUT)/$(subst .$(UNOOXT_EXT),,$(@F))) && $(SDK_ZIP) -u ../../../bin/$(@F) META-INF/manifest.xml


$(COMP_REGISTERFLAG) : $(COMP_PACKAGE)
ifeq "$(SDK_AUTO_DEPLOYMENT)" "YES"
	-$(DEL) $(subst \\,\,$(subst /,$(PS),$@))
	-$(MKDIR) $(subst /,$(PS),$(@D))
	$(DEPLOYTOOL) $(COMP_PACKAGE_URL)
	@echo flagged > $(subst /,$(PS),$@)
else
	@echo --------------------------------------------------------------------------------
	@echo  If you want to install your component automatically, please set the environment
	@echo  variable SDK_AUTO_DEPLOYMENT = YES. But note that auto deployment is only 
	@echo  possible if no office instance is running. 
	@echo --------------------------------------------------------------------------------
endif

# touch the target to renew the date for correct dependencies.
# Note: no touch under windows! The unoapploader.exe is copied always.
ifneq "$(OS)" "WIN"
	touch $@
endif

$(SAMPLE_NAME) : $(COMP_REGISTERFLAG)
	@echo --------------------------------------------------------------------------------
	@echo Please use the following command to execute the example!
	@echo --------
	@echo The simple C++ component was installed if SDK_AUTO_DEPLOYMENT = YES.
	@echo You can use this component inside your office installation, see the example
	@echo description. You can also load the "$(QM)NeuronicaTest.ods$(QM)" document containing
	@echo a StarBasic macro which uses this component.
	@echo -
	@echo $(MAKE) NeuronicaTest.ods.load
	@echo --------------------------------------------------------------------------------

%.run: $(OUT_BIN)/%$(EXE_EXT)
	$(subst /,$(PS),$(OUT_BIN))$(PS)$(basename $@)
#	cd $(subst /,$(PS),$(OUT_BIN)) && $(basename $@)

NeuronicaTest.ods.load : $(COMP_REGISTERFLAG)
	"$(OFFICE_PROGRAM_PATH)$(PS)soffice" $(basename $@)

.PHONY: clean
clean :
	-$(DELRECURSIVE) $(subst /,$(PS),$(SAMPLE_INC_OUT))
	-$(DELRECURSIVE) $(subst /,$(PS),$(SAMPLE_GEN_OUT))
	-$(DELRECURSIVE) $(subst /,$(PS),$(SAMPLE_SLO_OUT))
	-$(DELRECURSIVE) $(subst /,$(PS),$(SAMPLE_OBJ_OUT))
	-$(DEL) $(subst \\,\,$(subst /,$(PS),$(COMP_COMPONENTS)))
	-$(DEL) $(subst \\,\,$(subst /,$(PS),$(OUT_BIN)/$(COMP_NAME)*))
