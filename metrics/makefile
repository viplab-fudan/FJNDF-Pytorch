#!/bin/bash

#*** PARAMETER *****************************************************************
#--- GLOBAL ----------------------------
DIR_BINARY  	 = ./bin
DIR_VMAF         = ./source/vmaf
DIR_SSIMULACRA2  = ./source/ssimulacra2

NAME_VMAF = vmaf
NAME_SSIMULACRA2 = ssimulacra2

#--- LOCAL -----------------------------


#--- DERIVED ---------------------------


#*** USAGE *********************************************************************
default:
	@	echo "TARGETS                                                                        "
	@	echo "    download:    download all metric code under source dir                     "
	@	echo "    	download_vmaf   :   download vmaf                                        "
	@	echo "		download_ssimulacra2  :  download ssimulacra2                            "
	@	echo "    update:    build all metric bianry file under bin dir                      "
	@	echo "    	update_vmaf   :   build vmaf                                             "
	@	echo "		update_ssimulacra2   :   build ssimulacra2                               "
	@	echo "    check:    check version of all metric under bin dir                        "
	@	echo "    	check_vmaf   :   check_version of vmaf                                   "
	@	echo "    	check_ssimulacra25   :   check_version of ssimulacra2                    "
	@	echo "                                                                               "
	@	echo "                                                                               "


#*** MAIN BODY *****************************************************************
#--- NORMAL TARGETS --------------------



#--- TOOL TARGETS ----------------------
download_vmaf:
	@	git clone https://github.com/Netflix/vmaf.git $(DIR_VMAF)

download_ssimulacra2:
	@	git clone https://github.com/cloudinary/ssimulacra2.git $(DIR_SSIMULACRA2)

download:
	@	make download_vmaf
	@	make download_ssimulacra2

check_vmaf:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_VMAF) -v

check_ssimulacra2:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_SSIMULACRA2) -v

check:
	@	make check_vmaf
	@	make check_ssimulacra2

update_vmaf:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_VMAF)/libvmaf ;\
		meson setup --buildtype release --prefix /usr/local build ;\
		ninja -C build ;\
		ninja -C build install
	@	cp $(DIR_VMAF)/libvmaf/build/tools/$(NAME_VMAF) $(DIR_BINARY)/$(NAME_VMAF) ;\
		chmod +x $(DIR_BINARY)/$(NAME_VMAF)
	@	cp -rf $(DIR_VMAF)/model /usr/local/share
	@	make check_vmaf

update_ssimulacra2:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_SSIMULACRA2) ;\
		mkdir build ;\
		cd build ;\
		cmake ../src -G Ninja ;\
		ninja ssimulacra2
	@	cp $(DIR_SSIMULACRA2)/build/$(NAME_SSIMULACRA2) $(DIR_BINARY)/$(NAME_SSIMULACRA2) ;\
		chmod +x $(DIR_BINARY)/$(NAME_SSIMULACRA2)
	@	make check_ssimulacra2

update:
	@   mkdir -p $(DIR_BINARY)
	@   make update_vmaf
	@	make update_ssimulacra2