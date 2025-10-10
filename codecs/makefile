#!/bin/bash

#*** PARAMETER *****************************************************************
#--- GLOBAL ----------------------------
DIR_BINARY  = ./bin
DIR_X264    = ./source/x264
DIR_X265    = ./source/x265
DIR_AOM     = ./source/aom
DIR_VVENC   = ./source/vvenc
DIR_VVDEC   = ./source/vvdec
DIR_VQMT    = ./source/vqmt

NAME_ENC_X264 = x264
NAME_ENC_X265 = x265
NAME_ENC_AOM  = aomenc
NAME_ENC_VVENC = vvenc
NAME_DEC_AOM  = aomdec
NAME_DEC_VVDEC = vvdec
NAME_VQMT = vqmt

#--- LOCAL -----------------------------


#--- DERIVED ---------------------------


#*** USAGE *********************************************************************
default:
	@	echo "TARGETS                                                                        "
	@	echo "    download:    download all en/decoders under source dir                     "
	@	echo "    	download_x264   :   download x264 encoder                                "
	@	echo "		download_x265   :   download x265 encoder                                "
	@	echo "    	download_aom    :   download aom encoder                                 "
	@	echo "    	download_vvenc  :   download vvenc encoder                               "
	@	echo "    	download_vvdec  :   download vvdec decoder                               "
	@	echo "    	download_vqmt   :   download vqmt  			                             "
	@	echo "    update:    build all en/decoders bianry file under bin dir                 "
	@	echo "    	update_x264   :   build x264 encoder                                     "
	@	echo "		update_x265   :   build x265 encoder                                     "
	@	echo "    	update_aom    :   build aom en/decoder                                   "
	@	echo "    	update_vvenc  :   build vvenc encoder                                    "
	@	echo "    	update_vvdec  :   build vvdec decoder                                    "
	@	echo "    	update_vqmt   :   build vqmt                                             "
	@	echo "    check:    check version of all encoders under codec dir                    "
	@	echo "    	check_x264   :   check_version of x264 encoder                           "
	@	echo "    	check_x265   :   check_version of x265 encoder                           "
	@	echo "    	check_aom    :   check_version of aom  en/decoder                        "
	@	echo "    	check_vvenc  :   check_version of vvenc encoder                          "
	@	echo "    	check_vvdec  :   check_version of vvdec decoder                          "
	@	echo "    	check_vqmt   :   check_version of vqmt                                   "
	@	echo "                                                                               "
	@	echo "                                                                               "


#*** MAIN BODY *****************************************************************
#--- NORMAL TARGETS --------------------



#--- TOOL TARGETS ----------------------
download_x264:
	@	git clone https://code.videolan.org/videolan/x264.git $(DIR_X264)

download_x265:
	@	git clone https://bitbucket.org/multicoreware/x265_git.git $(DIR_X265)

download_aom:
	@	git clone https://aomedia.googlesource.com/aom $(DIR_AOM)

download_vvenc:
	@	git clone https://github.com/fraunhoferhhi/vvenc.git $(DIR_VVENC)

download_vvdec:
	@	git clone https://github.com/fraunhoferhhi/vvdec.git $(DIR_VVDEC)

download_vqmt:
	@	git clone https://github.com/rolinh/VQMT.git $(DIR_VQMT)

download:
	@	make download_x264
	@	make download_x265
	@	make download_aom
	@	make download_vvenc
	@	make download_vvdec
	@	make download_vqmt

check_x264:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_ENC_X264) --version

check_x265:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_ENC_X265) --version

check_aom:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_ENC_AOM) -v ;\
		./$(NAME_DEC_AOM) -v ;\

check_vvenc:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_ENC_VVENC) --version

check_vvdec:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_DEC_VVDEC) --version

check_vqmt:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_VQMT) -v

check:
	@	make check_x264
	@	make check_x265
	@	make check_aom
	@	make check_vvenc
	@   make check_vvdec
	@	make check_vqmt

update_x264:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_X264) ;\
		./configure --enable-shared --enable-static
	@	cd $(DIR_X264) ;\
		make -j8 ;\
	@	cp $(DIR_X264)/$(NAME_ENC_X264) $(DIR_BINARY)/$(NAME_ENC_X264) ;\
		chmod +x $(DIR_BINARY)/$(NAME_ENC_X264)
	@	make check_x264

update_x265:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_X265)/build/linux ;\
		cmake -G "Unix Makefiles" ../../source
	@	cd $(DIR_X265)/build/linux ;\
		make -j8 ;\
	@	cp $(DIR_X265)/build/linux/$(NAME_ENC_X265) $(DIR_BINARY)/$(NAME_ENC_X265) ;\
		chmod +x $(DIR_BINARY)/$(NAME_ENC_X265)
	@	make check_x265

update_aom:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_AOM) ;\
	    make build
	@	cd $(DIR_AOM)/build ;\
		cmake -DCMAKE_BUILD_TYPE=Release .. ;\
		make -j8 ;\
	@	cp $(DIR_AOM)/build/$(NAME_ENC_AOM) $(DIR_BINARY)/$(NAME_ENC_AOM)
		cp $(DIR_AOM)/build/$(NAME_DEC_AOM) $(DIR_BINARY)/$(NAME_DEC_AOM)
		chmod +x $(DIR_BINARY)/$(NAME_ENC_AOM)
		chmod +x $(DIR_BINARY)/$(NAME_DEC_AOM)
	@	make check_aom

update_vvenc:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_VVENC) ;\
	    mkdir -p build
	@	cd $(DIR_VVENC)/build/ ;\
		cmake ..
	@	cd $(DIR_VVENC)/build/ ;\
		make -j8 ;\
	@	cp $(DIR_VVENC)/bin/release-static/vvencFFapp $(DIR_BINARY)/$(NAME_ENC_VVENC) ;\
		chmod +x $(DIR_BINARY)/$(NAME_ENC_VVENC)
	@	make check_vvenc

update_vvdec:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_VVDEC) ;\
	    mkdir -p build
	@	cd $(DIR_VVDEC)/build/ ;\
		cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=1 -DVVDEC_INSTALL_VVDECAPP=ON .. ;\
		make -j8 ;\
	@	cp $(DIR_VVDEC)/bin/release-shared/vvdecapp $(DIR_BINARY)/$(NAME_DEC_VVDEC) ;\
		chmod +x $(DIR_BINARY)/$(NAME_DEC_VVDEC)
	@	make check_vvdec

update_vqmt:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_VQMT) ;\
		make -j8
	@	cp $(DIR_VQMT)/build/bin/Release/$(NAME_VQMT) $(DIR_BINARY)/$(NAME_VQMT) ;\
		chmod +x $(DIR_BINARY)/$(NAME_VQMT)
	@	make check_vqmt

update:
	@   mkdir -p $(DIR_BINARY)
	@   make update_x264
	@	make update_x265
	@	make update_aom
	@	make update_vvenc
	@   make update_vvdec
	@	make update_vqmt