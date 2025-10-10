#!/bin/bash

#*** PARAMETER *****************************************************************
#--- GLOBAL ----------------------------
DIR_BINARY = ./bin
DIR_FFMPEG  = ./source/ffmpeg
NAME_FFMPEG  = ffmpeg

DIR_VQMT = ./source/vqmt
NAME_VQMT = vqmt

#--- LOCAL -----------------------------


#--- DERIVED ---------------------------


#*** USAGE *********************************************************************
default:
	@	echo "TARGETS                                                                        "
	@	echo "    download:    download all en/decoders under source dir                     "
	@	echo "    	download_ffmpeg    :   download ffmpeg                                   "
	@	echo "    update:    build all en/decoders bianry file under bin dir                 "
	@	echo "    	update_ffmpeg    :   build aom en/decoder                                "
	@	echo "    check:    check version of all encoders under codec dir                    "
	@	echo "    	check_ffmpeg    :   check_version of aom  en/decoder                     "
	@	echo "                                                                               "
	@	echo "                                                                               "


#*** MAIN BODY *****************************************************************
#--- NORMAL TARGETS --------------------



#--- TOOL TARGETS ----------------------
download_ffmpeg:
	@	git clone https://git.ffmpeg.org/ffmpeg.git $(DIR_FFMPEG)

download_vqmt:
	@	git clone https://github.com/rolinh/VQMT.git $(DIR_VQMT)

download:
	@	make download_ffmpeg
	@	make download_vqmt

check_ffmpeg:
	$(NAME_FFMPEG) -v

check_vqmt:
	@	cd $(DIR_BINARY) ;\
		./$(NAME_VQMT) -v

check:
	@	make check_ffmpeg
	@	make check_vqmt

update_ffmpeg:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_FFMPEG) ;\
		./configure \
		--prefix="/usr/local" \
		--pkg-config-flags="--static" \
		--extra-cflags="-I/usr/local/include" \
		--extra-ldflags="-L/usr/local/lib" \
		--extra-libs="-lpthread -lm" \
		--enable-gpl \
		--enable-nonfree \
		--enable-static \
		--disable-shared \
		--disable-debug \
		--disable-doc \
		--enable-libx264 \
		--enable-libx265 \
		--enable-libaom \
		--enable-libvmaf \
		--enable-libfdk-aac \
		--enable-libmp3lame \
		--enable-libopus \
		--enable-libvorbis \
		--enable-libvpx
	@	cd $(DIR_FFMPEG) ;\
		make -j8 ;\
		make install

update_vqmt:
	@   mkdir -p $(DIR_BINARY)
	@	cd $(DIR_VQMT) ;\
		make -j8
	@	cp $(DIR_VQMT)/build/bin/Release/$(NAME_VQMT) $(DIR_BINARY)/$(NAME_VQMT) ;\
		chmod +x $(DIR_BINARY)/$(NAME_VQMT)
	@	make check_vqmt

update:
	@   make update_ffmpeg
	@	make update_vqmt