# ffmpeg is used to process video and calculate metrics
# Install ffmpeg with sources code and compile with libx264, libx265, libaom and libvamf
# ffmpeg
make download_ffmpeg
make update_ffmpeg


# vqmt is used to calculate ms-ssim and psnr-hvsm
# vqmt
make download_vqmt
make update_vqmt