python3 filt_one_video.py \
  --input ./datasets/kodak24/ori/kodim01.yuv --resolution 768x512 \
  --format yuv420p --bitdepth 8 \
  --cfg options/inference/infer_tob_kang_2023.yml --start_frame 0 --frame_num 1 \
  --platform cpu --gpu_ids 0 --cores 1 --output ./
