python3 filt_one_video.py \
  --input ./assets/kodim01.yuv --resolution 768x512 \
  --format yuv420p --bitdepth 8 \
  --cfg options/inference/infer_iccv_yan_lite_2025.yml --target 0 \
  --start_frame 0 --frame_num 1 \
  --platform cpu --gpu_ids 0 --cores 1 --output ./
