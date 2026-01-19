#!/usr/bin/env bash

FRAME_NUM=1
CORE=4
PLAT=cpu

DATASET_LIST=(
  kodak24
# hevc_sdr_ctc
# xiph_dataset_1080p
# uvg_dataset_1080p
# mcl_jcv_dataset_1080p
)

MODEL_LIST=(
  tcsvt_chou_1995
# tcsvt_wei_2009
# spl_bae_2013
# tip_bae_2016
# tob_kang_2023
# ojcas_sun_2024
# iccv_yan_2025
# iccv_yan_lite_2025
)

# This list is only for traditional JND models;
# for learning-based methods, keep only 0.
INJECT_METHOD_LIST=(0 1 2 3 4)

for DATASET in "${DATASET_LIST[@]}"; do
  (
    echo ">>> Processing dataset: ${DATASET}"
    for MODEL in "${MODEL_LIST[@]}"; do
      echo "  >>> Using model: ${MODEL}"
      for INJECT in "${INJECT_METHOD_LIST[@]}"; do
        echo "    >>> Inject method: ${INJECT}"
        python3 filt_one_dataset.py \
          --input_csv datasets/info/codec/${DATASET}_meta_info.csv \
          --input_dir datasets/${DATASET}/ori \
          --cfg options/inference/infer_${MODEL}.yml \
          --target "${INJECT}" \
          --frame_num "${FRAME_NUM}" \
          --platform "${PLAT}" --gpu_ids "${CORE}" --cores "${CORE}" \
          --inject_method "${INJECT}" \
          --output datasets/${DATASET}/${MODEL}_filter_${INJECT}/
      done
    done
  ) &
done

# Wait for all background tasks to complete
wait

echo "All datasets processed."