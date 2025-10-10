#!/usr/bin/env bash

FRAME_NUM=1
CORE=4
PLAT=cpu

DATASET_LIST=(hevc_sdr_ctc)
MODEL_LIST=(
            tcsvt_wei_2009
            spl_bae_2013
            tip_bae_2016
            tob_kang_2023
            ojcas_sun_2024
            iccv_yan_2025
            iccv_yan_lite_2025
)

for DATASET in "${DATASET_LIST[@]}"; do
  (
    echo ">>> Processing dataset: ${DATASET}"
    for MODEL in "${MODEL_LIST[@]}"; do
      echo "  >>> Using model: ${MODEL}"
      python3 filt_one_dataset.py \
        --input_csv datasets/info/codec/${DATASET}_meta_info.csv \
        --input_dir datasets/${DATASET}/ori \
        --cfg options/inference/infer_${MODEL}.yml \
        --frame_num "${FRAME_NUM}" \
        --platform "${PLAT}" --gpu_ids "${CORE}" --cores "${CORE}"\
        --output datasets/${DATASET}/${MODEL}/
    done
  ) &
done

# Wait for all background tasks to complete
wait

echo "All datasets processed."