#!/usr/bin/env bash
set -euo pipefail

# ———————— Configuration ————————
# Dataset group list (outermost loop)
DS_GROUP_LIST=(
  mcl_jcv_dataset_1080p
)

# Encoder list (parallel)
ENCODE_LIST=(
  x264
  x265
  aomenc
  vvenc
  )

# Dataset list (serial)
DATASET_LIST=(
  ori
  tcsvt_wei_2009_filter_1
  tcsvt_wei_2009_filter_2
  tcsvt_wei_2009_filter_3
  tcsvt_wei_2009_filter_4
  spl_bae_2013_filter_1
  spl_bae_2013_filter_2
  spl_bae_2013_filter_3
  spl_bae_2013_filter_4
  tip_bae_2016_filter_1
  tip_bae_2016_filter_2
  tip_bae_2016_filter_3
  tip_bae_2016_filter_4
  tob_kang_2023_filter_1
  tob_kang_2023_filter_2
  tob_kang_2023_filter_3
  tob_kang_2023_filter_4
  ojcas_sun_2024
  iccv_yan_2025
  iccv_yan_lite_2025
)

# Common parameters
CORE=4
METRICS="psnr psnr_hvsm ssim ms_ssim vmaf vmaf_neg"
FPS=25
FRAME_NUM=1
ENC_MODE="CQP"
QP_LIST="27,32,37,42"

# ——————————————————————

for DS_GROUP in "${DS_GROUP_LIST[@]}"; do
  echo "===== Processing dataset group: ${DS_GROUP} ====="

  # Each group has its own meta_info.csv and root input directory
  INPUT_CSV="../../datasets/info/codec/${DS_GROUP}_meta_info.csv"
  GROUP_INPUT_ROOT="../../datasets/${DS_GROUP}"

  for DATASET in "${DATASET_LIST[@]}"; do
    echo ">>> Processing dataset: ${DATASET}"
    INPUT_DIR="${GROUP_INPUT_ROOT}/${DATASET}"

    for ENCODER in "${ENCODE_LIST[@]}"; do
      (
        echo "    ▶ Launch encoder: ${ENCODER}"
        CFG_PATH="../cfg/${ENCODER}.yml"
        OUTPUT_DIR="../result/${DS_GROUP}/${DATASET}/${ENCODER}"
        DB_PATH="${OUTPUT_DIR}/result.db"

        # Ensure output directory exists
        mkdir -p "${OUTPUT_DIR}"

        # -- Backup current script to output directory --
        SCRIPT_NAME=$(basename "$0")
        cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"
        echo "      • Script backed up: $0 → ${OUTPUT_DIR}/${SCRIPT_NAME}"

        # -- Backup configuration file to output directory --
        cp "${CFG_PATH}" "${OUTPUT_DIR}/"
        CFG_BASENAME=$(basename "${CFG_PATH}")
        NEW_CFG_PATH="${OUTPUT_DIR}/${CFG_BASENAME}"
        echo "      • Config backed up: ${CFG_PATH} → ${NEW_CFG_PATH}"

        # Run the encoding script using the backed-up configuration
        python3 encode.py \
          --encoder   "${NEW_CFG_PATH}" \
          --input_csv "${INPUT_CSV}" \
          --input_dir "${INPUT_DIR}" \
          --core      "${CORE}" \
          --output    "${OUTPUT_DIR}" \
          --metrics   ${METRICS} \
          --db        "${DB_PATH}" \
          --fps       "${FPS}" \
          --frame_num "${FRAME_NUM}" \
          --enc_mode  "${ENC_MODE}" \
          --qp_list   "${QP_LIST}"

        echo "    ✔ Encoding finished: ${ENCODER}"
      ) &
    done

    # Wait for all encoders under this dataset to finish
    wait
    echo "<<< Dataset ${DATASET} all encoders finished"
  done

  echo "===== Group ${DS_GROUP} all datasets processed ====="
done