#!/usr/bin/env bash
set -euo pipefail

# ———————— Configuration Area ————————
# Encoder list (parallel processing)
ENCODE_LIST=(x264 x265)
# Dataset list (serial processing)
DATASET_LIST=(
              ori
              tip_bae_2016
              puc_he_2025
              arxiv_ma_2023
            )
# Dataset group name (for path generation)
DS_GROUP="hevc_sdr_ctc"
# ——————————————————————

# Common parameters
CORE=6
METRICS="psnr psnr_hvsm ssim ms_ssim vmaf vmaf_neg"
FPS=25
FRAME_NUM=1
ENC_MODE="CQP"
QP_LIST="17,22,27,32,37,42"
# ——————————————————————

for DATASET in "${DATASET_LIST[@]}"; do
  echo ">>> Processing dataset: ${DATASET}"
  INPUT_CSV="../../datasets/info/codec/${DS_GROUP}_meta_info.csv"
  INPUT_DIR="../../datasets/${DS_GROUP}/${DATASET}"

  for ENCODER in "${ENCODE_LIST[@]}"; do
    (
      echo "    ▶ Starting encoder: ${ENCODER}"
      CFG_PATH="../cfg/${ENCODER}.yml"
      OUTPUT_DIR="../result/${DS_GROUP}/${DATASET}/${ENCODER}"
      DB_PATH="${OUTPUT_DIR}/result.db"

      # Ensure output directory exists
      mkdir -p "${OUTPUT_DIR}"

      # —— Backup current script to output directory ——
      SCRIPT_NAME=$(basename "$0")
      cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"
      echo "      • Script backed up: $0 → ${OUTPUT_DIR}/${SCRIPT_NAME}"

      # —— Backup configuration file to output directory ——
      cp "${CFG_PATH}" "${OUTPUT_DIR}/"
      CFG_BASENAME=$(basename "${CFG_PATH}")
      NEW_CFG_PATH="${OUTPUT_DIR}/${CFG_BASENAME}"
      echo "      • Configuration backed up: ${CFG_PATH} → ${NEW_CFG_PATH}"

      # Run encoding script using backed up configuration
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

      echo "    ✔ Encoding completed: ${ENCODER}"
    ) &
  done

  # Wait for all encoders under this dataset to finish
  wait
  echo "<<< Dataset ${DATASET} all encoding completed"
done