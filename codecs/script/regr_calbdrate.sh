#!/usr/bin/env bash
set -euo pipefail

# -------- Configuration --------
# Dataset group list (outermost loop)
DS_GROUP_LIST=(
  hevc_sdr_ctc
  xiph_dataset_1080p
  mcl_jcv_dataset_1080p
  mcl_jci_dataset_1080p
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
  # tcsvt_wei_2009_filter_1
  # tcsvt_wei_2009_filter_2
  # tcsvt_wei_2009_filter_3
  # tcsvt_wei_2009_filter_4
  # spl_bae_2013_filter_1
  # spl_bae_2013_filter_2
  # spl_bae_2013_filter_3
  # spl_bae_2013_filter_4
  # tip_bae_2016_filter_1
  # tip_bae_2016_filter_2
  # tip_bae_2016_filter_3
  # tip_bae_2016_filter_4
  # tob_kang_2023_filter_1
  # tob_kang_2023_filter_2
  # tob_kang_2023_filter_3
  # tob_kang_2023_filter_4
  # ojcas_sun_2024_filter
  # iccv_yan_2025_filter
  # iccv_yan_lite_2025_filter
)

# ----------------------
for DS_GROUP in "${DS_GROUP_LIST[@]}"; do
  echo "===== Processing dataset group: ${DS_GROUP} ====="
  for DATASET in "${DATASET_LIST[@]}"; do
    echo ">>> Processing dataset: ${DATASET}"
    for ENCODER in "${ENCODE_LIST[@]}"; do
      (
        ANCHOR_DB="../result/${DS_GROUP}/ori/${ENCODER}/result.db"
        TEST_DB="../result/${DS_GROUP}/${DATASET}/${ENCODER}/result.db"
        python3 calbdrate.py \
          -a   ${ANCHOR_DB} \
          -t   ${TEST_DB} \
          -o   ${DS_GROUP}_${DATASET}_${ENCODER}.csv
      ) &
    done
    wait
  done
done

echo "All tasks completed!"