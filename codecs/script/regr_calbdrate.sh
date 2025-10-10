#!/usr/bin/env bash
set -euo pipefail

# ———————— 配置区 ————————
# 编码器列表(并行处理)
ENCODE_LIST=(x264 x265)
# 数据集列表(串行处理)
DATASET_LIST=(
              ori
              tip_bae_2016
              puc_he_2025
              arxiv_ma_2023
            )
# 数据集分组名称(用于生成路径)
DS_GROUP="hevc_sdr_ctc"
# ——————————————————————

for DATASET in "${DATASET_LIST[@]}"; do
  echo ">>> 处理数据集:${DATASET}"
  for ENCODER in "${ENCODE_LIST[@]}"; do
    (
      ANCHOR_DB="../result/${DS_GROUP}/ori/${ENCODER}/result.db"
      TEST_DB="../result/${DS_GROUP}/${DATASET}/${ENCODER}/result.db"
      echo $ANCHOR_DB
      python3 calbdrate.py \
        -a   ${ANCHOR_DB} \
        -t   ${TEST_DB} \
        -o   ${DATASET}_${ENCODER}.csv \
    ) &
  done

  wait
done

echo "所有任务完成！"
