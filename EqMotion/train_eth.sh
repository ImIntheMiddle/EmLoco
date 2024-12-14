#!/bin/bash

# 使用例: ./train_all_subsets.sh GPU_ID VALUELOSS_W EXP_NAME_PREFIX
GPU_ID=$1
EXP_NAME_PREFIX=$2

# subsetのリスト
SUBSETS=("eth" "hotel" "univ" "zara1" "zara2")
# SUBSETS=("eth")

# ValueLossの重み候補
VALUELOSS_W=(0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002)
# VALUELOSS_W=(0.005 0.04 0.05 0.06 0.07 0.08 0.09)

# EXP_NAMEがstandardの場合、各サブセットについてテストするだけ
if [ $EXP_NAME_PREFIX = "standard" ]; then
  for SUBSET in "${SUBSETS[@]}"
  do
    echo "Training on subset: $SUBSET"
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_eth_diverse.py --subset $SUBSET --exp_name "${EXP_NAME_PREFIX}"

    echo "Finished training on subset: $SUBSET"
  done
  exit 0
fi

# 各ValueLossの重み候補について、全てのsubsetで学習を行う
if [ $EXP_NAME_PREFIX = "valuenet" ]; then
  for VALUELOSS_W in "${VALUELOSS_W[@]}"
  do
    echo "Training with ValueLoss weight: $VALUELOSS_W"
    for SUBSET in "${SUBSETS[@]}"
    do
      echo "Training on subset: $SUBSET"
      # exp_nameにSUBSETを追加して実行
      CUDA_VISIBLE_DEVICES=$GPU_ID python main_eth_diverse.py --subset $SUBSET --valueloss_w $VALUELOSS_W --exp_name "${EXP_NAME_PREFIX}_${VALUELOSS_W}/${SUBSET}"

      echo "Finished training on subset: $SUBSET"
    done
  done
  exit 0
fi

# for SUBSET in "${SUBSETS[@]}"
# do
#   echo "Training on subset: $SUBSET"
#   # exp_nameにSUBSETを追加して実行
#   CUDA_VISIBLE_DEVICES=$GPU_ID python main_eth_diverse.py --subset $SUBSET --valueloss_w $VALUELOSS_W --exp_name "${EXP_NAME_PREFIX}_${}/${SUBSET}"

#   echo "Finished training on subset: $SUBSET"
# done
