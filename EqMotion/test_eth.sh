#!/bin/bash

# 使用例: ./train_all_subsets.sh GPU_ID VALUELOSS_W EXP_NAME_PREFIX
GPU_ID=$1
EXP_NAME_PREFIX=$2

# subsetのリスト
SUBSETS=("eth" "hotel" "univ" "zara1" "zara2")
# SUBSETS=("eth")
# ValueLossの重み候補
# VALUELOSS_W=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
# VALUELOSS_W=(0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
# VALUELOSS_W=(0.00001 0.00005 0.0001 0.0005 0.001)
VALUELOSS_W=(0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002)
# VALUELOSS_W=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# EXP_NAMEがstandardの場合、各サブセットについてテストするだけ
if [ $EXP_NAME_PREFIX = "standard" ]; then
  for SUBSET in "${SUBSETS[@]}"
  do
    echo "Testing on subset: $SUBSET"
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_eth_diverse.py --test --subset $SUBSET --exp_name "${EXP_NAME_PREFIX}"

    echo "Finished testing on subset: $SUBSET"
  done
  exit 0
fi

# valuenetの場合，各ValueLossの重み候補について、全てのsubsetでテスト
if [ $EXP_NAME_PREFIX = "valuenet" ]; then
  for VALUELOSS_W in "${VALUELOSS_W[@]}"
  do
    echo "Testing with ValueLoss weight: $VALUELOSS_W"
    for SUBSET in "${SUBSETS[@]}"
    do
      echo "Testing on subset: $SUBSET"
      CUDA_VISIBLE_DEVICES=$GPU_ID python main_eth_diverse.py --test --subset $SUBSET --valueloss_w $VALUELOSS_W --exp_name "${EXP_NAME_PREFIX}_${VALUELOSS_W}"

      echo "Finished testing on subset: $SUBSET"
    done
  done
  exit 0
fi