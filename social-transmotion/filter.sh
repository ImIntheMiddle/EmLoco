#!/bin/bash
# compare the performance of various valueloss weights
set -x # print the command that is going to be executed
export CUDA_VISIBLE_DEVICES=$1 # set GPU device
obs_len=(0 2)
thresh=(0.65 0.7)
exp=('jta_discount_nonnorm_standard_5heads_1110' 'jta_discount_nonnorm_value100_5heads_1110')
epoch=(22 26)

for i in ${!obs_len[@]}; do
    for k in ${!thresh[@]}; do
        for j in ${!exp[@]}; do
            # echo "python evaluate_jta.py --exp_name ${exp[j]} --epoch ${epoch[j]} --limit_obs ${obs_len[i]} --valueloss --multi_modal --filter_threshold ${thresh[k]}"
            python evaluate_jta.py --exp_name ${exp[j]} --epoch ${epoch[j]} --limit_obs ${obs_len[i]} --valueloss --multi_modal --filter_threshold ${thresh[k]}
        done
        python evaluate_jta.py --exp_name 'jta_multimodal_20heads_standard' --limit_obs ${obs_len[i]} --valueloss --multi_modal --filter_threshold ${thresh[k]}
        echo ${obs_len[i]}
        if [ ${obs_len[i]} -eq '2' ]; then # if the observation length is 2
            python evaluate_jta.py --exp_name 'jta_discount_nonnorm_value100_20heads_1112' --limit_obs 2 --valueloss --multi_modal --filter_threshold ${thresh[k]}
        fi
        if [ ${obs_len[i]} -eq '0' ]; then # if the observation length is 0
            python evaluate_jta.py --exp_name 'jta_discount_1101_20heads_value100' --limit_obs 0 --valueloss --multi_modal --filter_threshold ${thresh[k]}
        fi
    done
done

echo "Done"