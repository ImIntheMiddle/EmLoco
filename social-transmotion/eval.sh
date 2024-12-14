#!/bin/bash
# compare the performance of various valueloss weights
set -x # print the command that is going to be executed
export CUDA_VISIBLE_DEVICES=$1 # set GPU device
# VALUELOSS_LIST="10 50 100 150 250 500 1000" # list of value loss weights
EXP_LIST="jta_valuenet_100 jta_st_standard" # list of value loss weights
# EXP_LIST="noisy_traj_standard noisy_traj_value100" # list of value loss weights
# EXP_LIST="jta_valuenet_100" # list of value loss weights
NOISE=$2 # noise level
MODALITY="traj+all" # modality of the dataset
# LIMIT_OBS=$3 # limit the number of observations
LIMIT_OBS_LIST="1 3 5 0" # list of limit observations

# for LIMIT_OBS in ${LIMIT_OBS_LIST}; do # loop over the limit observations
#     echo "Limit Observations: ${LIMIT_OBS}"
#     for VALUELOSS in ${VALUELOSS_LIST}; do # loop over the value loss weights
#         echo "Value Loss: ${VALUELOSS}"
#         python evaluate_jta.py --exp_name "jta_valuenet_${VALUELOSS}" --modality ${MODALITY} --limit_obs ${LIMIT_OBS} --valueloss # run the evaluation script
#     done # end of for loop
# done # end of for loop

# for LIMIT_OBS in ${LIMIT_OBS_LIST}; do # loop over the limit observations
#     echo "Limit Observations: ${LIMIT_OBS}"
#     for EXP in ${EXP_LIST}; do # loop over the value loss weights
#         echo "Exp Name: ${EXP}"
#         python evaluate_jta.py --exp_name ${EXP} --modality ${MODALITY} --limit_obs ${LIMIT_OBS} # run the evaluation script
#     done # end of for loop
# done # end of for loop

for LIMIT_OBS in ${LIMIT_OBS_LIST}; do # loop over the limit observations
    echo "Limit Observations: ${LIMIT_OBS}"

    echo "Standard Noise${NOISE}"
    python evaluate_jta.py --exp_name jta_standard_noisy${NOISE} --modality ${MODALITY} --limit_obs ${LIMIT_OBS} # run the evaluation script

    echo "Ours Noise${NOISE}"
    python evaluate_jta.py --exp_name jta_value100_noisy${NOISE} --modality ${MODALITY} --limit_obs ${LIMIT_OBS} # run the evaluation script
done # end of for loop

# for VALUELOSS in ${VALUELOSS_LIST}; do # loop over the value loss weights
#     echo "Value Loss: ${VALUELOSS}"
#     python evaluate_jta.py --exp_name "jta_valuenet_${VALUELOSS}" --modality ${MODALITY} --limit_obs ${LIMIT_OBS} # run the evaluation script
# done # end of for loop