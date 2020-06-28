#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

export WANDB_RUN_ID=${JOB_ID}
export WANDB_MODE=dryrun

DATASET=DS
ID=${DATASET}_${JOB_ID}

python -u main.py \
    --id ${ID} \
    --dataset data/${DATASET} \
    --static_dim SD \
    --absolute_dim AD \
    --relative_dim RD \
    --dropout 0.4 \
    --gamma 6.0 \
    --alpha 0.5 \
    --lmbda 0.0005 \
    --learning_rate 0.0003 \
    --learning_rate_steps 10000,100000 \
    --weight_decay 0.0 \
    --criterion NS \
    --negative_sample_size 256 \
    --negative_time_sample_size 32 \
    --negative_max_time_gap 0 \
    --batch_size 64 \
    --test_batch_size 1 \
    --max_steps 200000 \
    --save_path models/${ID} \
    --metric MRR \
    --mode head \
    --valid_steps 10000 \
    --valid_approximation 0 \
    --log_steps 100 \
    --test_log_steps 100 \
    --tensorboard_dir logs/tensorboard/${ID} \
    --wandb_dir logs \
    --timezone "America/Montreal" \
    --do_train --do_valid --do_test \
    --negative_adversarial_sampling \
    --heuristic_evaluation --type_evaluation
