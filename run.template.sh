#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

export WANDB_RUN_ID=${JOB_ID}
export WANDB_MODE=dryrun

MODEL=RotatE
DATASET=DS
ID=${MODEL}_${DATASET}_${JOB_ID}

python -u codes/run.py \
    --id ${ID} \
    --dataset data/${DATASET} \
    --model ${MODEL} \
    --static_dim SD \
    --absolute_dim AD \
    --relative_dim RD \
    --dropout 0.2 \
    --gamma 6.0 \
    --alpha 0.5 \
    --lmbda 0.0 \
    --learning_rate 0.00003 \
    --learning_rate_steps 50000 \
    --weight_decay 0.0 \
    --criterion NS \
    --negative_sample_size 256 \
    --negative_time_sample_size 0 \
    --negative_max_time_gap 0 \
    --batch_size 64 \
    --test_batch_size 8 \
    --max_steps 100000 \
    --save_path models/${ID} \
    --metric MRR \
    --mode head \
    --valid_steps 5000 \
    --valid_approximation 50 \
    --log_steps 1000 \
    --test_log_steps 10 \
    --log_dir runs/${ID} \
    --timezone "America/Montreal" \
    --do_train --do_valid --do_test \
    --negative_adversarial_sampling \
    --heuristic_evaluation --type_evaluation
