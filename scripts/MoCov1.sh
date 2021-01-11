#!/bin/bash

set -e
set -x

epochs=${1:-200}
warmup=$(( epochs / 40 ))

data_dir="./data/ImageNet-Zip"
output_dir="./output/MoCov1/epoch-${epochs}"

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --crop 0.2 \
    --aug InstDisc \
    --zip --cache-mode part \
    --model MoCo \
    --contrast-temperature 0.07 \
    --lr-scheduler step \
    --warmup-epoch ${warmup} \
    --epochs ${epochs} \
    --output-dir "${output_dir}" \
    --save-freq 10

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_linear.py \
    --data-dir "${data_dir}" \
    --zip --cache-mode part \
    --output-dir "${output_dir}/eval" \
    --pretrained-model "${output_dir}/current.pth" \
