#!/bin/bash

set -e
set -x

epochs=${1:-200}
warmup=$(( epochs / 40 ))

data_dir="./data/ImageNet-Zip"
output_dir="./output/SimCLR"

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --crop 0.08 \
    --aug SimCLR \
    --zip --cache-mode part \
    --model SimCLR \
    --optimizer "lars" \
    --contrast-temperature 0.1 \
    --mlp-head \
    --base-lr 0.3 \
    --warmup-epoch ${warmup} \
    --weight-decay 1e-6 \
    --epochs ${epochs} \
    --output-dir "${output_dir}" \
    --save-freq 10 \


python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_linear.py \
    --data-dir "${data_dir}" \
    --zip --cache-mode part \
    --learning-rate 0.1 \
    --weight-decay 1e-6 \
    --output-dir "${output_dir}/eval" \
    --pretrained-model "${output_dir}/current.pth" \
