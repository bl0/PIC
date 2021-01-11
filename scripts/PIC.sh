#!/bin/bash

# script for PIC

set -e
set -x

epochs=${1:-400}
warmup=$(( epochs / 40 ))

data_dir="./data/ImageNet-Zip"
output_dir="./output/PIC/epoch-${epochs}"

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --crop 0.08 \
    --aug MultiCrop \
    --zip --cache-mode part \
    --arch resnet50 \
    --model PIC \
    --contrast-temperature 0.2 \
    --mlp-head \
    --warmup-epoch ${warmup} \
    --epochs ${epochs} \
    --output-dir "${output_dir}" \
    \
    --window-size 131072 \
    --window-stride 16384 \
    --use-sliding-window-sampler \
    --shuffle-per-epoch \
    \
    --crop2 0.14 \
    --image-size 160 \
    --image-size2 96 \
    --num-crop 1 \
    --num-crop2 3 \

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_linear.py \
    --data-dir "${data_dir}" \
    --zip --cache-mode part \
    --arch resnet50 \
    --output-dir "${output_dir}/eval" \
    --pretrained-model "${output_dir}/current.pth" \
