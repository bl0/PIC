# Parametric Instance Classification for Unsupervised Visual Feature Learning

By [Yue Cao](http://yue-cao.me)\*, [Zhenda Xie](https://scholar.google.com/citations?user=0C4cDloAAAAJ)\*, [Bin Liu](https://scholar.google.com/citations?user=-RYlJvYAAAAJ)*, [Yutong Lin](https://scholar.google.com/citations?user=mjUgH44AAAAJ), [Zheng Zhang](https://www.microsoft.com/en-us/research/people/zhez/), [Han Hu](https://ancientmooner.github.io/).

This repo is an official implementation of ["Parametric Instance Classification for Unsupervised Visual Feature Learning"](https://arxiv.org/abs/2006.14618v1) on PyTorch. It also contains unofficial implementation of several popular Unsupervised Visual Feature Learning methods, including [InstDisc](https://arxiv.org/abs/1805.01978.pdf), [MoCo](https://arxiv.org/abs/1911.05722), [MoCo v2](https://arxiv.org/abs/2003.04297), [SimCLR](https://arxiv.org/abs/2002.05709).



*Update on 2020/09/26*

Our paper was accepted by NeurIPS 2020!

## Introduction

This paper presents parametric instance classification (PIC) for unsupervised visual feature learning. Unlike the state-of-the-art approaches which do instance discrimination in a dual-branch non-parametric fashion, PIC directly performs a one-branch parametric instance classification, revealing a simple framework similar to supervised classification and without the need to address the information leakage issue. 

We show that the simple PIC framework can be as effective as the state-of-the-art approaches, i.e. SimCLR and MoCo v2, by adapting several common component settings used in the state-of-the-art approaches. 

We also propose two novel techniques to further improve effectiveness and practicality of PIC: 

1. A sliding-window data scheduler, instead of the previous epoch-based data scheduler, which addresses the extremely infrequent instance visiting issue in PIC and improves the effectiveness; 
2. A negative sampling and weight update correction approach to reduce the training time and GPU memory consumption, which also enables application of PIC to almost unlimited training images. 

We hope that the PIC framework can serve as a simple baseline to facilitate future study.

## Citation

```
@article{cao2020PIC,
  title={Parametric Instance Classification for Unsupervised Visual Feature Learning},
  author={Cao, Yue and Xie, Zhenda and Liu, Bin and Lin, Yutong and Zhang, Zheng and Hu, Han},
  booktitle={Advances in neural information processing systems},
  year={2020}
}
```

## Main Results

|           | #aug/iter X #epoch | Top-1 | Top-5 | Model |
| --------- | ------------------------- | ----- | ----- | ------- |
| InstDisc         | 1X200              | 60.6  | 82.6  | [download](https://drive.google.com/file/d/1ilTo2Lk0D8MIrLMY2FA9s2QtHnSfy35G/view?usp=sharing) |
| SimCLR                       | 2X100              | 64.7  | 86.0  |
| MoCo v2          | 2X100              | 64.6  | 85.9  | [download](https://drive.google.com/file/d/1dhOg2AZRhw42SOiXFmXedrRXhMY5gPOh/view?usp=sharing) |
| PIC (ours) | 1X200              | 68.6  | 88.8  | [download](https://drive.google.com/file/d/1eqtLv_RrBCgSEDhte6PueqAFlenaH50k/view?usp=sharing) |
| InstDisc         | 1X400              | 62.7  | 84.6  | [download](https://drive.google.com/file/d/1bWHvEZ9vyidtCVBZzkyTxGLmj7Vyla1j/view?usp=sharing) |
| SimCLR                       | 2X200              | 66.6  | 87.3  |
| MoCo v2          | 2X200              | 67.9  | 88.1  | [download](https://drive.google.com/file/d/1Y-PlmcFSLanIDjYr6Z2fPSYamWt4DO_7/view?usp=sharing) |
| PIC (ours) | 1X400              | 70.3  | 89.8  | [download](https://drive.google.com/file/d/1JdDfPr78BY_0MPeN2r_TX79KObyE-QO4/view?usp=sharing) |
| PIC (ours)                    | 1X1600             | 70.8  | 90.0  | 

**Notes**: 

* InstDisc and MoCo v2 refer to our re-implementation of InstDisc and MoCo v2. All model checkpoints can be found from [Google Drive](https://drive.google.com/drive/folders/12ihazCK8iogX3pvNA5tRJiLpcxxCiXOc?usp=sharing).

* To achieve better performance, our PIC adopts a multi-crop strategy, which is proposed in [SwAV](https://arxiv.org/abs/2006.09882). In each iteration, one 160 x 160 crop and three 96 x 96 crops of an image are fed into the model. With similar memory and compute requirements, PIC could achieve better performance than the original PIC model.

## Getting started

### Tested Environment

 - `Anaconda` with `python >= 3.6`
 - `pytorch=1.4, torchvison, cuda>=9.2`
 - [Optional] `Apex` for automatic mixed precision: Refer to https://github.com/NVIDIA/apex#quick-start
 - Others: ` pip install termcolor opencv-python tensorboard`

### Datasets

We use standard ImageNet dataset to pre-train the model, download it from http://www.image-net.org/ and unzip it.

* For standard folder dataset, move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

* To boost the performance when read images from massive small files is slow, we also support zipped ImageNet, which includes four files:
  * `train.zip`, `val.zip`: which stores the zipped folder for train and validate splits.

  * `train_map.txt`, `train_map.txt`: which stores the relative path in the corresponding zip file and ground truth label. 

    Make sure the data folder looks like this:

    ```
    $ tree data
    data
    └── ImageNet-Zip
        ├── train_map.txt
        ├── train.zip
        ├── val_map.txt
        └── val.zip

    $ head -n 5 data/ImageNet-Zip/val_map.txt
    ILSVRC2012_val_00000001.JPEG	65
    ILSVRC2012_val_00000002.JPEG	970
    ILSVRC2012_val_00000003.JPEG	230
    ILSVRC2012_val_00000004.JPEG	809
    ILSVRC2012_val_00000005.JPEG	516

    $ head -n 5 data/ImageNet-Zip/train_map.txt
    n01440764/n01440764_10026.JPEG	0
    n01440764/n01440764_10027.JPEG	0
    n01440764/n01440764_10029.JPEG	0
    n01440764/n01440764_10040.JPEG	0
    n01440764/n01440764_10042.JPEG	0
    ```

### Unsupervised training and linear evaluation

The implementation only supports **DistributedDataParallel** training with multiple GPU.

To do PIC pretraining and linear evaluation of a ResNet-50 model on ImageNet in a 4-GPU machine, run:

```
epochs=200
data_dir="./data/ImageNet-Zip"
output_dir="./output/PIC/epochs-${epochs}"

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --aug SimCLR \
    --crop 0.08 \
    --contrast-temperature 0.2 \
    --use-sliding-window-sampler \
    --model PIC \
    --mlp-head \
    --epochs ${epochs} \
    --output-dir ${output_dir}
    
python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    main_linear.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir}/eval \
    --pretrained-model ${output_dir}/current.pth
```

**Notes**:

* To use zipped ImageNet instead folder dataset, add `--zip` to the parameters.
  
  * To cache the dataset in the memory instead of reading from files every time, add `--cache-mode part`, which will sharding the dataset into nonoverlapping pieces for different GPU and only load from the corresponding one.
* To enable automatic mixed precision training, add `--amp-opt-level O1`.

* We have provided the scripts in `./scripts` to help reproduce the results of Our PIC and other methods. For example, to reproduce the results of PIC for 400 epoch, just run

  ```
  bash scripts/PIC.sh 400
  ```

* For additional options, run `python main_pretrain.py --help` and `python main_pretrain.py —help` to get help message. Or refer to [./contrast/option.py](./contrast/option.py).

## Known Issues

* For longer training (like training for 1600 epochs), the current implementation is unstable, and the loss may become *NaN* during training.
  We are now trying to figure out the cause of this phenomenon.
* Recent negative sampling and weight correction have not been implemented in this version. We will add these two techniques in the near future.

## References

Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* https://github.com/zhirongw/lemniscate.pytorch
* https://github.com/facebookresearch/moco
* https://github.com/HobbitLong/CMC
