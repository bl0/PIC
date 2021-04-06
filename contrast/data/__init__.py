import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import ImageFolder
from .sampler import SubsetSlidingWindowSampler


def get_loader(aug_type, args, two_crop=False, prefix='train'):
    transform = get_transform(aug_type, args.crop, args.image_size, args.num_crop,
                              args.crop2, args.image_size2, args.num_crop2)

    # dataset
    if args.zip:
        train_ann_file = prefix + "_map.txt"
        train_prefix = prefix + ".zip@/"
        train_dataset = ImageFolder(args.data_dir, train_ann_file, train_prefix,
                                    transform, two_crop=two_crop, cache_mode=args.cache_mode)
    else:
        train_folder = os.path.join(args.data_dir, prefix)
        train_dataset = ImageFolder(train_folder, transform=transform, two_crop=two_crop)

    # sampler
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    if args.use_sliding_window_sampler:
        sampler = SubsetSlidingWindowSampler(indices,
                                             window_stride=args.window_stride // dist.get_world_size(),
                                             window_size=args.window_size // dist.get_world_size(),
                                             shuffle_per_epoch=args.shuffle_per_epoch)
    elif args.zip and args.cache_mode == 'part':
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(train_dataset)

    # dataloader
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
