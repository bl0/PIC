import torch
import numpy as np
from PIL import ImageFilter
from torchvision import transforms
from .rand_augment import rand_augment_transform


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultiTransform(object):
    """apply transform to an image for k times"""
    def __init__(self, transform, k=3):
        self.transform = transform
        self.k = k

    def __call__(self, img):
        return torch.stack([self.transform(img) for i in range(self.k)])


def get_transform(aug_type, crop, image_size=224, num_crop=1,
                  crop2=0.14, image_size2=96, num_crop2=3):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if aug_type == "InstDisc":  # used in InstDisc and MoCo v1
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'MoCov2':  # used in MoCov2
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    elif aug_type == 'SimCLR':  # used in SimCLR and PIC
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'MultiCrop':  # used in PIC_MultiCrop
        assert crop < crop2
        transform1 = MultiTransform(
            transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(crop2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.ToTensor(),
                normalize,
            ]),
            k=num_crop)
        transform2 = MultiTransform(
            transforms.Compose([
                transforms.RandomResizedCrop(image_size2, scale=(crop, crop2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.ToTensor(),
                normalize,
            ]),
            k=num_crop2)
        transform = (transform1, transform2)
    elif aug_type == 'RandAug':  # used in InfoMin
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'NULL':  # used in linear evaluation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'val':  # used in validate
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
    else:
        supported = '[InstDisc, MoCov2, SimCLR, RandAug, Null, val]'
        raise NotImplementedError(f'aug_type "{aug_type}" not supported. Should in {supported}')

    return transform
