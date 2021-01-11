import argparse
import os

from contrast import resnet
from contrast.util import MyHelpFormatter


model_names = sorted(name for name in resnet.__all__
                     if name.islower() and callable(resnet.__dict__[name]))


def parse_option(stage='pre-train'):
    """ configs for pre-train or linear stage
    """
    parser = argparse.ArgumentParser(f'contrast {stage} stage', formatter_class=MyHelpFormatter)

    # dataset
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--crop', type=float, default=0.2 if stage == 'pre-train' else 0.08, help='minimum crop')
    parser.add_argument('--aug', type=str, default='NULL', choices=['NULL', 'InstDisc', 'MoCov2', 'SimCLR', 'RandAug', 'MultiCrop'],
                        help='which augmentation to use.')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')

    parser.add_argument('--num-workers', type=int, default=4, help='num of cpu workers per GPU to use')
    if stage == 'linear':
        parser.add_argument('--total-batch-size', type=int, default=256, help='total train batch size for all GPU')
    else:
        parser.add_argument('--batch-size', type=int, default=64, help='batch_size for single gpu')
    # sliding window sampler
    parser.add_argument('--window-size', type=int, default=131072, help='window size in sliding window sampler')
    parser.add_argument('--window-stride', type=int, default=16384, help='window stride in sliding window sampler')
    parser.add_argument('--use-sliding-window-sampler', action='store_true',
                        help='whether to use sliding window sampler')
    parser.add_argument('--shuffle-per-epoch', action='store_true',
                        help='shuffle indices in sliding window sampler per epoch')
    # multi crop
    parser.add_argument('--image-size', type=int, default=224, help='crop size')
    parser.add_argument('--image-size2', type=int, default=96, help='small crop size (for MultiCrop)')
    parser.add_argument('--crop2', type=float, default=0.14,
                        help='minimum crop for large crops, maximum crop for small crops')
    parser.add_argument('--num-crop', type=int, default=1, help='number of crops')
    parser.add_argument('--num-crop2', type=int, default=3, help='number of small crops')

    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names,
                        help="backbone architecture")
    if stage == 'pre-train':
        parser.add_argument('--model', type=str, default='PIC', choices=['PIC', 'MoCo', 'SimCLR', 'InstDisc'],
                            help='which model to use')
        parser.add_argument('--contrast-temperature', type=float, default=0.07, help='temperature in instance cls loss')
        parser.add_argument('--contrast-momentum', type=float, default=0.999,
                            help='momentume parameter used in MoCo and InstDisc')
        parser.add_argument('--contrast-num-negative', type=int, default=65536,
                            help='number of negative samples used in MoCo and InstDisc')
        parser.add_argument('--feature-dim', type=int, default=128, help='feature dimension')
        parser.add_argument('--mlp-head', action='store_true', help='use mlp head')

    # optimization
    if stage == 'pre-train':
        parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.03,
                            help='base learning when batch size = 256. final lr is determined by linear scale')
        parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'lars'],
                            help='optimizer in pre-train stage')
    else:
        parser.add_argument('--learning-rate', type=float, default=30, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4 if stage == 'pre-train' else 0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # misc
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='whether auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=100, help='print message frequency (iteration)')
    parser.add_argument('--save-freq', type=int, default=10, help='save checkpoint frequency (epoch)')
    parser.add_argument("--local_rank", type=int, required=True,
                        help='local rank for DistributedDataParallel, required by pytorch DDP')
    if stage == 'linear':
        parser.add_argument('--pretrained-model', type=str, required=True, help="path to the pretrained model")
        parser.add_argument('-e', '--eval', action='store_true', help='only evaluate')

    args = parser.parse_args()

    return args
