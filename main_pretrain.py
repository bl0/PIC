import json
import math
import os
import time
from shutil import copyfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from contrast import models
from contrast import resnet
from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.lr_scheduler import get_scheduler
from contrast.option import parse_option
from contrast.util import AverageMeter
from contrast.lars import add_weight_decay, LARS

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def build_model(args):
    encoder = resnet.__dict__[args.arch]
    model = models.__dict__[args.model](encoder, args).cuda()

    lr = args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        params = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum)
        optimizer = LARS(optimizer)

    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    return model, optimizer


def load_checkpoint(args, model, optimizer, scheduler, sampler=None):
    logger.info(f"=> loading checkpoint '{args.resume}'")

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])
    if args.use_sliding_window_sampler:
        sampler.load_state_dict(checkpoint['sampler'])

    logger.info(f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, sampler=None):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    if args.use_sliding_window_sampler:
        state['sampler'] = sampler.state_dict()
    file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))


def main(args):
    train_loader = get_loader(args.aug, args,
                              two_crop=args.model in ['MoCo', 'SimCLR', 'PIC'],
                              prefix='train')
    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    if args.use_sliding_window_sampler:
        args.warmup_epoch = math.ceil(args.warmup_epoch * len(train_loader.dataset) / args.window_size)
        args.epochs = math.ceil(args.epochs * len(train_loader.dataset) / args.window_size)

    model, optimizer = build_model(args)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    # optionally resume from a checkpoint
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, optimizer, scheduler, sampler=train_loader.sampler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save_checkpoint(args, epoch, model, optimizer, scheduler, sampler=train_loader.sampler)


def train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data = [item.cuda(non_blocking=True) for item in data]

        if args.model == 'PIC':
            # for PIC with multi crop, data[0], data[1], data[2] are x_large, x_small, y_idx
            logit, label = model(data[0], data[1], data[2])
        else:
            # for moco and SimCLR, two_crop=True, data[0] and data[1] are x1, x2
            # for InstDisc, two_crop=False, data[0] and data[1] are x, y_idx
            logit, label = model(data[0], data[1])

        loss = F.cross_entropy(logit, label)

        # backward
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters and print info
        loss_meter.update(loss.item(), data[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        train_len = len(train_loader)
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.3f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')

            # tensorboard logger
            if summary_writer is not None:
                step = (epoch - 1) * len(train_loader) + idx
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', loss_meter.val, step)


if __name__ == '__main__':
    opt = parse_option(stage='pre-train')

    if opt.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(vars(opt))

    main(opt)
