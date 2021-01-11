import torch
import torch.nn as nn
import torch.nn.functional as F

from contrast.util import dist_collect
from .base import BaseModel

LARGE_NUM = 1e9


class SimCLR(BaseModel):
    """
    Build a SimCLR model with: a encoder and syncBN
    """

    def __init__(self, base_encoder, args):
        """
        dim: feature dimension (default: 128)
        T: softmax temperature (default: 0.07)
        """
        super(SimCLR, self).__init__(base_encoder, args)

        self.contrast_temperature = args.contrast_temperature

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

    def forward(self, x1, x2):
        """
        Input:
            x1: a batch of first augmentation images
            x2: a batch of second augmentation images
        Output:
            logit, label
        """

        # compute features
        f1 = F.normalize(self.encoder(x1), dim=1)
        f2 = F.normalize(self.encoder(x2), dim=1)

        # gather features from all gpus
        batch_size_this = f1.size(0)
        f1_gather = dist_collect(f1)
        f2_gather = dist_collect(f2)
        batch_size_all = f1_gather.size(0)

        # compute mask
        gpu_index = torch.distributed.get_rank()
        label_index = torch.arange(batch_size_this) + gpu_index * batch_size_this
        mask = torch.zeros(batch_size_this, batch_size_all)
        mask.scatter_(1, label_index.view(-1, 1), 1)
        mask = mask.cuda()

        # compute logit
        logit_aa = torch.mm(f1, f1_gather.T) / self.contrast_temperature
        logit_aa = logit_aa - mask * LARGE_NUM
        logit_bb = torch.mm(f2, f2_gather.T) / self.contrast_temperature
        logit_bb = logit_bb - mask * LARGE_NUM
        logit_ab = torch.mm(f1, f2_gather.T) / self.contrast_temperature
        logit_ba = torch.mm(f2, f1_gather.T) / self.contrast_temperature

        logit_a = torch.cat([logit_ab, logit_aa], dim=1)
        logit_b = torch.cat([logit_ba, logit_bb], dim=1)
        logit = torch.cat([logit_a, logit_b], dim=0)

        # compute label
        label = label_index.repeat(2).cuda()

        return logit, label
