import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class InstDisc(BaseModel):
    """
    Build a InstDisc model with: a encoder, a memory bank
    """

    def __init__(self, base_encoder, args):
        """
        dim: feature dimension (default: 128)
        K: number of negative keys (default: 65536)
        m: momentum of updating memory bank (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(InstDisc, self).__init__(base_encoder, args)

        self.contrast_num_negative = args.contrast_num_negative
        self.contrast_momentum = args.contrast_momentum
        self.contrast_temperature = args.contrast_temperature
        self.num_instances = args.num_instances

        # create the memory
        self.register_buffer('memory', torch.randn(args.num_instances, args.feature_dim))
        self.memory = F.normalize(self.memory, dim=0)

    @torch.no_grad()
    def _momentum_update_memory(self, feature, y_idx):
        memory_pos = torch.index_select(self.memory, 0, y_idx.view(-1))
        memory_pos.mul_(self.contrast_momentum).add_(feature.detach() * (1 - self.contrast_momentum))
        updated_weight = F.normalize(memory_pos)
        self.memory.index_copy_(0, y_idx, updated_weight)

    def forward(self, x, y_idx):
        """
        Input:
            x: a batch of images
            y_idx: index of images
        Output:
            logits, targets
        """

        feature = F.normalize(self.encoder(x), dim=1)

        bs = feature.shape[0]

        # get positive and negative features from memory
        with torch.no_grad():
            # random generate indices of negative sample
            idx = torch.randint(self.num_instances, size=(bs, self.contrast_num_negative+1)).to(feature.device)

            # let first element to be positive sample
            idx[:, 0] = y_idx

            # get weight of positive and negative samples, shape [bs, K+1, dim]
            weight = self.memory[idx]

        # logits: (bs, K+1)
        logits = torch.einsum('bd,bkd->bk', [feature, weight])

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(bs, dtype=torch.long).cuda()

        # momentum update memory
        self._momentum_update_memory(feature, y_idx)

        return logits, labels
