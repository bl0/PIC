import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class PIC(BaseModel):
    """
    Build a simple PIC model with multi crop
    """

    def __init__(self, base_encoder, args):
        """
        dim: feature dimension (default: 128)
        m: momentum of updating memory bank (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(PIC, self).__init__(base_encoder, args)

        self.contrast_temperature = args.contrast_temperature
        self.num_instances = args.num_instances

        self.sim_matrix = nn.Parameter(torch.randn(size=(args.num_instances, args.feature_dim)), requires_grad=True)
        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x, x_small, y_idx):
        """
        Input:
            x: a batch of images
            x_small: a batch of images(small crops)
            y_idx: index of images
        Output:
            logits, targets
        """

        # large crops
        k = x.shape[1]
        if k == 1:
            x_list = [torch.squeeze(x, 1)]
        else:
            x_list = [torch.squeeze(x[:, i, ...], 1).contiguous() for i in range(k)]

        feature_list = [F.normalize(self.encoder(x), dim=1) for x in x_list]

        sim_matrix = F.normalize(self.sim_matrix, dim=1)

        logit_list = [torch.einsum('nc,kc->nk', [f, sim_matrix]) for f in feature_list]

        logit_list = [logit / self.contrast_temperature for logit in logit_list]

        # small crops
        k_small = x_small.shape[1]
        if k_small == 1:
            x_small_list = [torch.squeeze(x_small, 1)]
        else:
            x_small_list = [torch.squeeze(x_small[:, i, ...], 1).contiguous() for i in range(k_small)]

        feature_small_list = [F.normalize(self.encoder(x), dim=1) for x in x_small_list]

        # block gradient of sim_matrix
        sim_matrix_detach = sim_matrix.detach()

        logit_small_list = [torch.einsum('nc,kc->nk', [f, sim_matrix_detach]) for f in feature_small_list]

        logit_small_list = [logit / self.contrast_temperature for logit in logit_small_list]

        logit = torch.cat(logit_list + logit_small_list, dim=0)

        return logit, y_idx.repeat(k + k_small)
