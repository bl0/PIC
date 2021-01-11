import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base model with: a encoder
    """

    def __init__(self, base_encoder, args):
        super(BaseModel, self).__init__()

        # create the encoders
        self.encoder = base_encoder(low_dim=args.feature_dim, mlp_head=args.mlp_head)

    def forward(self, x1, x2):
        """
        Input: x1, x2 or x, y_idx
        Output: logits, labels
        """
        raise NotImplementedError
