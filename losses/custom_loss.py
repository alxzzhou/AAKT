import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, gamma=1):
        super(CustomLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, target, mask):
        inputs = torch.masked_select(inputs, mask.unsqueeze(-1).repeat(1, 2)).view(-1, 2)
        target = torch.masked_select(target, mask)
        inputs = torch.sigmoid(inputs[:, 1] - inputs[:, 0])

        loss = -(target * torch.log(inputs) + (1 - target) * torch.log(1. - inputs)).sum() / mask.sum()
        return loss
