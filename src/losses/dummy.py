import torch
from src.functional.dummy import l2_criterion


class DummyLoss(torch.nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()
        self.criterion = l2_criterion

    def forward(self, pred, gt):
        return self.criterion(pred, gt)
