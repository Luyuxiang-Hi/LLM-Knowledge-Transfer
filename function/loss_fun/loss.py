import torch
from .loss_utils import dice_loss

class SegLoss(torch.nn.Module):
    def __init__(self, bce_coef=0.25, dice_coef=0.75, name='segloss'):

        super(SegLoss, self).__init__()
        self.name = name
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def forward(self, pred_batch, gt_batch):

        dice = dice_loss(pred_batch, gt_batch)
        mean_dice = torch.mean(dice)

        bce_loss_fun = torch.nn.BCELoss()
        mean_bce = bce_loss_fun(pred_batch, gt_batch)

        return self.bce_coef * mean_bce + self.dice_coef * mean_dice
