import torch

def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    """

    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param smooth:
    @param eps:
    @return: (N, C)
    """
    numerator = 2 * torch.sum(y_true * y_pred, dim=(-1, -2))
    denominator = torch.sum(y_true, dim=(-1, -2)) + torch.sum(y_pred, dim=(-1, -2))
    return 1 - (numerator + smooth) / (denominator + smooth + eps)