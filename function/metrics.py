import torch


def compute_batch_metrics(gt, pred):  

    assert pred.shape == gt.shape, "预测和真实值的形状不匹配"
    pred = pred.reshape(-1).long()
    gt = gt.reshape(-1).long()
    
    assert torch.all((gt == 0) | (gt == 1)), "gt 只能包含 0 和 1"
    assert torch.all((pred == 0) | (pred == 1)), "pred 只能包含 0 和 1"

    tp = (gt*pred).sum()
    fp = ((pred-gt)==1).sum()
    fn = ((gt-pred)==1).sum()
    tn = ((gt+pred)==0).sum()

    IoU        = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else torch.tensor(-1, dtype=torch.float)
    Recall   = tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(-1, dtype=torch.float)
    Precision   = tp / (tp + fp) if (tp + fp) != 0 else torch.tensor(-1, dtype=torch.float)
    Accuracy = (tp + tn) / (tp +fp + fn + tn) if (tp +fp + fn + tn) != 0 else torch.tensor(-1, dtype=torch.float)
    F1sorce = 2*Precision*Recall/(Precision+Recall)


    return IoU, Precision, Accuracy, Recall, F1sorce



