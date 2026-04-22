import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1e-6):
    """
    Soft Dice loss between CAM and GT mask.
    Minimized when CAM overlaps well with mask (high precision + recall).
    """
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    return 1 - ((2 * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()


def outside_loss(cam, mask):
    """
    Penalizes CAM activation outside the GT mask (false-positive region).
    """
    return (cam * (1 - mask)).mean()


class LSECLoss(nn.Module):
    """
    L_total = L_cls + λ1 * L_align + λ2 * L_out

    L_cls   : CrossEntropy (+ label smoothing) — learn correct class
    L_align : Dice(CAM, mask) — CAM must overlap with lesion
    L_out   : mean(CAM * (1-mask)) — CAM must not leak outside lesion

    Baseline: λ1=0, λ2=0 → only L_cls
    Proposed: λ1=1.0, λ2=0.3
    """

    def __init__(self, lambda1=1.0, lambda2=0.3, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.l1 = lambda1
        self.l2 = lambda2

    def forward(self, logits, feat, model, labels, masks):
        L_cls      = self.ce(logits, labels)
        non_normal = (labels != 0).nonzero(as_tuple=True)[0]

        if len(non_normal) == 0 or (self.l1 == 0 and self.l2 == 0):
            return L_cls, {'cls': L_cls.item(), 'align': 0.0, 'out': 0.0}

        cam     = model.get_cam(feat[non_normal], labels[non_normal])
        L_align = dice_loss(cam, masks[non_normal])
        L_out   = outside_loss(cam, masks[non_normal])
        total   = L_cls + self.l1 * L_align + self.l2 * L_out

        return total, {
            'cls':   L_cls.item(),
            'align': L_align.item(),
            'out':   L_out.item(),
        }
