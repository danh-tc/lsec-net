import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LSECNet(nn.Module):
    """
    EfficientNet-B3 backbone + Intrinsic CAM head.
    Returns (logits, feat) where feat = [B, 1536, 7, 7].
    """

    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone   = timm.create_model(
            'efficientnet_b3', pretrained=pretrained,
            num_classes=0, global_pool=''
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        feat   = self.backbone(x)                          # [B, 1536, 7, 7]
        logits = self.classifier(self.dropout(self.gap(feat).flatten(1)))
        return logits, feat

    def get_cam(self, feat, labels, size=(224, 224)):
        """
        Class Activation Map using classifier weights.
        Returns [B, 1, H, W] normalized to [0, 1].
        """
        w   = self.classifier.weight[labels]               # [B, 1536]
        cam = torch.einsum('bchw,bc->bhw', feat, w)
        cam = F.relu(cam)
        B   = cam.shape[0]
        mn  = cam.view(B, -1).min(1)[0].view(B, 1, 1)
        mx  = cam.view(B, -1).max(1)[0].view(B, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return F.interpolate(cam.unsqueeze(1), size, mode='bilinear', align_corners=False)
