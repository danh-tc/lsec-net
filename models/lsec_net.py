import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LSECNet(nn.Module):
    """
    ConvNeXt-Tiny backbone + Intrinsic CAM head.
    Returns (logits, feat) where feat = [B, 768, 7, 7].

    weights_path: optional path to a pretrained backbone checkpoint.
                  Loaded with strict=False so partial state_dicts work.
    """

    def __init__(self, num_classes=3, pretrained=True, dropout=0.3, weights_path=None):
        super().__init__()
        self.backbone   = timm.create_model(
            'convnext_tiny', pretrained=pretrained,
            num_classes=0, global_pool=''
        )
        if weights_path is not None:
            state = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state:
                state = state['state_dict']
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"Loaded backbone weights: {weights_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        feat_dim        = self.backbone.num_features       # 768 for ConvNeXt-Tiny
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat   = self.backbone(x)                          # [B, 768, 7, 7]
        logits = self.classifier(self.dropout(self.gap(feat).flatten(1)))
        return logits, feat

    def get_cam(self, feat, labels, size=(224, 224)):
        """
        Class Activation Map using classifier weights.
        Returns [B, 1, H, W] normalized to [0, 1].
        """
        w   = self.classifier.weight[labels]               # [B, feat_dim]
        cam = torch.einsum('bchw,bc->bhw', feat, w)
        cam = F.relu(cam)
        B   = cam.shape[0]
        mn  = cam.view(B, -1).min(1)[0].view(B, 1, 1)
        mx  = cam.view(B, -1).max(1)[0].view(B, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return F.interpolate(cam.unsqueeze(1), size, mode='bilinear', align_corners=False)
