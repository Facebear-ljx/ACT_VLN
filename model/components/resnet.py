import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # 去掉 fc，只保留 backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # (B, 512, 1, 1)
        )

        self.out_dim = 512

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        if x.dim() == 6:
            # B x F x V x C x H x W -> take t=0, view=0
            x = x[:, 0, 0]
        elif x.dim() == 5:
            # B x V x C x H x W -> take view=0
            x = x[:, 0]
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {x.shape}")
        
        assert x.dim() == 4 , f"ResNet expects BCHW, got {x.shape}"
        
        feat = self.backbone(x)
        return feat.flatten(1)  # (B, 512)
