import torch
import torch.nn as nn
import math
from torchvision.models import resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor   # B, C, H, W

        not_mask = torch.ones_like(x[0, [0]])  # 1, C, H, W
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # 1, C, H, W
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # 1, C, H, W 
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # 1, C, H, W
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        return pos

class ImageEncoderWithSinePosition(nn.Module):
    def __init__(self, 
                 pretrained=False, 
                 num_pos_feats=256):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone = replace_bn_with_frozen(self.backbone)
        self.output_dim = self.backbone.fc.in_features
        del self.backbone.avgpool
        del self.backbone.fc
        
        self.position_embedder = PositionEmbeddingSine(
            num_pos_feats=num_pos_feats,
            temperature=10000,
            normalize=True,
            scale=2*math.pi
        )
        
    def encode_image_wo_pool(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def forward(self, x):
        # [B, 3, H, W] -> [B, 512, H, W]
        features = self.encode_image_wo_pool(x)
        
        # [B, 512, H, W]
        pos_embedding = self.position_embedder(features)
        
        return features, pos_embedding


def replace_bn_with_frozen(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            frozen_bn = FrozenBatchNorm2d(
                num_features=module.num_features,
                eps=module.eps
            )
            
            frozen_bn.load_state_dict(module.state_dict())
            
            setattr(model, name, frozen_bn)
        else:
            replace_bn_with_frozen(module)
    return model


if __name__ == "__main__":
    encoder = ImageEncoderWithSinePosition()
    dummy_input = torch.randn(8, 3, 224, 224)  # [B, C, H, W]
    output = encoder(dummy_input)
    
    print(f"input size: {dummy_input.shape}")
    print(f"output size: {output.shape}")  # [8, 512, 7, 7]