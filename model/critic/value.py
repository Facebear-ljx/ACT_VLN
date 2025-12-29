import torch
import torch.nn as nn

from model.components.resnet import ResNet18Encoder
from model.components.mlp import MLP

class ValueNet(nn.Module):
    def __init__(self,
                 proprio_dim: int = 2,
                 hidden_dim: int = 256,
                 pretrained_vision: bool = False):
        super().__init__()

        self.vision_encoder = ResNet18Encoder(pretrained=pretrained_vision)
        
        in_dim = self.vision_encoder.out_dim + proprio_dim
        # self.ln = nn.LayerNorm(in_dim)
        
        self.projector = MLP(
            in_dim=self.vision_encoder.out_dim + proprio_dim,
            hidden_dim=hidden_dim,
            out_dim=1
        )

    def forward(self, image, proprio):
        """
        image:   (B, 3, H, W)
        proprio: (B, D_p)
        """
        vision_feat = self.vision_encoder(image)
        x = torch.cat([vision_feat, proprio], dim=-1)
        # x = self.ln(x)
        v = self.projector(x)
        return v.squeeze(-1)   # (B,)
    
    
class QNet(nn.Module):
    def __init__(self,
                 proprio_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 pretrained_vision: bool = False):
        super().__init__()

        self.vision_encoder = ResNet18Encoder(pretrained=pretrained_vision)
        # self.ln = nn.LayerNorm(self.vision_encoder.out_dim + proprio_dim + action_dim)
        self.projector = MLP(
            in_dim=self.vision_encoder.out_dim + proprio_dim + action_dim,
            hidden_dim=hidden_dim,
            out_dim=1
        )

    def forward(self, image, proprio, action):
        """
        image:   (B, 3, H, W)
        proprio: (B, D_p)
        action:  (B, D_a)
        """
        vision_feat = self.vision_encoder(image)
        x = torch.cat([vision_feat, proprio, action], dim=-1)
        # x = self.ln(x)
        q = self.projector(x)
        return q.squeeze(-1)   # (B,)
    

class DoubleQNet(nn.Module):
    def __init__(self,
                 proprio_dim: int = 2,
                 action_dim: int = 2,
                 hidden_dim: int = 256,
                 pretrained_vision: bool = False):
        super().__init__()

        self.q1 = QNet(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            pretrained_vision=pretrained_vision
        )

        self.q2 = QNet(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            pretrained_vision=pretrained_vision
        )

    def forward(self, image, proprio, action):
        """
        Returns:
            q1, q2 : (B,), (B,)
        """
        q1 = self.q1(image, proprio, action)
        q2 = self.q2(image, proprio, action)
        return q1, q2

    def min(self, image, proprio, action):
        """
        Convenience function:
            return min(Q1, Q2)
        """
        q1, q2 = self.forward(image, proprio, action)
        return torch.min(q1, q2)


    def max(self, image, proprio, action):
        """
        Convenience function:
            return max(Q1, Q2)
        """
        q1, q2 = self.forward(image, proprio, action)
        return torch.max(q1, q2)    
    

    def mean(self, image, proprio, action):
        """
        Convenience function:
            return mean(Q1, Q2)
        """
        q1, q2 = self.forward(image, proprio, action)
        # tmp = torch.stack([q1, q2])  # shape: (2, B, 1) 或 (2, B)
        return 0.5 * (q1 + q2)
        # return torch.mean([q1, q2], dim=-1, keepdim=True)   
    
if __name__ == "__main__":
    B = 4
    image = torch.randn(B, 3, 224, 224)
    proprio = torch.randn(B, 6)
    action = torch.randn(B, 2)

    dq = DoubleQNet(proprio_dim=6, action_dim=2)

    q1, q2 = dq(image, proprio, action)
    q_min = dq.min(image, proprio, action)

    print(q1.shape, q2.shape, q_min.shape)
    # torch.Size([4]) torch.Size([4]) torch.Size([4])
