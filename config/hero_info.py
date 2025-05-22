

#################################################################################
#                     definition of heter info encoding                         #
#################################################################################
"""
    Encode a data source into a 32-dimensional binary/integer vector.
    
    - Bit 1 (index 0): domain ID
    - Bit 2 (index 1): control frequency
    - Bit 3 (index 2): degrees of freedom (DoF)
    - Bits 4-15 (indices 3-14): one-hot arm model
       ['WidowX', 'Franka', 'Google Robot', 'xArm6', 'Dual Franka', 'Agibot', 'UR5', 'Agilex', 'tienkun', 'AIRBot', 'Dual AIRBot', 'human hand']
    - Bits 16-22 (indices 15-20): one-hot viewpoints
       ['left_side_3rd', 'right_side_3rd', 'top_3rd', 'opposite_3rd','left_wrist', 'right_wrist', 'middle_wrist']
    - Bits 23-26 (indices 21-24): one-hot control mode
       ['rel_eef', 'abs_eef', 'rel_joint', 'abs_joint']
"""


DOMAIN_NAME_TO_INFO = {
    'Bridge-filter': [0,
                      5,
                      7,
                      1,0,0,0,0,0,0,0,0,0,0,0,
                      0,1,0,0,0,0,0,
                      1,0,0,0],
    'Droid-filter': [1,
                    15,
                    7,
                    0,1,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,0,0,0,1,
                    1,0,0,0],
    'language_table': [2,
                    10,
                    2,
                    0,0,0,1,0,0,0,0,0,0,0,0,
                    0,0,1,0,0,0,0,
                    1,0,0,0],
    'Roboset':  [3,
                    5,
                    7,
                    0,1,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,0,0,0,1,
                    1,0,0,0],
    'RT1-filter': [4,
                    3,
                    7,
                    0,0,1,0,0,0,0,0,0,0,0,0,
                    0,1,0,0,0,0,0,
                    1,0,0,0],
    'Calvin_0314': [5,
                    10,
                    7,
                    0,1,0,0,0,0,0,0,0,0,0,0,
                    0,1,0,0,0,0,1,
                    1,0,0,0],
    'Libero': [6,
                    10,
                    7,
                    0,1,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,1,0,0,1,
                    1,0,0,0],
    'Agilex': [7,
               30,
               14,
               0,0,0,0,0,0,0,1,0,0,0,0,
               0,0,1,0,1,1,0,
               0,0,0,1]
} 


import torch
import math
import torch.nn as nn


def sinusoidal_embedding(t, dim, max_period=100):
    """
    Create sinusoidal embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



#################################################################################
#                               HeteroInfoEncoder                               #
#################################################################################

class HeteroInfoEncoder(nn.Module):
    """
    Encodes a raw 32-dimensional source vector into a fixed-dimensional feature vector.

    - First three elements (domain_id, frequency, dof) are encoded using sinusoidal encoding.
    - The remaining elements (one-hot arm model, viewpoints, control mode) are projected via a linear layer.
    """
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size=256
    ):
        super().__init__()
        self.onehot_proj = nn.Linear(23, frequency_embedding_size)
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: Tensor of shape (batch, 32), raw input vector:
               [domain_id, frequency, dof,
                arm_onehot, view_onehot, ctrl_onehot]
        Returns:
            Tensor of shape (batch, dim)
        """
        # Split scalar and one-hot parts
        frequency = x[:, 1].to(torch.int64)
        dof = x[:, 2].to(torch.int64)

        # Sinusoidal encoding for scalar parts
        enc_freq = sinusoidal_embedding(frequency, self.frequency_embedding_size)
        enc_dof = sinusoidal_embedding(dof, self.frequency_embedding_size)
        enc_emb = self.onehot_proj(x[:, 3:].to(self.onehot_proj.weight.dtype)) + enc_freq + enc_dof
        # Combine and return
        return self.mlp(enc_emb)
    


#################################################################################
#                            DomainSpecificLinear                               #
#################################################################################

class ActionLearner(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 hidden_size = 256,
                 num_domains = 20,
                ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Embedding(num_domains, input_size * hidden_size + hidden_size)
        self.fc2 = nn.Embedding(num_domains, hidden_size * output_size + output_size)
        self.act = nn.GELU()
        nn.init.normal_(self.fc1.weight, mean=0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.02)
        
    
    def forward(self, x, hetero_info):
        domain_id = hetero_info[:, 0].to(torch.int64)
        B = domain_id.shape[0]
        fc1_param = self.fc1(domain_id)
        fc1_weight = fc1_param[:, :self.input_size*self.hidden_size].view(B, self.input_size, self.hidden_size)
        fc1_bias = fc1_param[:,self.input_size*self.hidden_size:].view(B, 1, self.hidden_size)
        
        fc2_param = self.fc2(domain_id)
        fc2_weight = fc2_param[:, :self.output_size*self.hidden_size].view(B, self.hidden_size, self.output_size)
        fc2_bias = fc2_param[:,self.output_size*self.hidden_size:].view(B, 1, self.output_size)
        
        x = torch.matmul(x, fc1_weight) + fc1_bias
        x = self.act(x)
        x = torch.matmul(x, fc2_weight) + fc2_bias
        return x