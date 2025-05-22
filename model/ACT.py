from turtle import position
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import copy


import timm.models
from timm.models import create_model
from einops import repeat, rearrange
from typing import Callable, Optional, Union, Tuple, List, Any

from components.image_encoder import ImageEncoderWithSinePosition


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class ACTModel(nn.Module):
       def __init__(
              self, 
              output_dim: int=14,  # a dim
              ac_num: int=30,
              nums_view: int=3,
              s_dim: int=14,  # qpos_dim
              hidden_dim: int=512,
              dim_feedforward: int=3200,
              nheads: int=8,
              dropout: float=0.1,
              num_encoder_layers: int=4,
              num_decoder_layers: int=1,
              kl_weight: float=10.,
              *args,
              **kwargs
       ):
              """A modified version from ACT implementation

              Args:
                  view_num (int, optional): _description_. Defaults to 3.
                  output_dim (int, optional): _description_. Defaults to 14.
                  ac_num. action chunk size. Defaults to 30 hz for agilex arm
                  s_dim (int, optional): qpos_dim, use qpos when > 0. Defaults to 14
                  hidden_dim (int, optional): _description_. Defaults to 512.
                  dim_feedforward (int, optional): _description_. Defaults to 3200.
                  nheads (int, optional): _description_. Defaults to 8.
                  dropout (float, optional): _description_. Defaults to 0.1.
                  num_encoder_layers (int, optional): _description_. Defaults to 4.
                  num_decoder_layers (int, optional): _description_. Defaults to 1.
                  kl_weight (float, optional). Defaults to 10.
              """
              super().__init__()
              # loss
              self.loss_fn = nn.L1Loss()
              self.kl_weight = kl_weight
              
              # vae transformer encoder
              self.ac_num = ac_num
              self.vae_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                                  nhead=nheads,
                                                                  dim_feedforward=dim_feedforward,
                                                                  dropout=dropout,
                                                                  batch_first=True)
              self.vae_encoder = nn.TransformerEncoder(self.vae_encoder_layer, num_layers=num_encoder_layers)

              # cnn backbone
              self.cnn_encoder = [ImageEncoderWithSinePosition(pretrained=False,
                                                              num_pos_feats=256) for _ in range(nums_view)]
              
              
              # act encoder
              self.act_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                                  nhead=nheads,
                                                                  dim_feedforward=dim_feedforward,
                                                                  dropout=dropout,
                                                                  batch_first=True)
              self.act_encoder = nn.TransformerEncoder(self.act_encoder_layer, num_layers=4)
              
              # act decoder
              self.act_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                                                  nhead=nheads,
                                                                  dim_feedforward=dim_feedforward,
                                                                  dropout=dropout,
                                                                  batch_first=True)
              self.act_decoder = nn.TransformerDecoder(self.act_decoder_layer, num_layers=num_decoder_layers)
              
              # vae encoder extra parameters
              self.s_dim = s_dim
              encode_token_num = 1 + ac_num # [CLS], a_seq
              self.latent_dim = 32 # final size of latent z
              self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
              self.encoder_action_proj = nn.Linear(output_dim, hidden_dim)  # project action to embedding
              if s_dim > 0:
                     self.encoder_joint_proj = nn.Linear(output_dim, hidden_dim)  # project qpos to embedding
                     encode_token_num += 1  # [CLS], qpos, a_seq
              self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
              self.register_buffer('pos_table', get_sinusoid_encoding_table(encode_token_num, hidden_dim))
              
              # act encoder extra parameters
              self.input_image_proj = nn.Conv2d(self.cnn_encoder[0].output_dim, hidden_dim, kernel_size=1) # project image feature to embedding
              self.input_robot_state_proj = nn.Linear(output_dim, hidden_dim) # project qpos to embedding
              self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
              additional_pos_embed_num = 2 if s_dim > 0 else 1 # cam1, cam2, cam3, qpos, [CLS]
              self.additional_pos_embed = nn.Embedding(additional_pos_embed_num, hidden_dim) # learned position embedding for proprio and latent      
              
              # act decoder extra parameters
              self.action_head = nn.Linear(hidden_dim, output_dim)
              self.is_pad_head = nn.Linear(hidden_dim, 1)
              self.query_embed = nn.Embedding(ac_num, hidden_dim)
       
       def latent_vae_encode(self, qpos: torch.Tensor, actions=None, is_pad=None):
              """
              qpos: batch, qpos_dim
              image: batch, num_cam, channel, height, width
              actions: batch, seq, action_dim,
              is_pad: batch, seq
              """
              is_training = actions is not None # train or val
              bs, _, _ = actions.shape
              if is_training:
                     # project action sequence to embedding dim, and concat with a CLS token
                     action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                     cls_embed = self.cls_embed.weight # (1, hidden_dim)
                     cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                     if self.s_dim > 0:
                            # use qpos information
                            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1+1, hidden_dim)
                     else:
                            # do not use qpos information
                            encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                     
                     # do not mask cls token
                     cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) if self.s_dim > 0 else torch.full((bs, 1), False).to(qpos.device)  # False: not a padding
                     is_pad = torch.full((bs, self.ac_num), False).to(qpos.device) # False: not a padding
                     is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1+1)
                     
                     # obtain position embedding
                     pos_embed = self.pos_table.clone().detach().repeat(bs, 1, 1) # (bs, seq+1+1, hidden_dim)
                     
                     # query model
                     encoder_input = encoder_input + pos_embed  # only add pos_embedding before atten, not exactly the same as original ACT model
                     encoder_output = self.vae_encoder(encoder_input, src_key_padding_mask=is_pad)
                     encoder_output = encoder_output[:, 0] # take cls output only
                     latent_info = self.latent_proj(encoder_output)
                     mu = latent_info[:, :self.latent_dim]
                     logvar = latent_info[:, self.latent_dim:]
                     latent_sample = reparametrize(mu, logvar)
                     latent_input = self.latent_out_proj(latent_sample)
              else:
                     mu = logvar = None
                     latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                     latent_input = self.latent_out_proj(latent_sample)
              return latent_input, mu, logvar      
       
       
       def encode_image(self, image: torch.Tensor):
              # Image observation features and position embeddings
              B, F, V, C, H, W = image.shape
              image = image.view(B*F, V, C, H, W)
              image_features = []
              for i in range(V):
                     image_feature, pos = self.cnn_encoder[i](image[:, i, :, :, :])
                     image_features.append(image_feature.unsqueeze(1))
              
              image_features = torch.cat(image_features, dim=1)  # B*F, V, C, H, W
              BF, _, C, H, W = image_features.shape
              image_features = image_features.view(BF*V, C, H, W)  # B*F*V, C, H, W
              
              image_features = self.input_image_proj(image_features)
              pos = pos[0].repeat(F*V, 1, 1, 1)
              
              # fold camera dimension into width dimension
              src = rearrange(image_features, '(b f v) c h w -> (b f) c h (w v)', v=V, f=F)
              pos = rearrange(pos, '(f v) c h w -> f c h (w v)', v=V, f=F)
              return src, pos
       
       def encode(self, src, latent_input, proprio_input, pos_embed):
              # flatten NxCxHxW to HWxNxC
              bs, c, h, w = src.shape  # B, C, H, W
              src = src.flatten(2).permute(0, 2, 1)  # B, C, H, W -> B, C, H*W -> B, H*W, C
              pos_embed = pos_embed.flatten(2).permute(0, 2, 1).repeat(bs, 1, 1) # 1, C, H, W -> 1, C, H*W -> 1, H*W, C -> B, H*W, C

              additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(0).repeat(bs, 1, 1) # bs, 2, dim
              pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=1)  # bs, 2+seq, dim

              addition_input = torch.stack([latent_input, proprio_input], axis=1) if proprio_input is not None else latent_input.unsqueeze(1)
              src = torch.cat([addition_input, src], axis=1)
              
              src = src + pos_embed
              memory = self.act_encoder(src)
              return memory

       def decode(self, memory, padding_mask):
              bs, _, _ = memory.shape
              query_embed = self.query_embed.weight  # n_chunk, dim
              query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)  # n_chunk, dim -> 1, n_chunk, dim -> bs, n_chunk, dim
              output = self.act_decoder(query_embed, memory, tgt_key_padding_mask=padding_mask)
              return output
       
       
       def forward(self, qpos: torch.Tensor, image_obs: torch.Tensor, action: torch.Tensor, is_pad=None):
              # vae encode, get style latent
              latent_input, mu, logvar = self.latent_vae_encode(qpos, action, is_pad)

              # image & proprioception features
              src_image, pos = self.encode_image(image_obs)
              proprio_input = self.input_robot_state_proj(qpos) if self.s_dim > 0 else None

              # act main branch encoder decoder
              memory = self.encode(src_image, latent_input, proprio_input, pos)
              output = self.decode(memory, is_pad)
              a_hat = self.action_head(output)
              
              # loss
              total_kld, _, _ = kl_divergence(mu, logvar)
              recons_loss = self.loss_fn(action, a_hat)
              kl_loss = self.kl_weight * total_kld[0]
              loss = recons_loss + kl_loss
              loss_dict = {"policy_loss": loss,
                           "recons_loss": recons_loss,
                           "kl_loss": kl_loss}
              return loss_dict

       @torch.no_grad
       def get_action(self, qpos: torch.Tensor, image: torch.Tensor):
              is_pad = torch.full((1, self.ac_num), False)
              # vae encode, get style latent
              latent_input, _, _ = self.latent_vae_encode(qpos, None, is_pad)

              # image & proprioception features
              src_image, pos = self.encode_image(image)
              proprio_input = self.input_robot_state_proj(qpos) if self.s_dim > 0 else None

              # act main branch encoder decoder
              memory = self.encode(src_image, latent_input, proprio_input, pos)
              output = self.decode(memory, is_pad)
              a_hat = self.action_head(output) # 1, n_chunk, a_dim
              return a_hat.squeeze(0).detach().cpu().numpy()


if __name__ == '__main__':
       model = ACTModel()
       example_input_image = torch.zeros((8, 1, 3, 3, 224, 224))
       example_input_qpos = torch.zeros((8, 14))
       example_action = torch.ones((8, 30, 14))
       is_pad = torch.full((8, 30), False)
       
       loss = model(example_input_qpos, example_input_image, example_action, is_pad)