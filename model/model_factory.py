from timm.models import register_model
from model.ACT import ACTModel


@register_model
def ACTAgent(**kwargs):
    model = ACTModel(
        output_dim = 14,
        ac_num = 30,
        nums_view = 3,
        s_dim = 14,
        hidden_dim = 512,
        dim_feedforward = 3200,
        nheads = 8,
        dropout = 0.1,
        num_encoder_layers = 4,
        num_decoder_layers = 1,
        kl_weight = 10,
    )
    return model