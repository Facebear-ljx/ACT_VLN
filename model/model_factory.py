from timm.models import register_model
from model.policy.ACT import ACTModel


DOMAIN_NAME_TO_ID = {
    'Libero': 0,
    'Calvin_Rel': 1,
    'Bridge': 2,
    "Robotwin": 3,
    "Airbot_agent": 4,
    'Calvin_0314': 1,
    "Calvin_Rel_bear": 1,
    "Agilex": 7
}


@register_model
def ACTAgent(**kwargs):
    model = ACTModel(
        output_dim = 14,
        ac_num = 30,
        # nums_view = 3,
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

@register_model
def ACT_VLNAgent(**kwargs):
    model = ACTModel(
        output_dim = 2,
        ac_num = 10,
        nums_view = 1,
        s_dim = 0,
        hidden_dim = 512,
        dim_feedforward = 3200,
        nheads = 8,
        dropout = 0.1,
        num_encoder_layers = 4,
        num_decoder_layers = 1,
        kl_weight = 10,
    )
    return model

@register_model
def ACT_VLNAgent_v(**kwargs):
    model = ACTModel(
        output_dim = 2,
        ac_num = 10,
        nums_view = 1,
        s_dim = 2,
        hidden_dim = 512,
        dim_feedforward = 3200,
        nheads = 8,
        dropout = 0.1,
        num_encoder_layers = 4,
        num_decoder_layers = 1,
        kl_weight = 10,
    )
    return model