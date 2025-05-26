import model.model_factory
from model.model_factory import DOMAIN_NAME_TO_ID
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm import create_model
import torch

from safetensors.torch import load_file
import numpy as np
import PIL.Image as Image

STATICS = {
    "Agilex": {
        "action_statics": {
        "min": [
            -0.9151122570037842,
            1.1071183681488037,
            -2.200299024581909,
            -0.3019382059574127,
            0.4741770029067993,
            -0.5739424824714661,
            0.001820000004954636,
            -0.07959697395563126,
            0.9255088567733765,
            -2.037510871887207,
            -0.1747092306613922,
            0.4894960820674896,
            -0.1265438050031662,
            0.000910000002477318
        ],
        "max": [
            0.01423079427331686,
            2.4604506492614746,
            -0.6342812776565552,
            0.30714985728263855,
            1.2363158464431763,
            0.33162784576416016,
            0.06894999742507935,
            0.9327830076217651,
            2.3305532932281494,
            -0.5803095698356628,
            0.44600623846054077,
            1.1629598140716553,
            0.7236294746398926,
            0.06909000128507614
        ],
        "mean": [
            -0.3930080831050873,
            1.6363461017608643,
            -1.2088271379470825,
            -0.03358646482229233,
            0.8590826988220215,
            -0.15526697039604187,
            0.04629657045006752,
            0.2819295823574066,
            1.432823657989502,
            -1.0165048837661743,
            -0.0066304984502494335,
            0.8095815777778625,
            0.2872689962387085,
            0.04664315655827522
        ],
        "std": [
            0.2522423565387726,
            0.2817229926586151,
            0.3015429973602295,
            0.09937842935323715,
            0.11391963064670563,
            0.11769859492778778,
            0.0231929961591959,
            0.20969651639461517,
            0.30106422305107117,
            0.25484684109687805,
            0.1329314410686493,
            0.09547023475170135,
            0.15177997946739197,
            0.017285047098994255
        ]
        },
        "proprio_stactics": {
        "min": [
            -0.9151122570037842,
            1.1071183681488037,
            -2.200299024581909,
            -0.3019382059574127,
            0.4741770029067993,
            -0.5739424824714661,
            0.001820000004954636,
            -0.07959697395563126,
            0.9255088567733765,
            -2.037510871887207,
            -0.1747092306613922,
            0.4894960820674896,
            -0.1265438050031662,
            0.000910000002477318
        ],
        "max": [
            0.01423079427331686,
            2.4604506492614746,
            -0.6342812776565552,
            0.3073715269565582,
            1.2363158464431763,
            0.33162784576416016,
            0.06894999742507935,
            0.9327830076217651,
            2.3305532932281494,
            -0.5803095698356628,
            0.44600623846054077,
            1.1629598140716553,
            0.7236294746398926,
            0.06909000128507614
        ],
        "mean": [
            -0.39296168088912964,
            1.636264443397522,
            -1.2087311744689941,
            -0.03356923907995224,
            0.8589358329772949,
            -0.15534070134162903,
            0.04628966748714447,
            0.2818305790424347,
            1.4326094388961792,
            -1.0162822008132935,
            -0.006646266672760248,
            0.809569776058197,
            0.2872677147388458,
            0.04664047807455063
        ],
        "std": [
            0.25218045711517334,
            0.28174611926078796,
            0.3015180230140686,
            0.09943129867315292,
            0.11392149329185486,
            0.11780452728271484,
            0.023189403116703033,
            0.2096998244524002,
            0.3010607361793518,
            0.2547939121723175,
            0.13294927775859833,
            0.09550005942583084,
            0.15179471671581268,
            0.01728242263197899
        ]
        },
    }
}


class ModelDeploy:
    def __init__(self,
                 ckpt_path,
                 model_name = "ACTAgent",
                 device = "cuda",
                 
                 action_normalization = "min-max",
                 proprio_normalization = "mean-std"
                 ):
        
        
        self.model = create_model(model_name)
        
        self.proprio_normalization = proprio_normalization
        self.action_normalization = action_normalization
        
        loaded_data = load_file(ckpt_path)
        print(self.model.load_state_dict(loaded_data))
        
        self.device = device
        self.model.to(device)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        
        print("current support domains:", DOMAIN_NAME_TO_ID.keys())

    @torch.no_grad()
    def infer(self,
              image_obs, # List[Image]
              proprio: np.ndarray = None, # F C 
              domain_name:str = "Agilex",      
              
              do_action_denormalize = True,
              do_proprio_normalize = True,         
              ):
        self.model.eval()
        statics = STATICS[domain_name]
        
        inputs = {
            'image_obs': torch.stack([self.image_transform(img).unsqueeze(0) for img in image_obs]).unsqueeze(0).to(self.device),
        }
        if proprio is not None:
            if do_proprio_normalize:
                if self.proprio_normalization == 'mean-std':
                    proprio = (proprio - np.array(statics['proprio_stactics']['mean'])[None,]) / (np.array(statics['proprio_stactics']['std'])[None,] + 1e-6)
                else: raise NotImplementedError
        
            inputs['qpos'] = torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(self.device)


        with torch.no_grad():
            action_pred = self.model.pred_action(**inputs).detach().squeeze(0).cpu().numpy()
        
        if do_action_denormalize:
            if self.action_normalization == 'min-max':
                action_pred = (action_pred + 1) / 2
                action_pred = action_pred * (np.array(statics['action_statics']['max'])[None,] - np.array(statics['action_statics']['min'])[None,] + 1e-6) + \
                                np.array(statics['action_statics']['min'])[None,]
            elif self.action_normalization == 'mean-std':
                action_pred = action_pred * (np.array(statics['action_statics']['std'])[None,] +1e-6) + np.array(statics['action_statics']['mean'])[None,]
            else: raise NotImplementedError
        
        return action_pred