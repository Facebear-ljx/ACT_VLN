import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from eval.deploy import ModelDeploy


# === Server Interface ===
class DeployServer:
    def __init__(
            self, 
            model: ModelDeploy,
        ):
        self.model = model
        
    def infer(self, payload: Dict[str, Any]):
        try:    
            image_list = []        
            
            if "image0" in payload.keys():
                image_list.append(Image.fromarray(json.loads(payload["image0"])))
            if "image1" in payload.keys():
                image_list.append(Image.fromarray(json.loads(payload["image1"])))
            if "image2" in payload.keys():
                image_list.append(Image.fromarray(json.loads(payload["image2"])))
            inputs = {
                # 'language_instruction': payload['language_instruction'],
                'image_obs': image_list,
                # 'domain_name': payload['domain_name'],
            }
            if "proprio" in payload.keys():
                inputs["proprio"] = json.loads(payload["proprio"])
                
            action = self.model.infer(**inputs)
            action_list = action.tolist()
            return JSONResponse(action_list)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            warning_str = "Your request threw an error; make sure your request complies with the expected Dict format:\n {'domain_name': str, e.g. 'Libero', see model/model_factory.py for details, \n'language_instruction': str, \n'image_obs': np.ndarray with shape [V, H, W, C], \n'proprio': np.ndarray with shape [B, C], \n'do_proprio_normalize': bool, \n'do_action_denormalize': bool}"
            logging.warning(
                warning_str
            )
            return warning_str

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.infer)
        uvicorn.run(self.app, host=host, port=port)
        

if __name__ == '__main__':
    # one example to deploy a model as a server
    
    ip = '0.0.0.0' # your host ip here
    port = 8000  # your port here
    
    model = ModelDeploy(
        ckpt_path = '/home/ljx/ljx/BearRL/exp/20250522/Agilex/ckpt-3600/', # your model ckpt here
        model_name = 'ACTAgent', # "siglip_base_depth_6_hidden_512_flow_matching" for flow matching
        device = "cuda",

        ## default settings, 
        action_normalization = "min-max",
        proprio_normalization = "mean-std",
    )
    
    server = DeployServer(model=model)
    server.run(host=ip, port=port)