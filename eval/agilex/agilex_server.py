import argparse
import os

from eval.server import DeployServer
from eval.deploy import ModelDeploy

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='single-process evaluation on Agilex robot')
    parser.add_argument('--ckpt_path', default='/home/ljx/ljx/BearRL/exp/20250522/Agilex/ckpt-3600/', type=str, help='load ckpt path')
    parser.add_argument('--model_name', default='ACTAgent', type=str, help='create model name')
    parser.add_argument('--action_norm', default='mean-std', help='min-max or mean-std, action denormalize method')
    parser.add_argument('--qpos_norm', default='mean-std', help='min-max or mean-std, qpos denormalize method')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    args = parser.parse_args()
    kwargs = vars(args)
    
    
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)
    
    # load your model
    model = ModelDeploy(
        ckpt_path = ckpt_path,
        model_name = kwargs['model_name'],
        device = torch.device("cuda"),
        action_normalization=kwargs['action_norm'],
        proprio_normalization=kwargs['qpos_norm']
    )
    
    # wrap the deploymodel as a server
    server = DeployServer(model)
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()