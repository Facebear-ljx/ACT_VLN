# copy from https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py

import argparse
import collections
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time

from termcolor import colored
import torch
from tqdm.auto import tqdm
import numpy as np

import json_numpy
import requests
import PIL.Image as Image 

logger = logging.getLogger(__name__)



class ClientModel():
    def __init__(self,
                 host,
                 port):

        self.url = f"http://{host}:{port}/act"
        self.reset()
        
    def reset(self):
        """
        This is called
        """
        # currently, we dont use historical observation, so we dont need this fc
        
        self.action_plan = collections.deque()
        return None

    def step(self, obs):
        """
        Args:
            obs: (dict) environment observations
        Returns:
            action: (np.array) predicted action
        """
        if not self.action_plan:
            main_view = obs['images']['cam_high']   #  np.ndarray with shape (480, 640, 3)
            left_wrist_view = obs['images']['cam_left_wrist']   # np.ndarray with shape (480, 640, 3) 
            right_wrist_view = obs['images']['cam_right_wrist']   # np.ndarray with shape (480, 640, 3) 
            
            proprio = obs['qpos'][None, ].astype(np.float32)  # np.ndarray with shape (14,)
            query = {"proprio": json_numpy.dumps(proprio),  # (14, )
                    "image0": json_numpy.dumps(main_view),
                    "image1": json_numpy.dumps(left_wrist_view),
                    "image2": json_numpy.dumps(right_wrist_view),}
                
            response = requests.post(self.url, json=query)
            action = response.json()
            self.action_plan.extend(action)
                
        # binary gripper
        action_predict = np.array(self.action_plan.popleft())
        # action_predict[-1] = 1 if action_predict[-1] > 0. else -1
        return action_predict
    
def generate_test_obs():
    """
    Generate a test observation dictionary for the step() function.
    
    Returns:
        obs (dict): A dictionary with the required structure for testing step().
                   Contains:
                   - 'images': dict with three camera views (converted to (3, H, W))
                   - 'qpos': joint positions (shape (14,))
    """
    # Create dummy image data (480x640x3) and convert to (3, H, W)
    def create_dummy_image():
        img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        return img
    
    # Create observation dictionary
    obs = {
        'images': {
            'cam_high': create_dummy_image(),
            'cam_left_wrist': create_dummy_image(),
            'cam_right_wrist': create_dummy_image()
        },
        'qpos': np.random.uniform(-1.0, 1.0, size=14).astype(np.float32)  # Random joint positions
    }
    
    return obs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8030, type=int, help="Your client port")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    kwargs = vars(args)

    policy = ClientModel(kwargs['host'], kwargs['port'])
    policy.step(generate_test_obs())


if __name__ == "__main__":
    main()