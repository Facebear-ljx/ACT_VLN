# copy from https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py

import argparse
import collections
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import imageio

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

import json_numpy
import requests
import PIL.Image as Image 

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class ClientModel(CalvinBaseModel):
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

    def step(self, obs, goal):
        """
        Args:
            obs: (dict) environment observations
            goal: (str) language goal 
        Returns:
            action: (np.array) predicted action
        """
        # if len(self.action_plan) < 1:
            # image multi view
        # main_view = obs['rgb_obs']['rgb_static']   # np.ndarray with shape (200, 200, 3)
        # wrist_view = obs['rgb_obs']['rgb_gripper']   # np.ndarray with shape (84, 84, 3)
        
        # # H, W, C = main_view.shape
        # # wrist_view = np.asarray(Image.fromarray(wrist_view).resize((H, W), 3)) # BICUBIC interpolation
        # # image_obs = np.concatenate([main_view[None, ], wrist_view[None, ]], axis=0)  # np.ndarray with shape (2, 200, 200)
        
        # # proprio
        # proprio = obs['robot_obs'][None, ]  # np.ndarray with shape (15,)
        # query = {"domain_name": "Calvin_Rel",
        #         "proprio": json_numpy.dumps(proprio),  # (1, 15)
        #         "language_instruction": goal,
        #         "image0": json_numpy.dumps(main_view),
        #         "image1": json_numpy.dumps(wrist_view),
        #         "do_proprio_normalize": True,
        #         "do_action_denormalize": True}
        
        # response = requests.post(self.url, json=query)
        # action = response.json()
        # # self.action_plan.extend(action)
        # for idx, a in enumerate(action): 
        #     Flag = (self.action_plan[idx] == 0).all()
        #     self.action_plan[idx] = self.action_plan[idx] + np.array(a)
            
        #     if not Flag: self.action_plan[idx] /= 2
                
        # # binary gripper
        # action_predict = self.action_plan.pop(0)
        # self.action_plan.append(np.zeros(7))
        
        # print(self.action_plan)
        if not self.action_plan:
            main_view = obs['rgb_obs']['rgb_static']   # np.ndarray with shape (200, 200, 3)
            wrist_view = obs['rgb_obs']['rgb_gripper']   # np.ndarray with shape (84, 84, 3)
            
            proprio = obs['robot_obs'][None, ]  # np.ndarray with shape (15,)
            query = {"domain_name": "Calvin_Rel",
                    "proprio": json_numpy.dumps(proprio),  # (1, 15)
                    "language_instruction": goal,
                    "image0": json_numpy.dumps(main_view),
                    "image1": json_numpy.dumps(wrist_view),
                    "do_proprio_normalize": True,
                    "do_action_denormalize": True}
            
            response = requests.post(self.url, json=query)
            action = response.json()
            # self.action_plan.extend(action)
            self.action_plan.extend(action)
                
        # binary gripper
        action_predict = np.array(self.action_plan.popleft())
        action_predict[-1] = 1 if action_predict[-1] > 0. else -1
        return action_predict


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[0] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        with open(f"{eval_log_dir}/log.txt", 'a+') as f:
            list_r = count_success(results)
            list_r.append(sum(list_r))
            print(" ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(list_r)]) + "|", file=f)

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success, imgs, lang_annotation = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        save_path = f'{eval_log_dir}/{lang_annotation}_{success}.mp4'
        save_video(save_path, imgs, fps=30)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()
    
    imgs = []
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        
        obs, _, _, current_info = env.step(action)
        
        main_view = obs['rgb_obs']['rgb_static']
        H, W, C = main_view.shape
        wrist_view = np.asarray(Image.fromarray(obs['rgb_obs']['rgb_gripper']).resize((H, W)))
        image_obs = np.concatenate([main_view, wrist_view], axis=1)  # np.ndarray with shape (200, 400, 3)
        imgs.append(image_obs)
        
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True, imgs, lang_annotation
    if debug:
        print(colored("fail", "red"), end=" ")
        
    return False, imgs, lang_annotation

def save_video(save_path, images, fps=30):
    imageio.mimsave(save_path, images, fps=fps)

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", default='/data2/calvin/task_ABC_D', type=str, help="Path to the dataset root directory.")
    parser.add_argument("--host", default='10.140.0.144', help="Your client host ip")
    parser.add_argument("--port", default=8002, type=int, help="Your client port")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument("--eval_log_dir", default='/home/dodo/ljx/HeteroDiffusionPolicy/eval/calvin/results/test', type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    kwargs = vars(args)

    env = make_env(kwargs['dataset_path'])
    model = ClientModel(host=kwargs['host'], port=kwargs['port'])
    evaluate_policy(model, 
                    env, 
                    epoch=None, 
                    eval_log_dir=kwargs['eval_log_dir'], 
                    debug=kwargs['debug'])


if __name__ == "__main__":
    main()