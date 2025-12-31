from mmengine import fileio
import numpy as np
import argparse
import io
import h5py
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
if __name__ == '__main__':
    parser = argparse.ArgumentParser('training script', add_help=False)
    
    parser.add_argument('--top_path', type=str) # NOTE!!!!! require absolute path
    
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--action_key', type=str)
    parser.add_argument('--proprio_key', type=str)
    # parser.add_argument('--language_instruction_key', type=str)
    parser.add_argument('--observation_key', nargs='+', type=str)
    
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    print("======starting scan======")
    datalist = fileio.list_dir_or_file(args.top_path, suffix='.hdf5', recursive=True, list_dir=False)
    
    metas = {
        'top_path': args.top_path,
        'dataset_name': args.dataset_name,
        'action_key': args.action_key,
        'proprio_key': args.proprio_key,
        # 'language_instruction_key': args.language_instruction_key,
        'observation_key': args.observation_key,
        'action_statics':{},
        'proprio_stactics':{},
        'datalist': []
    }
    all_action = []
    all_proprio = []
    
    flag = False
    
    for path in tqdm(datalist):
        
        f = io.BytesIO(fileio.get(fileio.join_path(args.top_path,path)))
        data = h5py.File(f,'r')
        if not flag:
            for idx, key in enumerate(args.observation_key):
                img = data[key][-1]
                plt.imsave(f"{args.dataset_name}-{idx}.jpg", cv2.imdecode(img, cv2.IMREAD_COLOR))
            traj_len = data[metas["action_key"]].shape[0]
            assert traj_len == data[metas["proprio_key"]].shape[0] and \
            traj_len == data[metas["observation_key"][0]].shape[0], \
            f"Errors of data structure in {args.dataset_name}-{path}"
            
            
            # print("languague-test:", data["language_instruction"][()].decode())
            flag = True

        try:
            all_action.append(data[args.action_key])
        except:
            print(path)
            break
        all_proprio.append(data[args.proprio_key])
        metas['datalist'].append((path, len(data[args.action_key])))
    
    
    all_action = np.concatenate(all_action).T
    metas['action_statics']['min'] = np.percentile(all_action, 0.1, axis=-1).tolist()
    metas['action_statics']['max'] = np.percentile(all_action, 99.9, axis=-1).tolist()
    metas['action_statics']['mean'] = all_action.mean(axis=-1).tolist()
    metas['action_statics']['std'] = all_action.std(axis=-1).tolist()


    all_proprio = np.concatenate(all_proprio).T
    metas['proprio_stactics']['min'] = np.percentile(all_proprio, 0.1, axis=-1).tolist()
    metas['proprio_stactics']['max'] = np.percentile(all_proprio, 99.9, axis=-1).tolist()
    metas['proprio_stactics']['mean'] = all_proprio.mean(axis=-1).tolist()
    metas['proprio_stactics']['std'] = all_proprio.std(axis=-1).tolist()

    with open(fileio.join_path(args.save_path, f"{args.dataset_name}.json"), "w") as f:
        json.dump(metas, f, indent=4)