#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
from pathlib import Path

import argparse
import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio

from model.critic.value import DoubleQNet, ValueNet


# ============================================================
# Utils
# ============================================================

def random_shifts_aug(x: torch.Tensor, pad: int = 30):
    x = x.float()
    c, h, w = x.size()
    padding = (pad, pad, pad, pad)
    x = F.pad(x, padding, "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    shift = torch.randint(0, 2 * pad + 1, size=(1, 1, 2), device=x.device)
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x.unsqueeze(0), grid.unsqueeze(0),
                         padding_mode="zeros", align_corners=False).squeeze(0)


def build_image_transform(train_aug: bool):
    ops = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225), inplace=True),
    ]
    if train_aug:
        ops.insert(1, transforms.ColorJitter(0.2, 0.2, 0.2, 0.0))
    return transforms.Compose(ops)


def decode_image_bytes_to_tensor(img_bytes, image_tf, device, use_random_shift):
    img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = image_tf(Image.fromarray(img_rgb))
    if use_random_shift:
        x = random_shifts_aug(x)
    return x.unsqueeze(0).to(device), img_bgr


# ============================================================
# Meta
# ============================================================
def load_all_metas(metas_path: str):
    metas = []

    for file in fileio.list_dir_or_file(
        metas_path,
        list_dir=False,     # ⭐ 关键：不列目录
        list_file=True,     # ⭐ 只列文件
        suffix=".json",
        recursive=True,
    ):
        with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
            meta = json.load(f)
        metas.append(meta)

    if not metas:
        raise RuntimeError(f"No meta json found under metas_path={metas_path}")

    return metas


def find_meta_for_episode(metas, episode_path):
    return max(
        (m for m in metas if episode_path.startswith(m.get("top_path", ""))),
        key=lambda m: len(m["top_path"])
    )


# ============================================================
# Normalization
# ============================================================

def normalize(x, stat, mode):
    x = np.asarray(x, np.float32)
    if mode == "mean-std":
        return (x - stat["mean"]) / (np.asarray(stat["std"]) + 1e-6)
    elif mode == "min-max":
        return (x - stat["min"]) / (np.asarray(stat["max"]) - stat["min"] + 1e-6) * 2 - 1
    else:
        raise ValueError(mode)


# ============================================================
# Model loader
# ============================================================

def load_models(ckpt_dir, proprio_dim=2, action_dim=2, hidden_dim=256, pretrained_vision=False):
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device

    # IMPORTANT: 你训练时 save_state 里注册了这些对象（按你脚本）：
    # q, v, qh, vh, qh_tgt, q_opt, v_opt, qh_opt, vh_opt
    # 所以这里也要按同样顺序 prepare，否则 load_state 会对不上文件编号。
    q = DoubleQNet(proprio_dim, action_dim, hidden_dim, pretrained_vision)
    v = ValueNet(proprio_dim, hidden_dim, pretrained_vision)
    qh = DoubleQNet(proprio_dim, action_dim, hidden_dim, pretrained_vision)
    vh = ValueNet(proprio_dim, hidden_dim, pretrained_vision)
    qh_tgt = DoubleQNet(proprio_dim, action_dim, hidden_dim, pretrained_vision)

    # optim 仅用于让 load_state 对齐（推理不会 step）
    q_opt = torch.optim.AdamW(q.parameters(), lr=1e-4)
    v_opt = torch.optim.AdamW(v.parameters(), lr=1e-4)
    qh_opt = torch.optim.AdamW(qh.parameters(), lr=1e-4)
    vh_opt = torch.optim.AdamW(vh.parameters(), lr=1e-4)

    q, v, qh, vh, qh_tgt, q_opt, v_opt, qh_opt, vh_opt = accelerator.prepare(
        q, v, qh, vh, qh_tgt, q_opt, v_opt, qh_opt, vh_opt
    )

    accelerator.load_state(ckpt_dir)

    qh_eval = accelerator.unwrap_model(qh).eval().to(device)
    vh_eval = accelerator.unwrap_model(vh).eval().to(device)
    
    return accelerator, qh_eval, vh_eval



def plot_qh_vh_curves(qh_arr, vh_arr, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    qh_arr = np.asarray(qh_arr)
    vh_arr = np.asarray(vh_arr)

    # 1. qh_mean over time
    plt.figure(figsize=(6, 4))
    plt.plot(qh_arr, lw=2)
    plt.xlabel("timestep")
    plt.ylabel("qh_mean")
    plt.title("qh_mean over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "qh_mean_curve.png", dpi=200)
    plt.close()

    # 2. vh over time
    plt.figure(figsize=(6, 4))
    plt.plot(vh_arr, lw=2, color="tab:orange")
    plt.xlabel("timestep")
    plt.ylabel("vh")
    plt.title("vh over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "vh_curve.png", dpi=200)
    plt.close()

    # 3. qh_mean vs vh scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(qh_arr, vh_arr, s=8, alpha=0.6)
    plt.xlabel("qh_mean")
    plt.ylabel("vh")
    plt.title("vh vs qh_mean")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "qh_vs_vh_scatter.png", dpi=200)
    plt.close()




def overlay_text(
    img_bgr: np.ndarray,
    lines,
    origin=(10, 30),
    line_gap=28,
    font_scale=0.7,
    color=(0, 0, 255),
    thickness=2,
):
    """
    Draw multiple lines of text on an image (BGR).
    """
    img = img_bgr.copy()
    x, y0 = origin
    for i, text in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return img


def write_qh_vh_to_hdf5(hdf5_file, qh, vh, qh_key='qh_values', vh_key='vh_values'):
    """
    将qh和vh值写入hdf5文件
    
    Args:
        hdf5_file: hdf5文件路径
        qh_mean: qh_mean数组
        vh: vh数组
        qh_key: qh_mean在hdf5中的key名称
        vh_key: vh在hdf5中的key名称
    """
    try:
        with h5py.File(hdf5_file, 'a') as f:  # 'a'模式：如果文件存在则追加，不存在则创建
            # 如果key已存在，先删除
            if qh_key in f:
                del f[qh_key]
            if vh_key in f:
                del f[vh_key]
            
            # 写入新数据
            f.create_dataset(qh_key, data=qh, compression='gzip')
            f.create_dataset(vh_key, data=vh, compression='gzip')
            
            # 添加元数据
            # f.attrs[f'{qh_key}_shape'] = qh.shape
            # f.attrs[f'{vh_key}_shape'] = vh.shape
            # f.attrs[f'{qh_key}_mean'] = float(qh.mean())
            # f.attrs[f'{vh_key}_mean'] = float(vh.mean())
            
        return True
        
    except Exception as e:
        print(f"Error writing to {hdf5_file}: {e}")
        return False


# ============================================================
# Episode processing
# ============================================================

@torch.no_grad()
def process_one_episode(
    ep_path,
    data_root,
    out_root,
    exp_name,
    ckpt_name,
    task_name,
    metas,
    accelerator,
    qh,
    vh,
    fps,
    qh_key,
    vh_key,
):
    rel_ep = Path(ep_path).relative_to(data_root).with_suffix("")

    out_dir = (
        out_root
        / exp_name
        / ckpt_name
        / task_name
        / rel_ep
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = find_meta_for_episode(metas, ep_path)

    h = h5py.File(io.BytesIO(fileio.get(ep_path)), "r")
    T = h[meta["action_key"]].shape[0]

    writer = imageio.get_writer(out_dir / "cam_view0_with_values.mp4", fps=fps)
    qh_arr, vh_arr = [], []

    image_tf = build_image_transform(False)

    for t in tqdm(range(T)):
        img_t, img_bgr = decode_image_bytes_to_tensor(
            h[meta["observation_key"][0]][t],
            image_tf,
            accelerator.device,
            False,
        )
        pro_unnormalize = h[meta["proprio_key"]][max(t-1, 0)]
        pro = normalize(
            h[meta["proprio_key"]][max(t - 1, 0)],
            meta["proprio_stactics"],
            "mean-std",
        )
        act = normalize(
            h[meta["action_key"]][t],
            meta["action_statics"],
            "mean-std",
        )

        pro_t = torch.from_numpy(pro).float().unsqueeze(0).to(accelerator.device)
        act_t = torch.from_numpy(act).float().unsqueeze(0).to(accelerator.device)

        q1, q2 = qh(img_t, pro_t, act_t)
        qh_mean = 0.5 * (q1 + q2)
        v = vh(img_t, pro_t)

        qh_arr.append(qh_mean.item())
        vh_arr.append(v.item())


        lines = [
            f"t = {t:04d}",
            f"qh_mean = {qh_mean.detach().cpu().numpy().item():.4f}",
            f"vh      = {v.detach().cpu().numpy().item():.4f}",
            f"qh - vh = {qh_mean.detach().cpu().numpy().item() - v.detach().cpu().numpy().item():+.4f}",
            f"pro = [{pro_unnormalize[0]:+.3f}, {pro_unnormalize[1]:+.3f}]",
        ]
        frame = overlay_text(img_bgr, lines)
        writer.append_data(frame)

    writer.close()
    h.close()

    np.save(out_dir / "qh_mean.npy", qh_arr)
    np.save(out_dir / "vh.npy", vh_arr)
    plot_qh_vh_curves(qh_arr, vh_arr, out_dir)   # ⭐ 新增

    write_qh_vh_to_hdf5(ep_path, qh_arr, vh_arr, qh_key=qh_key, vh_key=vh_key)

    print(f"[OK] {out_dir}")


# ============================================================
# Main
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser('Visualize Feasible Value', add_help=False)
    
    # Data paths
    parser.add_argument('--data_roots', nargs='+', 
                        default=['/home/dodo/ljx/vln_data/run_in_the_circle_self_generate',
                                '/home/dodo/ljx/vln_data/run_circle_in_air',
                                '/home/dodo/ljx/vln_data/run_circle_in_air_suboptimal',
                                '/home/dodo/ljx/vln_data/run_in_the_circle_self_generate_1229'],
                        help='List of data root directories to process')
    
    parser.add_argument('--metas_path', 
                        default='/home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_1229_450',
                        help='Path to meta files directory')
    
    parser.add_argument('--ckpt_dir', 
                        default='/home/dodo/ljx/AIR3L/exp/20251231/iql_expectile09_vln_mixed_1229_450/ckpt-60000',
                        help='Path to checkpoint directory')
    
    parser.add_argument('--exp_name', 
                        default='iql_expectile09_vln_mixed_1231_450_25offset',
                        help='Experiment name for output organization')
    
    # Output settings
    parser.add_argument('--out_root', 
                        default='viz_path_all',
                        help='Root directory for output files')
    
    # Model parameters
    parser.add_argument('--proprio_dim', default=2, type=int,
                        help='Proprioception dimension')
    parser.add_argument('--action_dim', default=2, type=int,
                        help='Action dimension')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Hidden dimension for models')
    parser.add_argument('--pretrained_vision', action='store_true',
                        help='Use pretrained vision encoder')
    
    # Video settings
    parser.add_argument('--fps', default=10, type=int,
                        help='Frames per second for output videos')
    
    # HDF5 keys for storing QH/VH values
    parser.add_argument('--qh_key', 
                        default='test/qh_values',
                        help='Key name for storing QH mean values in HDF5 files')
    
    parser.add_argument('--vh_key', 
                        default='test/vh_values',
                        help='Key name for storing VH values in HDF5 files')
    
    return parser


def main():
    parser = argparse.ArgumentParser('Visualize Feasible Value', parents=[get_args_parser()])
    args = parser.parse_args()
    
    print("Visualize Feasible Value Script")
    print("=" * 50)
    print(f"Data roots: {args.data_roots}")
    print(f"Metas path: {args.metas_path}")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Output root: {args.out_root}")
    print("=" * 50)
    
    out_root = Path(args.out_root)
    ckpt_name = Path(args.ckpt_dir).name
    
    # Load models once
    print("Loading models...")
    metas = load_all_metas(args.metas_path)
    accelerator, qh, vh = load_models(
        args.ckpt_dir, 
        args.proprio_dim, 
        args.action_dim, 
        args.hidden_dim, 
        args.pretrained_vision
    )
    print("Models loaded successfully!")
    
    # Process each data root
    for data_root in args.data_roots:
        if not os.path.exists(data_root):
            print(f"Warning: Data root {data_root} does not exist, skipping...")
            continue
            
        task_name = Path(data_root).name
        print(f"\nProcessing task: {task_name}")
        print(f"Data root: {data_root}")
        
        # Find all episodes
        episodes = sorted(Path(data_root).rglob("*.hdf5"))
        
        # if args.max_episodes:
            # episodes = episodes[:args.max_episodes]
            # print(f"Limited to {args.max_episodes} episodes")
        
        print(f"Found {len(episodes)} episodes to process")
        
        # Process each episode
        for i, ep in enumerate(episodes):
            print(f"Processing episode {i+1}/{len(episodes)}: {ep.name}")
            try:
                process_one_episode(
                    str(ep),
                    data_root,
                    out_root,
                    args.exp_name,
                    ckpt_name,
                    task_name,
                    metas,
                    accelerator,
                    qh,
                    vh,
                    fps=args.fps,
                    qh_key=args.qh_key,
                    vh_key=args.vh_key,
                )
            except Exception as e:
                print(f"Error processing {ep}: {e}")
                continue
        
        print(f"Completed task: {task_name}")
    
    print("\n" + "=" * 50)
    print("All tasks completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()