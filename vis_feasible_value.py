#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
from pathlib import Path

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
    accelerator.load_state(ckpt_dir)
    return accelerator, accelerator.unwrap_model(qh).eval(), accelerator.unwrap_model(vh).eval()



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


    print(f"[OK] {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    for data_root in ['/home/dodo/ljx/vln_data/run_in_the_circle_self_generate', \
                    "/home/dodo/ljx/vln_data/run_circle_in_air", \
                    "/home/dodo/ljx/vln_data/run_circle_in_air_suboptimal", \
                    "/home/dodo/ljx/vln_data/run_in_the_circle_self_generate_1229"]:
        metas_path = "/home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_1229_450"
        ckpt_dir = "/home/dodo/ljx/AIR3L/exp/20251231/iql_expectile09_vln_mixed_1229_450/ckpt-60000"
        exp_name = "iql_expectile09_vln_mixed_1231_450_25offset"
        
        out_root = Path("viz_path_all")
        task_name = Path(data_root).name
        ckpt_name = Path(ckpt_dir).name

        metas = load_all_metas(metas_path)
        accelerator, qh, vh = load_models(ckpt_dir)

        for ep in sorted(Path(data_root).rglob("*.hdf5")):
            process_one_episode(
                str(ep),
                data_root,
                out_root,
                exp_name,
                ckpt_name,
                task_name,
                metas,
                accelerator,
                qh,
                vh,
                fps=10,
            )


if __name__ == "__main__":
    main()