#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IQL (Implicit Q-Learning) training script for Value V(s) + Double Q(s,a)
- Vision encoder: ResNet18 (single-frame, single-view RGB)
- Projector: 2-layer MLP
- Inputs: image + proprio (+ action for Q)
- Dataset expected keys (per batch):
    image_input:      (B, V, C, H, W) or (B, C, H, W)
    proprio:          (B, Dp)
    action:           (B, T, Da) or (B, Da)   (we will use the first step if (B,T,Da))
    reward:           (B,) or (B,1)  (chunk-reward already aggregated)
    next_done:        (B,)  (1 means terminal after executing this chunk)
    next_image_input: optional, same shape as image_input
    next_proprio:     optional, same shape as proprio
- If next_* are missing, we fallback to current state as next (script still runs, but targets degrade).
"""

import argparse
import datetime
import os
import random
import time
from pathlib import Path

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from accelerate import Accelerator

from timm import create_model  # not used, but kept to match your environment
from utils.count_parameter import count_parameters

from config.hero_info import DOMAIN_NAME_TO_INFO
from dataset.Dataset_resample import create_dataloader

from model.critic.value import DoubleQNet, ValueNet
# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def ema_update_(target: nn.Module, source: nn.Module, tau: float):
    # target = tau * target + (1-tau) * source
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(tau).add_(sp.data, alpha=(1.0 - tau))


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """
    IQL expectile regression:
      minimize w * diff^2,
      w = expectile if diff > 0 else (1 - expectile)
    where diff = (q - v)
    """
    w = torch.where(diff > 0, torch.tensor(expectile, device=diff.device, dtype=diff.dtype),
                    torch.tensor(1.0 - expectile, device=diff.device, dtype=diff.dtype))
    return (w * diff.pow(2)).mean()


def safe_expectile_loss(diff: torch.Tensor, expectile: float):
    """
    diff = qc - vc
    """
    weight = torch.where(
        diff < 0,
        torch.tensor(expectile, device=diff.device),
        torch.tensor(1.0 - expectile, device=diff.device),
    )
    return (weight * diff.pow(2)).mean()


def to_state_tensors(batch, device):
    """
    Normalize different batch shapes:
    - image_input could be (B,V,C,H,W) or (B,C,H,W)
      we take view 0 if V exists.
    - action could be (B,T,Da) or (B,Da)
      we take the first action if T exists (chunk action sequence -> action at t=0).
    """
    img = batch["image_input"]
    if img.dim() == 5:
        # (B,V,C,H,W) -> view0
        img = img[:, 0]
    # ensure float
    img = img.to(device=device, non_blocking=True).float()

    proprio = batch["proprio"].to(device=device, non_blocking=True).float()

    act = batch["action"]
    if act.dim() == 3:
        # (B,T,Da) -> use first step (or you can pool/flatten if you prefer)
        act = act[:, 0]
    act = act.to(device=device, non_blocking=True).float()

    # reward: allow (B,) or (B,1) or (B,T)
    rew = batch["reward"]
    if isinstance(rew, np.ndarray):
        rew = torch.from_numpy(rew)
    rew = rew.to(device=device, non_blocking=True).float()
    if rew.dim() > 1:
        rew = rew.view(rew.shape[0], -1).mean(dim=1)  # robust fallback
        
    # cost: allow (B,) or (B,1) or (B,T)
    cos = batch["feasibility"]
    if isinstance(cos, np.ndarray):
        cos = torch.from_numpy(cos)
    cos = cos.to(device=device, non_blocking=True).float()
    if cos.dim() > 1:
        cos = cos.view(cos.shape[0], -1).mean(dim=1)  # robust fallback    

    done = batch["next_done"]
    if isinstance(done, np.ndarray):
        done = torch.from_numpy(done)
    done = done.to(device=device, non_blocking=True).float()
    if done.dim() > 1:
        done = done.view(done.shape[0], -1)[:, 0]

    # next state (optional)
    if "next_image_input" in batch and "next_proprio" in batch:
        nimg = batch["next_image_input"]
        if nimg.dim() == 5:
            nimg = nimg[:, 0]
        nimg = nimg.to(device=device, non_blocking=True).float()
        npro = batch["next_proprio"].to(device=device, non_blocking=True).float()
    else:
        # fallback: use current state as next (still runnable)
        nimg, npro = img, proprio

    return img, proprio, act, rew, cos, done, nimg, npro


# -------------------------
# Args / DDP init (slurm style, similar to your script)
# -------------------------
def get_args_parser():
    p = argparse.ArgumentParser("IQL V/Q training", add_help=False)

    # data / io
    p.add_argument("--metas_path", default="", type=str)
    p.add_argument("--output_dir", default="runnings_iql/", type=str)
    p.add_argument("--resume", default=None, type=str)

    # optimization
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--lr_q", default=3e-4, type=float)
    p.add_argument("--lr_v", default=3e-4, type=float)
    p.add_argument("--weight_decay", default=0.01, type=float)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--precision", default="bf16", type=str)

    # iql
    p.add_argument("--gamma", default=0.99, type=float)
    p.add_argument("--expectile_v", default=0.7, type=float)
    p.add_argument("--expectile_vc", default=0.9, type=float)
    p.add_argument("--ema_tau", default=0.995, type=float)  # target smoothing

    # model dims
    p.add_argument("--proprio_dim", default=2, type=int)   # your base_vel is 2
    p.add_argument("--action_dim", default=2, type=int)    # your cmd_vel is 2
    p.add_argument("--hidden_dim", default=256, type=int)
    p.add_argument("--pretrained_vision", action="store_true")

    # logging/checkpoint
    p.add_argument("--log_interval", default=10, type=int)
    p.add_argument("--save_interval", default=10000, type=int)

    # distributed (slurm)
    p.add_argument("--seed", default=1000, type=int)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument("--port", default=29529, type=int)

    # dataset options (mirror your usage)
    p.add_argument("--action_norm", default="min-max", type=str)
    p.add_argument("--max_freq", default=10, type=int)  # should match your chunk length
    p.add_argument("--im_padding", action="store_true")

    return p


def slurm_env_init(args):
    # If running single GPU locally, these envs may not exist.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
        args.gpu = args.rank
        torch.cuda.set_device(args.gpu)

        torch.distributed.init_process_group(
            backend="nccl", rank=args.rank, world_size=args.world_size
        )
        torch.distributed.barrier()

        seed = args.seed + args.rank
        set_seed(seed)
        cudnn.benchmark = True
    else:
        args.rank = 0
        args.world_size = 1
        set_seed(args.seed)
        cudnn.benchmark = True

    return args


# -------------------------
# Main
# -------------------------
def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.precision,
        log_with="wandb",
        project_dir=output_dir
    )
    accelerator.init_trackers("VLN_VQ_Training", config=vars(args))

    device = accelerator.device

    # Build models
    q = DoubleQNet(
        proprio_dim=args.proprio_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        pretrained_vision=args.pretrained_vision
    )
    v = ValueNet(
        proprio_dim=args.proprio_dim,
        hidden_dim=args.hidden_dim,
        pretrained_vision=args.pretrained_vision
    )
    
    qh = DoubleQNet(
        proprio_dim=args.proprio_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        pretrained_vision=args.pretrained_vision
    )
    vh = ValueNet(
        proprio_dim=args.proprio_dim,
        hidden_dim=args.hidden_dim,
        pretrained_vision=args.pretrained_vision
    )
    
    # Targets
    q_tgt = DoubleQNet(
        proprio_dim=args.proprio_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        pretrained_vision=args.pretrained_vision
    )
    
    q_tgt = copy.deepcopy(q)
    qh_tgt = copy.deepcopy(qh)
    # q_tgt.load_state_dict(q.state_dict())
    # qh_tgt.load_state_dict(qh.state_dict())
    q_tgt.eval()
    qh_tgt.eval()
    for m in (q_tgt, qh_tgt):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # Dataloader
    # NOTE: you should ensure your dataset outputs reward, next_done, and (ideally) next_image_input/next_proprio.
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.metas_path,
        rank=args.rank,
        world_size=args.world_size,
        DOMAIN_NAME_TO_INFO=DOMAIN_NAME_TO_INFO,
        action_normalization=args.action_norm,
        action_sequence_length=args.max_freq,
        max_freq=args.max_freq,
        im_padding=args.im_padding
    )

    # Optims
    q_opt = torch.optim.AdamW(q.parameters(), lr=args.lr_q, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    v_opt = torch.optim.AdamW(v.parameters(), lr=args.lr_v, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    qh_opt = torch.optim.AdamW(qh.parameters(), lr=args.lr_q, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    vh_opt = torch.optim.AdamW(vh.parameters(), lr=args.lr_v, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Prepare with accelerate
    q, v, qh, vh, qh_tgt, q_opt, v_opt, qh_opt, vh_opt, train_dataloader = accelerator.prepare(q, v, qh, vh, qh_tgt, q_opt, v_opt, qh_opt, vh_opt, train_dataloader)
    # q_tgt, qh_tgt = accelerator.prepare(q_tgt, qh_tgt)
    
    accelerator.print("Q params:", count_parameters(q, unit="M", fmt=".4f"))
    accelerator.print("V params:", count_parameters(v, unit="M", fmt=".4f"))
    accelerator.print("Qh params:", count_parameters(qh, unit="M", fmt=".4f"))
    accelerator.print("Vh params:", count_parameters(vh, unit="M", fmt=".4f"))    

    iters = 0
    if args.resume is not None:
        accelerator.print(f">>> resume from {args.resume}")
        accelerator.load_state(args.resume)

    accelerator.print(f"Start IQL training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        ep_start = time.time()
        q.train()
        v.train()
        qh.train()
        vh.train()

        # distributed sampler epoch set (if exists)
        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        past_time = time.time()

        for batch in train_dataloader:
            data_time = time.time() - past_time
            
            # --------
            # Unpack
            # --------
            img, pro, act, rew, cos, done, nimg, npro = to_state_tensors(batch, device=device)

            # Feasible value update
            with torch.no_grad():
                target_qh = (1 - args.gamma) * cos + (args.gamma) * torch.max(vh(nimg, npro), cos)
                target_qh = (1 - done) * target_qh + cos * done

            qh1, qh2 = qh(img, pro, act)
            qh_loss = F.mse_loss(qh1, target_qh) + F.mse_loss(qh2, target_qh)

            qh_opt.zero_grad(set_to_none=True)
            accelerator.backward(qh_loss)
            qh_opt.step()

            # --------
            # Feasible Update V (expectile regression to Q_min(s,a) target)
            # diff = q_min_tgt(s,a) - v(s)
            # --------
            with torch.no_grad():
                qh_tgt1, qh_tgt2 = qh(img, pro, act)  # very strange here
                qh_max = (qh_tgt1 + qh_tgt2) / 2
                assert qh_max.shape == (img.shape[0],), qh_max.shape

            vh_pred = vh(img, pro)
            diffh = qh_max - vh_pred

            neg_ratio = (diffh < 0).float().mean()
            pos_ratio = (diffh >= 0).float().mean()

            neg_diff_mean = diffh[diffh < 0].mean() if (diffh < 0).any() else torch.tensor(0.0, device=diffh.device)
            pos_diff_mean = diffh[diffh >= 0].mean() if (diffh >= 0).any() else torch.tensor(0.0, device=diffh.device)
            vh_loss = safe_expectile_loss(diffh, args.expectile_vc)

            vh_opt.zero_grad(set_to_none=True)
            accelerator.backward(vh_loss)
            vh_opt.step()
            
            vh_terminal = vh_pred[done > 0.5].mean().detach().float() if (done > 0.5).any() else torch.tensor(0.0, device=vh_pred.device)

            # --------
            # EMA targets
            # --------
            # Need to update the *unwrapped* params. With accelerate, q_tgt/v_tgt are not prepared;
            # we update using accelerator.unwrap_model(q/v) and then copy EMA into q_tgt/v_tgt.
            with torch.no_grad():
                # q_src = accelerator.unwrap_model(q)
                # ema_update_(q_tgt, q_src, args.ema_tau)
                qh_src = accelerator.unwrap_model(qh)
                ema_update_(qh_tgt, qh_src, args.ema_tau)

            # --------
            # Logging / ckpt
            # --------
            time_per_iter = time.time() - past_time
            if iters % args.log_interval == 0:
                accelerator.log(
                    {
                        # # ===== reward IQL =====
                        # "loss/q_loss": q_loss.detach().float(),
                        # "loss/v_loss": v_loss.detach().float(),
                        # "stats/reward_mean": rew.mean().detach().float(),
                        # "stats/done_rate": done.mean().detach().float(),
                        # "stats/q1_mean": q1.mean().detach().float(),
                        # "stats/q2_mean": q2.mean().detach().float(),
                        # "stats/q_min_mean": torch.min(q1, q2).mean().detach().float(),
                        # "stats/v_mean": v_pred.mean().detach().float(),

                        # ===== feasible (cost) IQL =====
                        "loss/qh_loss": qh_loss.detach().float(),
                        "loss/vh_loss": vh_loss.detach().float(),
                        "stats/qh1_mean": qh1.mean().detach().float(),
                        "stats/qh2_mean": qh2.mean().detach().float(),
                        "stats/qh_tgt_model_mean": qh_max.mean().detach().float(),
                        "stats/tgt_qh_mean": target_qh.mean().detach().float(),
                        "stats/vh_mean": vh_pred.mean().detach().float(),
                        "stats/cost_mean": cos.mean().detach().float(),
                        "stats/vh_terminal": vh_terminal,
                        "stats/vh_neg_diff_mean": neg_diff_mean,
                        "stats/vh_pos_diff_mean": pos_diff_mean,
                        "stats/vh_neg_ratio": neg_ratio,
                        "stats/vh_pos_ratio": pos_ratio,

                        # ===== timing =====
                        "time/time_per_iter": time_per_iter,
                        "time/data_time": data_time,
                    },
                    step=iters
                )

                accelerator.print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Iter {iters}] "
                    # f"Q={q_loss.item():.4f} V={v_loss.item():.4f} | "
                    f"Qh_loss={qh_loss.item():.4f} Vh_loss={vh_loss.item():.4f} | "
                    f"rew={rew.mean().item():.3f} cost={cos.mean().item():.3f} | "
                    f"vh_mean={vh_pred.mean().detach().float():.3f} | "
                    f"done={done.mean().item():.6f} | "
                    f"time={time_per_iter:.3f}s data={data_time:.3f}s"
                )


            if iters % args.save_interval == 0 and iters != 0:
                accelerator.print("======== saving checkpoint ========")
                accelerator.save_state(os.path.join(output_dir, f"ckpt-{iters}"))

            iters += 1
            past_time = time.time()

        total_time = time.time() - ep_start
        accelerator.print("Epoch time:", str(datetime.timedelta(seconds=int(total_time))))

    accelerator.save_state(os.path.join(output_dir, f"ckpt-{iters}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("iql training", parents=[get_args_parser()])
    args = parser.parse_args()
    args = slurm_env_init(args)
    main(args)