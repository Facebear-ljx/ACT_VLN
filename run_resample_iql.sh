#!/usr/bin/env bash

port=12363

# ===== conda env =====
source ~/miniconda3/bin/deactivate
source ~/miniconda3/bin/activate vln_model

# ===== wandb mirror =====
export WANDB_BASE_URL=https://api.bandw.top

# ===== launch =====
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train_iql_value.py \
  --epochs 3000 \
  --batch_size 32 \
  --precision no \
  --lr_q 3e-4 \
  --lr_v 3e-4 \
  --weight_decay 1e-4 \
  --gamma 0.99 \
  --expectile_v 0.7 \
  --expectile_vc 0.9 \
  --ema_tau 0.995 \
  --action_norm mean-std \
  --max_freq 1 \
  --save_interval 10000 \
  --log_interval 10 \
  --output_dir /home/dodo/ljx/AIR3L/exp/20251228/iql_vln_feasible \
  --metas_path /home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_1229 \
  --port $port \
  --seed 12314
