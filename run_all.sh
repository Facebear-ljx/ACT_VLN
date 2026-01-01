#!/usr/bin/env bash
port=12363

# ===== conda env =====
source ~/miniconda3/bin/deactivate
source ~/miniconda3/bin/activate vln_model

# ===== wandb mirror =====
export WANDB_BASE_URL=https://api.bandw.top


BATCH_SIZE=32
VALUE_EPOCH=6
POLICY_EPOCH=6
expectile_vc=0.9
save_interval=2500

# data info
metas_path=/home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_0101
data_roots=(
  /home/dodo/ljx/vln_data/run_circle_in_air
  /home/dodo/ljx/vln_data/run_circle_in_air_suboptimal
  /home/dodo/ljx/vln_data/run_in_the_circle_self_generate_1229
  /home/dodo/ljx/vln_data/run_in_the_circle_self_generate
  /home/dodo/ljx/vln_data/run_circle_in_air_recover
)

# ckpt save path
# data/exp_name
exp_name=20260101_mixed_450   # used for ckpt name and video dir name
output_dir=/home/dodo/ljx/AIR3L/exp/20260101/$exp_name

# resume info
vh_resume=/home/dodo/ljx/AIR3L/exp/20260101/20260101_mixed_1229_450/value/ckpt-5000
policy_resume=/home/dodo/ljx/AIR3L/exp/20260101/20260101_mixed_1229_450/policy/ckpt-10000

# infer info
ckpt=ckpt-5000
seed=12134

# ===== launch Value training=====
echo "Training Feasible value function"
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train_iql_value.py \
  --epochs $VALUE_EPOCH \
  --batch_size $BATCH_SIZE \
  --precision no \
  --lr_q 3e-4 \
  --lr_v 3e-4 \
  --weight_decay 1e-4 \
  --gamma 0.99 \
  --expectile_vc $expectile_vc \
  --ema_tau 0.995 \
  --action_norm mean-std \
  --max_freq 1 \
  --save_interval $save_interval \
  --log_interval 10 \
  --output_dir $output_dir/value \
  --metas_path $metas_path \
  --port $port \
  --seed $seed \
  --resume $vh_resume \

# # ======== launch value infer =====
echo "Infer feasible values for all data"
echo "data_roots=[$data_roots]"
vh_key=$output_dir/$ckpt/vh_key
qh_key=$output_dir/$ckpt/qh_key
# vh_key=$output_dir/vh_key
# qh_key=$output_dir/qh_key
python vis_feasible_value.py \
    --data_roots ${data_roots[@]} \
    --metas_path $metas_path \
    --ckpt_dir $output_dir/value/$ckpt \
    --exp_name $exp_name \
    --out_root viz_path_all \
    --qh_key $qh_key \
    --vh_key $vh_key \
    --fps 10 \


# ======= launch policy training ====
echo "train weighted bc for all data"
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train_weighted_bc.py \
        --model ACT_VLNAgent_v \
        --epochs $POLICY_EPOCH \
        --batch-size $BATCH_SIZE \
        --precision no \
        --learning_rate 1e-5 \
        --weight_decay 1e-4 \
        --action_norm mean-std \
        --output_dir $output_dir/policy \
        --metas_path $metas_path \
        --qh_key $qh_key \
        --vh_key $vh_key \
        --save_interval $save_interval \
        --port $port \
        --seed $seed \
        --resume $policy_resume \