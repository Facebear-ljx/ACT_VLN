
port=12363
source ~/miniconda3/bin/deactivate
source ~/miniconda3/bin/activate vln_model
export WANDB_BASE_URL=https://api.bandw.top
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train_resample.py \
        --model ACT_VLNAgent_v \
        --epochs 3000 \
        --batch-size 32 \
        --precision no \
        --learning_rate 1e-5 \
        --weight_decay 1e-4 \
        --action_norm mean-std \
        --output_dir /home/dodo/ljx/AIR3L/exp/20251227/vln_200succ_v_map_resample \
        --metas_path /home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_succ \
        --save_interval 10000 \
        --port $port \
        --seed 12314