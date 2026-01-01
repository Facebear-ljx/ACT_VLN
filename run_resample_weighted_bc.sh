
port=12363
source ~/miniconda3/bin/deactivate
source ~/miniconda3/bin/activate vln_model
export WANDB_BASE_URL=https://api.bandw.top
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train_weighted_bc.py \
        --model ACT_VLNAgent_v \
        --epochs 10 \
        --batch-size 32 \
        --precision no \
        --learning_rate 1e-5 \
        --weight_decay 1e-4 \
        --action_norm mean-std \
        --output_dir /home/dodo/ljx/AIR3L/exp/20260101/test_weightedbc \
        --metas_path /home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_1229_450 \
        --qh_key /home/dodo/ljx/AIR3L/viz_path_all/iql_expectile09_vln_mixed_1229_450/ckpt-240000/qh_mean_values \
        --vh_key /home/dodo/ljx/AIR3L/viz_path_all/iql_expectile09_vln_mixed_1229_450/ckpt-240000/vh_values \
        --resume /home/dodo/ljx/AIR3L/exp/20251227_bc/vln_200succ_v_map_resample/ckpt-160000 \
        --save_interval 5000 \
        --port $port \
        --seed 12314