#!/bin/bash

# 激活环境
source ~/miniconda3/bin/deactivate
source ~/miniconda3/bin/activate vln_model

# 基本运行 - 使用默认参数
# echo "Running with default parameters..."
# python vis_feasible_value.py

# 自定义参数运行
echo "Running with custom parameters..."
python vis_feasible_value.py \
    --data_roots /home/dodo/ljx/vln_data/run_circle_in_air \
                 /home/dodo/ljx/vln_data/run_circle_in_air_suboptimal \
                 /home/dodo/ljx/vln_data/run_in_the_circle_self_generate \
                 /home/dodo/ljx/vln_data/run_in_the_circle_self_generate_1229 \
    --metas_path /home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_1229_450 \
    --ckpt_dir /home/dodo/ljx/AIR3L/exp/20251231/iql_expectile09_vln_mixed_1229_450/ckpt-60000 \
    --exp_name test \
    --out_root viz_path_all \
    --fps 10 \

# 只处理特定任务
# echo "Processing only run_circle_in_air task..."
# python vis_feasible_value.py \
#     --single_task run_circle_in_air \
#     --max_episodes 3 \
#     --fps 20

# # 使用不同的模型参数
# echo "Running with different model parameters..."
# python vis_feasible_value.py \
#     --proprio_dim 4 \
#     --action_dim 4 \
#     --hidden_dim 512 \
#     --pretrained_vision \
#     --single_task run_in_the_circle \
#     --max_episodes 2