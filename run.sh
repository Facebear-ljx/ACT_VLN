port=12358
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc-per-node=4 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
        --model ACTAgent \
        --epochs 100 \
        --batch-size 64 \
        --precision fp32 \
        --learning_rate 1e-5 \
        --output_dir /home/ljx/ljx/BearRL/exp/20250522/Agilex \
        --metas_path /home/ljx/ljx/BearRL/meta_files/3ViewData/Agilex \
        --save_interval 10000 \
        --port $port