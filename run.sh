port=12359
source /opt/miniconda3/bin/deactivate
source /opt/miniconda3/bin/activate air3l
export CUDA_VISIBLE_DEVICES=0
/home/ljx/.conda/envs/air3l/bin/torchrun --nproc-per-node=1 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
        --model ACTAgent \
        --epochs 3000 \
        --batch-size 128 \
        --precision no \
        --learning_rate 1e-5 \
        --output_dir /home/ljx/ljx/BearRL/exp/20250523/test \
        --metas_path /home/ljx/ljx/BearRL/meta_files/3ViewData/Agilex \
        --save_interval 10000 \
        --port $port