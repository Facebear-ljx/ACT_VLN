# srun -p mozi-S1 -N1 \
python HDF5Scanner.py \
    --top_path /mnt/ssd0/data/agilex/processed \
    --dataset_name Agilex \
    --action_key action \
    --language_instruction_key language_instruction \
    --proprio_key proprio \
    --observation_key 'observation/top_image' 'observation/left_wrist_image' 'observation/right_wrist_image' \
    --save_path /home/ljx/ljx/BearRL/meta_files/3ViewData/Agilex \