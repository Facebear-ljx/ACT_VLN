# srun -p mozi-S1 -N1 \
python HDF5Scanner.py \
    --top_path /home/dodo/ljx/vln_data \
    --dataset_name vln_succ \
    --action_key base_action \
    --proprio_key base_action \
    --observation_key 'observations/images/cam_high' \
    --save_path /home/dodo/ljx/AIR3L/meta_files/1ViewData/vln_mixed_1229 \