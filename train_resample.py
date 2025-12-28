import argparse
import datetime
import os
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import subprocess
from accelerate import Accelerator

import model.model_factory
from config.hero_info import DOMAIN_NAME_TO_INFO
from dataset.Dataset_resample import create_dataloader
from timm import create_model
from utils.count_parameter import count_parameters


def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    
    # Base Settings
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--action_norm', default='min-max', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--metas_path', default='', type=str)
    parser.add_argument('--precision', default='bf16', type=str)
    
    
    parser.add_argument('--model', default='ACTAgent', type=str)
    parser.add_argument('--seed', default=1000, type=int)
    
    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=20000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    
    
    parser.add_argument('--output_dir', default='runnings/',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--resume', default=None, help='model resume from checkpoint')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29529, type=int, help='port')

    return parser

def main(args):
    
    output_dir = Path(args.output_dir)
    from accelerate import DistributedDataParallelKwargs
    accelerator = Accelerator(mixed_precision = args.precision,
                              log_with="wandb", 
                              project_dir=output_dir)
                            #   kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    accelerator.init_trackers("ACT_Training", config=vars(args))
    torch.distributed.barrier()
    
    
    model = create_model(args.model)

    print("ac_num:", model.ac_num)
    train_dataloader = create_dataloader(
        batch_size = args.batch_size,
        metas_path = args.metas_path,
        rank = args.rank,
        world_size = args.world_size,
        DOMAIN_NAME_TO_INFO=DOMAIN_NAME_TO_INFO,
        action_normalization=args.action_norm,
        action_sequence_length=model.ac_num,
        max_freq=model.ac_num,
        im_padding=True
    )
    
    model = model.to(torch.float32)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas = (0.9, 0.999))
    model, optim = accelerator.prepare(model, optim)
    print(count_parameters(model, unit='M', fmt='.4f'))
    
    iters = 0
    if args.resume is not None:
        accelerator.print('>>>>>> resume from {}'.format(args.resume))
        accelerator.load_state(args.resume)
        
    accelerator.print(f"Start training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        ep_start_time = time.time()
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        past_time = time.time()
        for data in train_dataloader:
            data_time = time.time() - past_time
            torch.distributed.barrier()
            
            #### training
            inputs = {
                'image_obs': data['image_input'].cuda(non_blocking=True),
                'action': data['action'].cuda(non_blocking=True),
                'qpos': data['proprio'].cuda(non_blocking=True),
                'is_pad': ~data['action_mask'].cuda(),
            }
            # print('imag:',  data['image_input'].shape)
            # print('action:',  data['action'].shape)
            # print('qpos:', data['proprio'].shape)
            # print('is_pad:', data['action_mask'].shape) 
            optim.zero_grad()
            loss = model(**inputs)
            accelerator.backward(loss['policy_loss'])
            optim.step()
            
            #### log
            time_per_iter = time.time() - past_time
            
            if iters % args.log_interval == 0: 
                for key in loss:
                    accelerator.log({key: loss[key]}, step=iters)
                loss_value = loss['policy_loss']
                accelerator.print(f"[Epoch] {epoch}/{args.epochs} [Iter {iters}] [Training Loss] {loss_value} [time_per_iter] {time_per_iter} [data_time] {data_time} ")
            
            if iters % args.save_interval == 0 and iters != 0:
                accelerator.print("========start saving models=========")
                accelerator.save_state(os.path.join(output_dir, f"ckpt-{iters}"))
            
            torch.distributed.barrier()
            iters += 1
            
            past_time = time.time()
            
        total_time = time.time() - ep_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    accelerator.save_state(os.path.join(output_dir, f"ckpt-{iters}"))

def slurm_env_init(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)
    args.gpu = args.rank
    torch.cuda.set_device(args.gpu)
    
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    
    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    print('ddp end init')
    torch.distributed.barrier()
    
    # fix the seed for reproducibility
    seed = args.seed + args.rank # torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("ddp setup done")
    return args


if __name__ == '__main__':
     
    parser = argparse.ArgumentParser('training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(slurm_env_init(args))