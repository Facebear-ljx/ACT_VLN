#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将viz_path_all中的qh_mean.npy和vh.npy值写入到vln_data目录下对应的hdf5文件中
"""

import argparse
import os
import glob
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re


def find_matching_hdf5_files(vln_data_dir, task_name):
    """
    在vln_data目录中查找匹配的hdf5文件
    
    Args:
        vln_data_dir: vln_data目录路径
        task_name: 任务名称，如'run_circle_in_air'
    
    Returns:
        list: 匹配的hdf5文件路径列表
    """
    # 构建搜索模式
    search_patterns = [
        os.path.join(vln_data_dir, task_name, "*.hdf5"),
        os.path.join(vln_data_dir, task_name, "**", "*.hdf5"),
        os.path.join(vln_data_dir, "**", task_name, "*.hdf5"),
        os.path.join(vln_data_dir, "**", "*.hdf5")  # 最后搜索所有hdf5文件
    ]
    
    hdf5_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            # 检查文件名或路径是否包含任务名
            if task_name.lower() in file.lower():
                hdf5_files.append(file)
    
    # 去重
    hdf5_files = list(set(hdf5_files))
    return hdf5_files


def extract_episode_info_from_path(episode_path):
    """
    从episode路径中提取episode信息
    
    Args:
        episode_path: episode目录路径
    
    Returns:
        dict: 包含episode信息的字典
    """
    episode_name = os.path.basename(episode_path)
    
    # 提取episode编号
    episode_match = re.search(r'episode_(\d+)', episode_name)
    episode_num = int(episode_match.group(1)) if episode_match else None
    
    return {
        'episode_name': episode_name,
        'episode_num': episode_num,
        'episode_path': episode_path
    }


def match_episode_to_hdf5(episode_info, hdf5_files):
    """
    将episode信息匹配到对应的hdf5文件
    
    Args:
        episode_info: episode信息字典
        hdf5_files: hdf5文件列表
    
    Returns:
        str or None: 匹配的hdf5文件路径
    """
    episode_num = episode_info['episode_num']
    
    if episode_num is None:
        return None
    
    # 尝试多种匹配策略
    for hdf5_file in hdf5_files:
        hdf5_basename = os.path.basename(hdf5_file)
        
        # 策略1: 文件名包含相同的数字
        if str(episode_num) in hdf5_basename:
            return hdf5_file
        
        # 策略2: 提取hdf5文件名中的数字进行匹配
        hdf5_numbers = re.findall(r'\d+', hdf5_basename)
        if str(episode_num) in hdf5_numbers:
            return hdf5_file
    
    # 如果没有精确匹配，返回第一个文件（如果只有一个的话）
    if len(hdf5_files) == 1:
        return hdf5_files[0]
    
    return None


def write_qh_vh_to_hdf5(hdf5_file, qh_mean, vh, qh_key='qh_mean_values', vh_key='vh_values'):
    """
    将qh_mean和vh值写入hdf5文件
    
    Args:
        hdf5_file: hdf5文件路径
        qh_mean: qh_mean数组
        vh: vh数组
        qh_key: qh_mean在hdf5中的key名称
        vh_key: vh在hdf5中的key名称
    """
    try:
        with h5py.File(hdf5_file, 'a') as f:  # 'a'模式：如果文件存在则追加，不存在则创建
            # 如果key已存在，先删除
            if qh_key in f:
                del f[qh_key]
            if vh_key in f:
                del f[vh_key]
            
            # 写入新数据
            f.create_dataset(qh_key, data=qh_mean, compression='gzip')
            f.create_dataset(vh_key, data=vh, compression='gzip')
            
            # 添加元数据
            f.attrs[f'{qh_key}_shape'] = qh_mean.shape
            f.attrs[f'{vh_key}_shape'] = vh.shape
            f.attrs[f'{qh_key}_mean'] = float(qh_mean.mean())
            f.attrs[f'{vh_key}_mean'] = float(vh.mean())
            
        return True
        
    except Exception as e:
        print(f"Error writing to {hdf5_file}: {e}")
        return False


def process_task_directory(task_dir, vln_data_dir, qh_key='qh_mean_values', vh_key='vh_values'):
    """
    处理单个任务目录
    
    Args:
        task_dir: 任务目录路径
        vln_data_dir: vln_data目录路径
        qh_key: qh_mean在hdf5中的key名称
        vh_key: vh在hdf5中的key名称
    
    Returns:
        dict: 处理结果统计
    """
    task_name = os.path.basename(task_dir)
    print(f"\nProcessing task: {task_name}")
    
    # 查找匹配的hdf5文件
    hdf5_files = find_matching_hdf5_files(vln_data_dir, task_name)
    print(f"Found {len(hdf5_files)} matching HDF5 files for task {task_name}")
    
    if not hdf5_files:
        print(f"No HDF5 files found for task {task_name}")
        return {'processed': 0, 'failed': 0, 'skipped': 1}
    
    # 获取所有episode目录
    episode_dirs = glob.glob(os.path.join(task_dir, "episode_*"))
    episode_dirs.sort()
    
    print(f"Found {len(episode_dirs)} episodes")
    
    processed = 0
    failed = 0
    skipped = 0
    
    for episode_dir in tqdm(episode_dirs, desc=f"Processing {task_name}"):
        # 检查qh_mean.npy和vh.npy是否存在
        qh_mean_path = os.path.join(episode_dir, "qh_mean.npy")
        vh_path = os.path.join(episode_dir, "vh.npy")
        
        if not (os.path.exists(qh_mean_path) and os.path.exists(vh_path)):
            print(f"Missing files in {episode_dir}")
            skipped += 1
            continue
        
        try:
            # 加载qh_mean和vh数据
            qh_mean = np.load(qh_mean_path)
            vh = np.load(vh_path)
            
            # 提取episode信息
            episode_info = extract_episode_info_from_path(episode_dir)
            
            # 匹配到对应的hdf5文件
            matched_hdf5 = match_episode_to_hdf5(episode_info, hdf5_files)
            
            if matched_hdf5 is None:
                print(f"No matching HDF5 file for {episode_info['episode_name']}")
                skipped += 1
                continue
            
            # 写入hdf5文件
            success = write_qh_vh_to_hdf5(matched_hdf5, qh_mean, vh, qh_key, vh_key)
            
            if success:
                processed += 1
                print(f"✓ {episode_info['episode_name']} -> {os.path.basename(matched_hdf5)}")
            else:
                failed += 1
                
        except Exception as e:
            print(f"Error processing {episode_dir}: {e}")
            failed += 1
    
    return {'processed': processed, 'failed': failed, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser(description='Write QH/VH values to HDF5 files')
    
    parser.add_argument('--viz_path_all', 
                        default='/home/dodo/ljx/AIR3L/viz_path_all/iql_expectile09_vln_mixed_1229_450/ckpt-240000',
                        help='Path to viz_path_all checkpoint directory')
    parser.add_argument('--vln_data_dir', 
                        default='/home/dodo/ljx/vln_data',
                        help='Path to vln_data directory')
    parser.add_argument('--qh_key', 
                        default='qh_mean_values',
                        help='Key name for qh_mean in HDF5 files')
    parser.add_argument('--vh_key', 
                        default='vh_values',
                        help='Key name for vh in HDF5 files')
    parser.add_argument('--task_filter', 
                        default=None,
                        help='Only process tasks matching this filter (optional)')
    
    args = parser.parse_args()
    
    print("QH/VH to HDF5 Writer")
    print("=" * 50)
    print(f"Source directory: {args.viz_path_all}")
    print(f"Target directory: {args.vln_data_dir}")
    print(f"QH key: {args.qh_key}")
    print(f"VH key: {args.vh_key}")
    print("=" * 50)
    
    # 检查源目录是否存在
    if not os.path.exists(args.viz_path_all):
        print(f"Error: Source directory {args.viz_path_all} does not exist")
        return
    
    # 检查目标目录是否存在
    if not os.path.exists(args.vln_data_dir):
        print(f"Error: Target directory {args.vln_data_dir} does not exist")
        return
    
    # 获取所有任务目录
    task_dirs = glob.glob(os.path.join(args.viz_path_all, "*"))
    task_dirs = [d for d in task_dirs if os.path.isdir(d)]
    
    # 应用任务过滤器
    if args.task_filter:
        task_dirs = [d for d in task_dirs if args.task_filter.lower() in os.path.basename(d).lower()]
    
    print(f"Found {len(task_dirs)} task directories to process")
    
    total_stats = {'processed': 0, 'failed': 0, 'skipped': 0}
    
    # 处理每个任务目录
    qh_key = f"{args.viz_path_all}/{args.qh_key}"
    vh_key = f"{args.viz_path_all}/{args.vh_key}"
    for task_dir in task_dirs:
        stats = process_task_directory(task_dir, args.vln_data_dir, qh_key, vh_key)
        
        # 累计统计
        for key in total_stats:
            total_stats[key] += stats[key]
        
        print(f"Task {os.path.basename(task_dir)}: "
              f"Processed={stats['processed']}, "
              f"Failed={stats['failed']}, "
              f"Skipped={stats['skipped']}")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total failed: {total_stats['failed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print("=" * 50)


if __name__ == '__main__':
    main()