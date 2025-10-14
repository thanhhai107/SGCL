import argparse
import torch
import os

# ===== DISTRIBUTED TRAINING PATCH =====
# This patch fixes the torch.distributed.barrier error for single GPU training
# Works on both local and Kaggle environments

def is_kaggle_environment():
    """Check if running in Kaggle environment"""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None or 'kaggle' in os.environ.get('HOSTNAME', '').lower()

def safe_barrier(*args, **kwargs):
    """Safe wrapper for torch.distributed.barrier"""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.barrier(*args, **kwargs)
        return
    except Exception as e:
        print(f"Warning: Distributed barrier call failed (safe to ignore for single GPU): {e}")
        return

def safe_is_initialized():
    """Safe wrapper for torch.distributed.is_initialized"""
    try:
        return torch.distributed.is_initialized()
    except:
        return False

def safe_get_rank():
    """Safe wrapper for torch.distributed.get_rank"""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0
    except:
        return 0

def safe_get_world_size():
    """Safe wrapper for torch.distributed.get_world_size"""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1
    except:
        return 1

# Apply the patch
torch.distributed.barrier = safe_barrier
torch.distributed.is_initialized = safe_is_initialized
torch.distributed.get_rank = safe_get_rank
torch.distributed.get_world_size = safe_get_world_size

# Setup environment variables for single GPU training
if is_kaggle_environment():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'

# ===== END DISTRIBUTED PATCH =====

from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=str, default='rec', help='mode of training')
    parser.add_argument('--model', '-m', type=str, default='SGCL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='amazon-beauty', help='name of datasets')
    parser.add_argument('--lr', '-l', type=str, default=0.001, help='learing rate')
    parser.add_argument('--weight_decay', '-w', type=str, default=1e-5, help='weight_decay')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='gpu id')
    parser.add_argument('--suffix', '-s', type=str, default=None, help='log suffix')
    parser.add_argument('--config_files', type=str, default='config.yaml', help='config files')
    parser.add_argument('--tem', '-t', type=str, default=None, help='temperature')
    
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, gpu=args.gpu, suffix=args.suffix, lr=args.lr, weight_decay=args.weight_decay, tem=args.tem, config_file_list=config_file_list, saved=False)