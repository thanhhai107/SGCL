"""
Distributed training patch for RecBole-GNN to handle single GPU training.
This module provides a safe wrapper for torch.distributed functions.
"""

import torch
import functools


def safe_barrier(*args, **kwargs):
    """
    Safe wrapper for torch.distributed.barrier that only executes if process group is initialized.
    For single GPU training, this function does nothing.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.barrier(*args, **kwargs)
    else:
        # If distributed training is not initialized, just return without doing anything
        # This is safe for single GPU training
        return


def apply_distributed_patch():
    """
    Apply the distributed training patch to torch.distributed module.
    This should be called before importing RecBole modules.
    """
    # Store the original barrier function
    torch.distributed._original_barrier = torch.distributed.barrier
    # Replace with our safe version
    torch.distributed.barrier = safe_barrier


def remove_distributed_patch():
    """
    Remove the distributed training patch and restore original function.
    """
    if hasattr(torch.distributed, '_original_barrier'):
        torch.distributed.barrier = torch.distributed._original_barrier
        delattr(torch.distributed, '_original_barrier')


# Auto-apply patch when this module is imported
apply_distributed_patch()
