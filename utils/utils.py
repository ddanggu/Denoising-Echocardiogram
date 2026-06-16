from typing import List, Any

import os
import sys
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from natsort import natsorted
from pathlib import Path

import torch


#### GPU Functions ####
_dummy_memory: torch.Tensor | None = None
def allocate_dummy_gpu_memory(gb_coef: int | float):
    # _module = sys.modules[__name__]
    global _dummy_memory
    gb_size = int(gb_coef * 1024**3)

    # _module._dummy_memory = torch.empty((gb_size, ), dtype=torch.int8, device='cuda')
    _dummy_memory = torch.empty((gb_size, ), dtype=torch.int8, device='cuda')
    print(f"Allocated dummy GPU memory: {gb_size / 1024**2:.2f} MiB")

def release_dummy_gpu_memory():
    import gc
    # _module = sys.modules[__name__]
    global _dummy_memory

    # _module._dummy_memory = None
    _dummy_memory = None

    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released dummy GPU memory. {torch.cuda.memory_allocated() / 1024**2:.2f} MiB allocated, {torch.cuda.memory_reserved() / 1024**2:.2f} MiB reserved")


#### Base Functions ####
def listdir(p: str | Path):
    p = Path(p)
    return natsorted(item.name for item in p.iterdir())

def globsort(p: str | Path):
    p = Path(p)
    return natsorted(p.iterdir())

def mk_dir(ps: str | Path | List[str | Path]):
    if not isinstance(ps, (list, tuple)):
        ps = [ps]
    
    for p in ps:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        
        os.chmod(p, 0o777)

def minMaxscaler(data: Any, lb: float = 0., ub: float | None = None, 
                 m: Any | None = None, M: Any | None = None):
    m_val = data.min() if m is None else m
    M_val = data.max() if M is None else M

    if ub is None:
        ub = 1 if lb == 0 else abs(lb)

    return (ub - lb) * (data - m_val) / (M_val - m_val) + lb