import numpy as np
import tifffile as tiff

from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def list_us_image(dataset_dir: str | Path, split: str | None, image_dir: str):
    data_dir = Path(dataset_dir) / split / image_dir if split is not None else Path(dataset_dir) / image_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Expected image directory: {data_dir}')

    return sorted(
        [p.name for p in data_dir.iterdir() 
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )

def load_us_image(path: str | Path, resize: int | tuple[int, int] | None = 112):
    path = Path(path)

    if path.suffix.lower() in ['.tif', '.tiff']:
        arr = tiff.imread(path)
    else:
        img = Image.open(path).convert('L')
        arr = np.asarray(img)

    arr = np.clip(arr, 0, 1)
    x = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)
    if resize is not None:
        size = (resize, resize) if isinstance(resize, int) else resize
        x = F.interpolate(x.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze(0)
    
    # x = x * 2. - 1.
    x = (x - 0.5) / 0.5
    
    return x

class USDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, files: list[str] | None = None,
                 resize: int | tuple[int, int] | None = 112, labeled: bool = True):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.resize      = resize
        self.labeled     = labeled 
        self.files       = list(files) if files is not None else list_us_image(self.dataset_dir, None, 'noisy')

    def __len__(self):
        return len(self.files)
    
    def _load(self, folder: str, fname: str):
        return load_us_image(self.dataset_dir / folder / fname, self.resize)
    
    def __getitem__(self, idx: int):
        fname = self.files[idx]
        noisy = self._load('noisy', fname)

        if self.labeled:
            clean = self._load('clean', fname)
        else:
            clean = noisy.clone()

        return noisy, clean, fname
    
def split_labeled_files(files: list[str], labeled_ratio: float = 0.3, seed: int = 42):
    if not 0.0 < labeled_ratio <= 1.0:
        raise ValueError(f'labeled_ratio must be in (0, 1], got {labeled_ratio}')
    
    n_labeled = max(1, int(len(files) * labeled_ratio))
    indices = np.random.default_rng(seed).permutation(len(files))
    labeled = [files[i] for i in indices[:n_labeled]]
    unlabeled = [files[i] for i in indices[n_labeled:]]
    
    return labeled, unlabeled
    
def build_loaders(root: str | Path, resize: int | tuple[int, int] | None = 112,
                  batch_size: int = 4, supervised_only: bool = False, labeled_ratio: float = 0.3, seed: int = 42, 
                  num_workers: int = 2, pin_memory: bool = False, persistent_workers: bool | None = None, prefetch_factor: int | None = 2):
    root = Path(root)
    train_dir = root / 'train'
    valid_dir = root / 'valid'

    train_dataset = USDataset(train_dir, resize=resize, labeled=True)
    valid_dataset = USDataset(valid_dir, resize=resize, labeled=True)
    train_files   = train_dataset.files

    if persistent_workers is None: persistent_workers = num_workers > 0
    loader_kwargs = {
        'batch_size'        : batch_size,
        'num_workers'       : num_workers,
        'pin_memory'        : pin_memory,
        # 'persistent_workers': persistent_workers,
    }
    # if num_workers > 0 and prefetch_factor is not None:
    #     loader_kwargs['prefetch_factor'] = prefetch_factor
    
    if supervised_only:
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
        valid_loader = DataLoader(valid_dataset, shuffle=True, drop_last=True, **loader_kwargs)

        return train_loader, valid_loader
    
    labeled_files, unlabeled_files = split_labeled_files(train_files, labeled_ratio, seed)

    labeled_dataset     = USDataset(train_dir, labeled_files,   resize=resize, labeled=True)
    unlabeled_dataset   = USDataset(train_dir, unlabeled_files, resize=resize, labeled=False)

    labeled_loader      = DataLoader(labeled_dataset,   shuffle=True,   drop_last=True,     **loader_kwargs)
    unlabeled_loader    = DataLoader(unlabeled_dataset, shuffle=True,   drop_last=True,     **loader_kwargs)
    valid_loader        = DataLoader(valid_dataset,     shuffle=False,  drop_last=False,    **loader_kwargs)

    return labeled_loader, unlabeled_loader, valid_loader