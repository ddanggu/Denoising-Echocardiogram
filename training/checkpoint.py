from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizers: dict, epoch: int, metrics: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizers: dict | None = None, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizers is not None:
        for name, opt in optimizers.items():
            if name in ckpt.get("optimizers", {}):
                opt.load_state_dict(ckpt["optimizers"][name])

    return ckpt
