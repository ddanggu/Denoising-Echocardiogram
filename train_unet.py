"""
Supervised training script for UNet dehazing.

Expected dataset layout:
  dataset_path/
    train/noisy, train/clean
    valid/noisy, valid/clean
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import importlib.util


def _load_build_loaders():
    dataset_path = ROOT / "datasets" / "dataset.py"
    spec = importlib.util.spec_from_file_location("sammed_dataset", dataset_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_loaders


build_loaders = _load_build_loaders()
from networks.unet import UNet
from training.checkpoint import save_checkpoint, load_checkpoint
from training.losses import EdgeLoss, SSIM, SSIMLoss


def get_opt():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, default="/workspace/share/haze")
    p.add_argument("--scale", type=int, default=128, help="Resize to NxN")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints_unet")
    p.add_argument("--save_freq", type=int, default=20)
    p.add_argument("--val_freq", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--out_ch", type=int, default=1)
    p.add_argument("--init_ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lambda_pair", type=float, default=1.0)
    p.add_argument("--lambda_ssim", type=float, default=0.0, help="0 disables SSIMLoss")
    p.add_argument("--lambda_edge", type=float, default=0.0, help="0 disables EdgeLoss")
    p.add_argument("--edge_mode", type=str, default="sobel", choices=["sobel", "laplacian"])
    p.add_argument("--edge_sigma", type=float, default=0.8)
    p.add_argument("--no_edge_blur", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    opt = p.parse_args()
    opt.isTrain = True
    return opt


def _fmt(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def batch_psnr(pred, gt):
    pred01 = (pred.float() * 0.5 + 0.5).clamp(0, 1)
    gt01 = (gt.float() * 0.5 + 0.5).clamp(0, 1)
    mse = F.mse_loss(pred01, gt01, reduction="none").mean(dim=[1, 2, 3])
    return (10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))).mean().item()


def validate(model, val_loader, device):
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    ssim_metric = SSIM(reduction="mean")
    with torch.no_grad():
        for img, label, _ in val_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            bs = img.size(0)
            psnr_sum += batch_psnr(pred, label) * bs
            ssim_sum += ssim_metric(pred * 0.5 + 0.5, label * 0.5 + 0.5).item() * bs
            n += bs
    model.train()
    return {"psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1)}


def save_state(tag, model, optimizer, epoch, metrics, checkpoints_dir):
    ckpt_dir = Path(checkpoints_dir)
    save_checkpoint(ckpt_dir / f"{tag}_ckpt_SS.pth", model, {"G": optimizer}, epoch, metrics)
    save_checkpoint(ckpt_dir / f"{tag}_net_SS_G.pth", model, {"G": optimizer}, epoch, metrics)


def train():
    opt = get_opt()
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    for split in ("train", "valid"):
        split_noisy = Path(opt.dataset_path) / split / "noisy"
        split_clean = Path(opt.dataset_path) / split / "clean"
        if not split_noisy.is_dir() or not split_clean.is_dir():
            print(f"ERROR: expected directories not found: {split_noisy}, {split_clean}")
            return

    train_loader, val_loader = build_loaders(
        root=opt.dataset_path,
        resize=opt.scale,
        batch_size=opt.batch_size,
        supervised_only=True,
        num_workers=opt.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")

    model = UNet(opt.in_ch, opt.out_ch, opt.init_ch, opt.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs)
    pair_loss = torch.nn.L1Loss()
    ssim_loss = SSIMLoss() if opt.lambda_ssim > 0 else None
    edge_loss = EdgeLoss(mode=opt.edge_mode, blur=not opt.no_edge_blur, sigma=opt.edge_sigma) if opt.lambda_edge > 0 else None

    start_epoch = 1
    best_psnr, best_epoch = -float("inf"), 0
    if opt.resume:
        ckpt = load_checkpoint(opt.resume, model, {"G": optimizer}, device=str(device))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("metrics", {}).get("best_psnr", best_psnr))
        best_epoch = int(ckpt.get("metrics", {}).get("best_epoch", best_epoch))

    save_state("init", model, optimizer, start_epoch - 1, {"best_psnr": best_psnr, "best_epoch": best_epoch}, opt.checkpoints_dir)

    train_start = time.time()
    for epoch in range(start_epoch, opt.n_epochs + 1):
        epoch_start = time.time()
        acc = {"pair": 0.0, "ssim": 0.0, "edge": 0.0, "G_total": 0.0}

        for img, label, _ in train_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss_pair = pair_loss(pred, label) * opt.lambda_pair
            loss_ssim = pred.new_tensor(0.0)
            if ssim_loss is not None:
                loss_ssim = ssim_loss(pred * 0.5 + 0.5, label * 0.5 + 0.5) * opt.lambda_ssim
            loss_edge = pred.new_tensor(0.0)
            if edge_loss is not None:
                loss_edge = edge_loss(pred * 0.5 + 0.5, label * 0.5 + 0.5) * opt.lambda_edge
            loss = loss_pair + loss_ssim + loss_edge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc["pair"] += loss_pair.item()
            acc["ssim"] += loss_ssim.item()
            acc["edge"] += loss_edge.item()
            acc["G_total"] += loss.item()

        scheduler.step()
        steps = max(len(train_loader), 1)
        lr_now = optimizer.param_groups[0]["lr"]
        log = "  ".join(f"{k}={v / steps:.4f}" for k, v in acc.items())

        val_str = ""
        metrics = {"best_psnr": best_psnr, "best_epoch": best_epoch}
        if epoch % opt.val_freq == 0:
            val = validate(model, val_loader, device)
            val_str = f"  val_PSNR={val['psnr']:.2f}dB  val_SSIM={val['ssim']:.4f}"
            if val["psnr"] > best_psnr:
                best_psnr, best_epoch = val["psnr"], epoch
                metrics.update({"best_psnr": best_psnr, "best_epoch": best_epoch, **val})
                save_state("best", model, optimizer, epoch, metrics, opt.checkpoints_dir)
                val_str += " <- best"

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        remaining = epoch_time * (opt.n_epochs - epoch)
        print(f"[{epoch:03d}/{opt.n_epochs}] {log}  lr={lr_now:.6f}{val_str}"
              f"  [{epoch_time:.0f}s/epoch | elapsed {_fmt(elapsed)} | ETA {_fmt(remaining)}]")

        if epoch % opt.save_freq == 0:
            save_state(str(epoch), model, optimizer, epoch, metrics, opt.checkpoints_dir)

    save_state("final", model, optimizer, opt.n_epochs, {"best_psnr": best_psnr, "best_epoch": best_epoch}, opt.checkpoints_dir)
    print(f"\nTraining complete. Total time: {_fmt(time.time() - train_start)}")
    print(f"Best val PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")
    print(f"Best checkpoint: {Path(opt.checkpoints_dir) / 'best_net_SS_G.pth'}")


if __name__ == "__main__":
    train()
