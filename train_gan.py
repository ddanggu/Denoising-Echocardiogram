"""
Supervised training script for paired GAN dehazing.
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
from networks.gan import GAN
from training.checkpoint import save_checkpoint, load_checkpoint
from training.losses import EdgeLoss, GANLoss, SSIM, SSIMLoss


def get_opt():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, default="/workspace/share/haze")
    p.add_argument("--scale", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints_gan")
    p.add_argument("--save_freq", type=int, default=20)
    p.add_argument("--val_freq", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lr_G", type=float, default=2e-4)
    p.add_argument("--lr_D", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--out_ch", type=int, default=1)
    p.add_argument("--G_init_ch", type=int, default=64)
    p.add_argument("--D_init_ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out_act", type=str, default=None)
    p.add_argument("--norm", type=str, default="batch")
    p.add_argument("--gan_mode", type=str, default="lsgan", choices=["lsgan", "vanilla"])
    p.add_argument("--lambda_gan", type=float, default=1.0)
    p.add_argument("--lambda_pair", type=float, default=100.0)
    p.add_argument("--lambda_ssim", type=float, default=0.0)
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
    model.G.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    ssim_metric = SSIM(reduction="mean")
    with torch.no_grad():
        for img, label, _ in val_loader:
            img, label = img.to(device), label.to(device)
            pred = model.G(img)
            bs = img.size(0)
            psnr_sum += batch_psnr(pred, label) * bs
            ssim_sum += ssim_metric(pred * 0.5 + 0.5, label * 0.5 + 0.5).item() * bs
            n += bs
    model.G.train()
    return {"psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1)}


def save_state(tag, model, optimizers, epoch, metrics, checkpoints_dir):
    ckpt_dir = Path(checkpoints_dir)
    save_checkpoint(ckpt_dir / f"{tag}_ckpt_SS.pth", model, optimizers, epoch, metrics)
    save_checkpoint(ckpt_dir / f"{tag}_net_SS_G.pth", model.G, {"G": optimizers["G"]}, epoch, metrics)
    save_checkpoint(ckpt_dir / f"{tag}_net_SS_D.pth", model.D, {"D": optimizers["D"]}, epoch, metrics)


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

    model = GAN(opt.in_ch, opt.out_ch, opt.G_init_ch, opt.D_init_ch, opt.depth, opt.out_act, opt.norm).to(device)
    optimizer_G = torch.optim.Adam(model.G.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
    optimizer_D = torch.optim.Adam(model.D.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=opt.n_epochs)
    optimizers = {"G": optimizer_G, "D": optimizer_D}
    gan_loss = GANLoss(opt.gan_mode)
    pair_loss = torch.nn.L1Loss()
    ssim_loss = SSIMLoss() if opt.lambda_ssim > 0 else None
    edge_loss = EdgeLoss(mode=opt.edge_mode, blur=not opt.no_edge_blur, sigma=opt.edge_sigma) if opt.lambda_edge > 0 else None

    start_epoch = 1
    best_psnr, best_epoch = -float("inf"), 0
    if opt.resume:
        ckpt = load_checkpoint(opt.resume, model, optimizers, device=str(device))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("metrics", {}).get("best_psnr", best_psnr))
        best_epoch = int(ckpt.get("metrics", {}).get("best_epoch", best_epoch))

    save_state("init", model, optimizers, start_epoch - 1, {"best_psnr": best_psnr, "best_epoch": best_epoch}, opt.checkpoints_dir)

    train_start = time.time()
    for epoch in range(start_epoch, opt.n_epochs + 1):
        epoch_start = time.time()
        acc = {"G_gan": 0.0, "G_pair": 0.0, "G_ssim": 0.0, "G_edge": 0.0, "G_total": 0.0, "D": 0.0}

        for img, label, _ in train_loader:
            img, label = img.to(device), label.to(device)

            with torch.no_grad():
                fake = model.G(img)
            loss_D_real = gan_loss(model.D(label), True)
            loss_D_fake = gan_loss(model.D(fake.detach()), False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            fake = model.G(img)
            loss_G_gan = gan_loss(model.D(fake), True) * opt.lambda_gan
            loss_G_pair = pair_loss(fake, label) * opt.lambda_pair
            loss_G_ssim = fake.new_tensor(0.0)
            if ssim_loss is not None:
                loss_G_ssim = ssim_loss(fake * 0.5 + 0.5, label * 0.5 + 0.5) * opt.lambda_ssim
            loss_G_edge = fake.new_tensor(0.0)
            if edge_loss is not None:
                loss_G_edge = edge_loss(fake * 0.5 + 0.5, label * 0.5 + 0.5) * opt.lambda_edge
            loss_G = loss_G_gan + loss_G_pair + loss_G_ssim + loss_G_edge
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            acc["G_gan"] += loss_G_gan.item()
            acc["G_pair"] += loss_G_pair.item()
            acc["G_ssim"] += loss_G_ssim.item()
            acc["G_edge"] += loss_G_edge.item()
            acc["G_total"] += loss_G.item()
            acc["D"] += loss_D.item()

        sched_G.step()
        sched_D.step()
        steps = max(len(train_loader), 1)
        lr_now = optimizer_G.param_groups[0]["lr"]
        log = "  ".join(f"{k}={v / steps:.4f}" for k, v in acc.items())

        val_str = ""
        metrics = {"best_psnr": best_psnr, "best_epoch": best_epoch}
        if epoch % opt.val_freq == 0:
            val = validate(model, val_loader, device)
            val_str = f"  val_PSNR={val['psnr']:.2f}dB  val_SSIM={val['ssim']:.4f}"
            if val["psnr"] > best_psnr:
                best_psnr, best_epoch = val["psnr"], epoch
                metrics.update({"best_psnr": best_psnr, "best_epoch": best_epoch, **val})
                save_state("best", model, optimizers, epoch, metrics, opt.checkpoints_dir)
                val_str += " <- best"

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        remaining = epoch_time * (opt.n_epochs - epoch)
        print(f"[{epoch:03d}/{opt.n_epochs}] {log}  lr={lr_now:.6f}{val_str}"
              f"  [{epoch_time:.0f}s/epoch | elapsed {_fmt(elapsed)} | ETA {_fmt(remaining)}]")

        if epoch % opt.save_freq == 0:
            save_state(str(epoch), model, optimizers, epoch, metrics, opt.checkpoints_dir)

    save_state("final", model, optimizers, opt.n_epochs, {"best_psnr": best_psnr, "best_epoch": best_epoch}, opt.checkpoints_dir)
    print(f"\nTraining complete. Total time: {_fmt(time.time() - train_start)}")
    print(f"Best val PSNR: {best_psnr:.2f} dB at epoch {best_epoch}")
    print(f"Best checkpoint: {Path(opt.checkpoints_dir) / 'best_net_SS_G.pth'}")


if __name__ == "__main__":
    train()
