"""
Shared inference utilities for SamMed dehazing models.
"""

import glob
import os
import time
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTS = ("*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp",
              "*.TIFF", "*.TIF", "*.PNG", "*.JPG", "*.JPEG", "*.BMP")


def collect_files(input_path):
    input_path = str(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(input_path, ext)))
    return sorted(files)


def load_image(path, scale):
    path = Path(path)
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(path)
    else:
        arr = np.asarray(Image.open(path).convert("L"))
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    if scale is not None:
        size = (scale, scale) if isinstance(scale, int) else tuple(scale)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
    x = (x.squeeze(0) - 0.5) / 0.5
    return x


def to_numpy_01(tensor):
    arr = tensor.squeeze().detach().cpu().float().numpy()
    return np.clip(arr * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


def save_tiff(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, np.asarray(arr, dtype=np.float32))


def psnr(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim_np(pred, gt, window_size=11):
    from scipy.signal import convolve2d
    sigma = 1.5
    half = window_size // 2
    coords = np.arange(window_size) - half
    kernel_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    def filt(x):
        return convolve2d(x, kernel_2d, mode="same", boundary="symm")

    mu1 = filt(pred)
    mu2 = filt(gt)
    mu1sq = mu1 ** 2
    mu2sq = mu2 ** 2
    mu12 = mu1 * mu2
    s1sq = filt(pred * pred) - mu1sq
    s2sq = filt(gt * gt) - mu2sq
    s12 = filt(pred * gt) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu12 + c1) * (2 * s12 + c2)
    den = (mu1sq + mu2sq + c1) * (s1sq + s2sq + c2)
    return float(np.mean(num / (den + 1e-8)))


def parse_roi(s):
    x, y, w, h = [int(v) for v in s.split(",")]
    return y, y + h, x, x + w


def compute_snr_cnr(arr, signal_roi, noise_roi):
    sr0, sr1, sc0, sc1 = signal_roi
    nr0, nr1, nc0, nc1 = noise_roi
    mu_s = float(np.mean(arr[sr0:sr1, sc0:sc1]))
    mu_n = float(np.mean(arr[nr0:nr1, nc0:nc1]))
    std_n = float(np.std(arr[nr0:nr1, nc0:nc1]))
    snr = 20.0 * np.log10(mu_s / std_n) if std_n > 1e-10 and mu_s > 0 else 0.0
    cnr = 20.0 * np.log10(abs(mu_s - mu_n) / std_n) if std_n > 1e-10 and abs(mu_s - mu_n) > 1e-10 else 0.0
    return float(snr), float(cnr)


def load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt
    return ckpt, ckpt if isinstance(ckpt, dict) else {}


def strip_prefix(state, prefix):
    plen = len(prefix)
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def run_inference(opt, model, forward_fn):
    device = torch.device("cpu") if opt.gpu < 0 else torch.device(f"cuda:{opt.gpu}")
    model.to(device)
    model.eval()

    files = collect_files(opt.input)
    if not files:
        print(f"No image files found in: {opt.input}")
        return

    print(f"Checkpoint : {opt.checkpoint}")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU name   : {torch.cuda.get_device_name(opt.gpu)}")
    print(f"Images     : {len(files)}")
    print(f"Output     : {opt.output}")

    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    psnr_list, ssim_list = [], []
    snr_noisy_list, snr_pred_list = [], []
    cnr_noisy_list, cnr_pred_list = [], []
    infer_times = []
    sig_roi = parse_roi(opt.roi_signal) if opt.roi_signal and opt.roi_noise else None
    noise_roi = parse_roi(opt.roi_noise) if opt.roi_signal and opt.roi_noise else None

    total_start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(files), opt.batch_size):
            batch_paths = files[i:i + opt.batch_size]
            tensors = [load_image(p, opt.scale) for p in batch_paths]
            batch = torch.stack(tensors, dim=0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred = forward_fn(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_ms = (time.perf_counter() - t0) * 1000.0 / len(batch_paths)
            infer_times.extend([infer_ms] * len(batch_paths))

            for j, src_path in enumerate(batch_paths):
                fname = os.path.basename(src_path)
                stem = os.path.splitext(fname)[0]
                pred_arr = to_numpy_01(pred[j])
                out_path = output_dir / f"{stem}_dehazed.tiff"
                save_tiff(pred_arr, out_path)

                roi_str = ""
                if sig_roi is not None and noise_roi is not None:
                    noisy_arr = to_numpy_01(batch[j])
                    snr_pred, cnr_pred = compute_snr_cnr(pred_arr, sig_roi, noise_roi)
                    snr_noisy, cnr_noisy = compute_snr_cnr(noisy_arr, sig_roi, noise_roi)
                    snr_noisy_list.append(snr_noisy)
                    snr_pred_list.append(snr_pred)
                    cnr_noisy_list.append(cnr_noisy)
                    cnr_pred_list.append(cnr_pred)
                    roi_str = (f"  SNR:{snr_noisy:.1f}->{snr_pred:.1f}dB(Delta{snr_pred-snr_noisy:+.1f})"
                               f"  CNR:{cnr_noisy:.1f}->{cnr_pred:.1f}dB(Delta{cnr_pred-cnr_noisy:+.1f})")

                metric_str = ""
                if opt.metrics and opt.gt:
                    gt_path = Path(opt.gt) / fname
                    if gt_path.exists():
                        gt_arr = to_numpy_01(load_image(gt_path, opt.scale))
                        p = psnr(pred_arr, gt_arr)
                        s = ssim_np(pred_arr, gt_arr)
                        psnr_list.append(p)
                        ssim_list.append(s)
                        metric_str = f"  PSNR={p:.2f}dB  SSIM={s:.4f}"
                    else:
                        metric_str = "  (no GT found)"

                print(f"  [{i+j+1:4d}/{len(files)}] {fname}{metric_str}{roi_str}"
                      f"  infer={infer_ms:.1f}ms -> {out_path}")

    total_elapsed = time.perf_counter() - total_start

    if psnr_list:
        print(f"\nMean PSNR : {np.mean(psnr_list):.2f} dB")
        print(f"Mean SSIM : {np.mean(ssim_list):.4f}")

    if snr_noisy_list:
        print(f"\n=== SNR / CNR (signal ROI: {opt.roi_signal} noise ROI: {opt.roi_noise}) ===")
        print(f"  SNR noisy   : {np.mean(snr_noisy_list):.2f} dB (std {np.std(snr_noisy_list):.2f})")
        print(f"  SNR output  : {np.mean(snr_pred_list):.2f} dB (std {np.std(snr_pred_list):.2f})")
        print(f"  Delta SNR   : {np.mean(np.array(snr_pred_list) - np.array(snr_noisy_list)):+.2f} dB")
        print(f"  CNR noisy   : {np.mean(cnr_noisy_list):.2f} dB (std {np.std(cnr_noisy_list):.2f})")
        print(f"  CNR output  : {np.mean(cnr_pred_list):.2f} dB (std {np.std(cnr_pred_list):.2f})")
        print(f"  Delta CNR   : {np.mean(np.array(cnr_pred_list) - np.array(cnr_noisy_list)):+.2f} dB")

    if infer_times:
        print("\n=== Inference Time ===")
        print(f"  Per image mean : {np.mean(infer_times):.2f} ms")
        print(f"  Per image std  : {np.std(infer_times):.2f} ms")
        print(f"  Throughput     : {1000 / np.mean(infer_times):.1f} images/sec")
        print(f"  Total elapsed  : {total_elapsed:.1f}s")

    print(f"\nDone. TIFF results saved to: {opt.output}")
