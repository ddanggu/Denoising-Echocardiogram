# Denoising-Echocardiogram

Echocardiogram 영상의 noise, haze, clutter 성분을 줄이고 더 선명한 clean ultrasound image를 복원하기 위한 딥러닝 기반 denoising 프로젝트입니다.

본 프로젝트는 paired noisy-clean echocardiogram 데이터를 이용해 U-Net, GAN, CycleGAN 기반 모델을 학습하고, 학습된 checkpoint로 단일 이미지 또는 폴더 단위 inference를 수행할 수 있도록 구성되어 있습니다.

## Features

- Paired echocardiogram denoising / dehazing 학습 지원
- U-Net, GAN, CycleGAN 모델 제공
- TIFF, PNG, JPG, BMP 등 일반 이미지 입력 지원
- 학습 중 validation PSNR / SSIM 평가
- Inference 결과를 TIFF 형식으로 저장
- Ground truth가 있는 경우 PSNR / SSIM 계산 가능
- ROI 지정 시 SNR / CNR 개선 정도 계산 가능

## Project Structure

```text
Dehazing/
├── train_unet.py          # U-Net supervised training
├── train_gan.py           # GAN supervised training
├── train_cyclegan.py      # CycleGAN supervised training
├── infer_unet.py          # U-Net inference
├── infer_gan.py           # GAN inference
├── infer_cyclegan.py      # CycleGAN inference
├── infer_common.py        # Shared inference utilities
├── datasets/
│   └── dataset.py         # Dataset loader
├── networks/
│   ├── unet.py
│   └── gan.py
└── training/
    ├── checkpoint.py
    └── losses.py
```

## Dataset Format

Training data should be arranged as paired noisy-clean images with the same file names.

```text
dataset_path/
├── train/
│   ├── noisy/
│   │   ├── sample_001.tiff
│   │   └── ...
│   └── clean/
│       ├── sample_001.tiff
│       └── ...
└── valid/
    ├── noisy/
    │   ├── sample_001.tiff
    │   └── ...
    └── clean/
        ├── sample_001.tiff
        └── ...
```

Supported image formats:

```text
.tiff, .tif, .png, .jpg, .jpeg, .bmp
```

Input images are expected to be normalized image arrays in the range `[0, 1]`.

## Requirements

Main dependencies:

```text
python
torch
numpy
tifffile
Pillow
scipy
```

Install example:

```bash
pip install torch numpy tifffile pillow scipy
```

## Training

Move to the project directory first:

```bash
cd SamMed/Dehazing
```

### Train U-Net

```bash
python train_unet.py \
  --dataset_path /path/to/dataset \
  --scale 128 \
  --batch_size 8 \
  --n_epochs 150 \
  --gpu 0
```

Default checkpoint directory:

```text
./checkpoints_unet
```

Best model:

```text
./checkpoints_unet/best_net_SS_G.pth
```

### Train GAN

```bash
python train_gan.py \
  --dataset_path /path/to/dataset \
  --scale 128 \
  --batch_size 8 \
  --n_epochs 150 \
  --gpu 0
```

Default checkpoint directory:

```text
./checkpoints_gan
```

Best generator checkpoint:

```text
./checkpoints_gan/best_net_SS_G.pth
```

### Train CycleGAN

```bash
python train_cyclegan.py \
  --dataset_path /path/to/dataset \
  --scale 128 \
  --batch_size 8 \
  --n_epochs 150 \
  --gpu 0
```

Default checkpoint directory:

```text
./checkpoints_cyclegan
```

Best noisy-to-clean generator checkpoint:

```text
./checkpoints_cyclegan/best_net_SS_G_XtoY.pth
```

### Useful Training Options

```text
--dataset_path      Dataset root path
--scale             Resize image to NxN, default: 128
--batch_size        Batch size, default: 8
--n_epochs          Number of training epochs, default: 150
--checkpoints_dir   Directory to save checkpoints
--save_freq         Checkpoint save frequency
--val_freq          Validation frequency
--gpu               GPU index
--resume            Resume training from checkpoint
--lambda_ssim       SSIM loss weight, 0 disables it
--lambda_edge       Edge loss weight, 0 disables it
--edge_mode         sobel or laplacian
```

For GAN / CycleGAN:

```text
--lambda_gan        GAN loss weight
--lambda_pair       Paired L1 reconstruction loss weight
--gan_mode          lsgan or vanilla
```

## Inference

Inference can be run on either a single image or a directory of images.
The output images are saved as TIFF files with the suffix `_dehazed.tiff`.

### U-Net Inference

```bash
python infer_unet.py \
  --input /path/to/noisy_images \
  --output ./results_unet \
  --checkpoint ./checkpoints_unet/best_net_SS_G.pth \
  --scale 128 \
  --gpu 0
```

### GAN Inference

```bash
python infer_gan.py \
  --input /path/to/noisy_images \
  --output ./results_gan \
  --checkpoint ./checkpoints_gan/best_net_SS_G.pth \
  --scale 128 \
  --gpu 0
```

### CycleGAN Inference

```bash
python infer_cyclegan.py \
  --input /path/to/noisy_images \
  --output ./results_cyclegan \
  --checkpoint ./checkpoints_cyclegan/best_net_SS_G_XtoY.pth \
  --direction XtoY \
  --scale 128 \
  --gpu 0
```

Use CPU inference with:

```bash
--gpu -1
```

## Evaluation During Inference

If ground truth clean images are available, PSNR and SSIM can be calculated during inference.

```bash
python infer_unet.py \
  --input /path/to/test/noisy \
  --gt /path/to/test/clean \
  --metrics \
  --output ./results_unet \
  --checkpoint ./checkpoints_unet/best_net_SS_G.pth
```

The ground truth directory should contain files with the same names as the input images.

## ROI-based SNR / CNR Measurement

SNR and CNR can be calculated by specifying signal and noise ROIs.

ROI format:

```text
x,y,width,height
```

Example:

```bash
python infer_unet.py \
  --input /path/to/noisy_images \
  --output ./results_unet \
  --checkpoint ./checkpoints_unet/best_net_SS_G.pth \
  --roi_signal 40,40,20,20 \
  --roi_noise 90,90,20,20
```

The script reports SNR and CNR before and after denoising.

## Output

For each input image:

```text
input_name.tiff -> input_name_dehazed.tiff
```

The inference script also prints:

```text
- Inference time per image
- Throughput
- PSNR / SSIM, if ground truth is provided
- SNR / CNR, if ROI is provided
```
