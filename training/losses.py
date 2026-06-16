import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class GANLoss(nn.Module):
    def __init__(self, mode: str = 'lsgan'):
        super().__init__()
        
        self.loss = nn.MSELoss() if mode == 'lsgan' else nn.BCELoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool):
        target = torch.ones_like(pred, dtype=pred.dtype, device=pred.device) \
            if target_is_real else torch.zeros_like(pred, dtype=pred.dtype, device=pred.device)
        
        return self.loss(pred, target)

def _gaussian(window_size: int, sigma: float):
    coords  = torch.arange(window_size).float() - window_size // 2
    gauss   = torch.exp(-(coords**2) / (2 * sigma**2))

    return gauss / gauss.sum()

def _create_window(window_size: int, channel: int, sigma: float = 1.5):
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    
    return window

def _reduce(x: torch.Tensor, reduction: str):
    if reduction == 'mean':
        return x.mean()
    
    elif reduction == 'sum':
        return x.sum()
    
    elif reduction == 'none':
        return x
    
    else:
        raise ValueError(f'Invalid reduction: {reduction}')

class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction
        self.eps = eps
    
    def _compute_ssim_map(self, pred: torch.Tensor, target: torch.Tensor):
        channel = pred.shape[1]
        window  = _create_window(self.window_size, channel, self.sigma).to(
            device=pred.device,
            dtype=pred.dtype,
        )
        padding = self.window_size // 2

        mu_pred     = F.conv2d(pred, window, padding=padding, groups=channel)
        mu_target   = F.conv2d(target, window, padding=padding, groups=channel)

        mu_pred_sq      = mu_pred ** 2
        mu_target_sq    = mu_target ** 2
        mu_pred_target  = mu_pred * mu_target

        sigma_pred_sq = (
            F.conv2d(pred * pred, window, padding=padding, groups=channel)
            - mu_pred_sq
        )
        sigma_target_sq = (
            F.conv2d(target * target, window, padding=padding, groups=channel)
            - mu_target_sq
        )
        sigma_pred_target = (
            F.conv2d(pred * target, window, padding=padding, groups=channel)
            - mu_pred_target
        )

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (
            (mu_pred_sq + mu_target_sq + C1)
            * (sigma_pred_sq + sigma_target_sq + C2)
            + self.eps
        )
        ssim_map = numerator / denominator

        return ssim_map

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        ssim_map = self._compute_ssim_map(pred, target)
        ssim = ssim_map.reshape(ssim_map.shape[0], -1).mean(dim=1)

        return _reduce(ssim, self.reduction)

class SSIMLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.compute_ssim = SSIM(reduction='none')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        ssim = self.compute_ssim(pred, target)
        loss = 1.0 - ssim

        return _reduce(loss, self.reduction)

class PSNR(nn.Module):
    def __init__(self, max_val: float = 1.0, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        
        self.max_val = max_val
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred    = pred.contiguous().reshape(pred.shape[0], -1)
        target  = target.contiguous().reshape(target.shape[0], -1)

        mse     = torch.mean((pred - target) ** 2, dim=1)
        max_val = pred.new_tensor(self.max_val)

        psnr    = 10.0 * torch.log10((max_val ** 2) / (mse + self.eps))
        
        if      self.reduction == 'mean': return psnr.mean()
        elif    self.reduction == 'sum':  return psnr.sum()
        elif    self.reduction == 'none': return psnr
        else:   raise ValueError(f'Invalid reduction: {self.reduction}')

# ---- Edge Loss ---- #
def gaussian_kernel(kernel_size=3, sigma=0.8, device='cpu', dtype=torch.float32):
    ax = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel2d = kernel[:, None] @ kernel[None, :]
    return kernel2d.view(1, 1, kernel_size, kernel_size)


def gaussian_blur(x, kernel_size=3, sigma=0.8):
    kernel = gaussian_kernel(kernel_size, sigma, device=x.device, dtype=x.dtype)
    return F.conv2d(x, kernel.expand(x.size(1), 1, kernel_size, kernel_size),
                    padding=kernel_size // 2, groups=x.size(1))


def edge_map(x, mode='sobel', blur=True, sigma=0.8):
    if blur:
        x = gaussian_blur(x, kernel_size=5, sigma=sigma)

    if mode == 'sobel':
        gx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        gy = torch.tensor([[-1., -2., -1.],
                           [0., 0., 0.],
                           [1., 2., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        grad_x = F.conv2d(x, gx, padding=1)
        grad_y = F.conv2d(x, gy, padding=1)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge

    lap = torch.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    edge = F.conv2d(x, lap, padding=1)
    return torch.abs(edge)

class EdgeLoss(nn.Module):
    def __init__(self, mode='sobel', blur=True, sigma=0.8):
        super(EdgeLoss, self).__init__()
        self.mode = mode
        self.blur = blur
        self.sigma = sigma

    def forward(self, x, y):
        edge_x = edge_map(x, self.mode, blur=self.blur, sigma=self.sigma)
        edge_y = edge_map(y, self.mode, blur=self.blur, sigma=self.sigma)
        return F.l1_loss(edge_x, edge_y)