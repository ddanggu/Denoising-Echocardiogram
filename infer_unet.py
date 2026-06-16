import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infer_common import load_state_dict, run_inference, strip_prefix

from networks.unet import UNet


def get_opt():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="./results_unet")
    p.add_argument("--checkpoint", type=str, default="./checkpoints_unet/best_net_SS_G.pth")
    p.add_argument("--scale", type=int, default=128)
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gt", type=str, default=None)
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--roi_signal", type=str, default=None)
    p.add_argument("--roi_noise", type=str, default=None)
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--out_ch", type=int, default=1)
    p.add_argument("--init_ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    return p.parse_args()


def load_model(opt, device):
    model = UNet(opt.in_ch, opt.out_ch, opt.init_ch, opt.depth)
    state, _ = load_state_dict(opt.checkpoint, device)
    model.load_state_dict(state)
    return model


def forward_model(model, batch):
    return model(batch)


def main():
    opt = get_opt()
    device = torch.device("cpu") if opt.gpu < 0 else torch.device(f"cuda:{opt.gpu}")
    model = load_model(opt, device)
    run_inference(opt, model, forward_model)


if __name__ == "__main__":
    main()
