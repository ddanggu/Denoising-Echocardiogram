import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infer_common import load_state_dict, run_inference, strip_prefix

from networks.gan import CycleGAN


def get_opt():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="./results_cyclegan")
    p.add_argument("--checkpoint", type=str, default="./checkpoints_cyclegan/best_net_SS_G_XtoY.pth")
    p.add_argument("--scale", type=int, default=128)
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gt", type=str, default=None)
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--roi_signal", type=str, default=None)
    p.add_argument("--roi_noise", type=str, default=None)
    p.add_argument("--direction", type=str, default="XtoY", choices=["XtoY", "YtoX"])
    p.add_argument("--in_ch", type=int, default=1)
    p.add_argument("--out_ch", type=int, default=1)
    p.add_argument("--G_init_ch", type=int, default=64)
    p.add_argument("--D_init_ch", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out_act", type=str, default=None)
    p.add_argument("--norm", type=str, default="batch")
    return p.parse_args()


def load_model(opt, device):
    model = CycleGAN(opt.in_ch, opt.out_ch, opt.G_init_ch, opt.D_init_ch, opt.depth, opt.out_act, opt.norm)
    state, _ = load_state_dict(opt.checkpoint, device)
    gen = model.G_XtoY if opt.direction == "XtoY" else model.G_YtoX
    prefix = "G_XtoY." if opt.direction == "XtoY" else "G_YtoX."
    if any(k.startswith("G_XtoY.") or k.startswith("G_YtoX.") for k in state):
        model.load_state_dict(state, strict=False)
        return gen
    prefixed = strip_prefix(state, prefix)
    if prefixed:
        gen.load_state_dict(prefixed)
    else:
        gen.load_state_dict(state)
    return gen


def forward_model(model, batch):
    return model(batch)


def main():
    opt = get_opt()
    device = torch.device("cpu") if opt.gpu < 0 else torch.device(f"cuda:{opt.gpu}")
    model = load_model(opt, device)
    run_inference(opt, model, forward_model)


if __name__ == "__main__":
    main()
