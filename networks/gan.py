import torch
import torch.nn as nn
import torch.nn.functional as F

def build_channel_schedule(init_ch: int, depth: int):
    if depth < 1:
        raise ValueError(f'depth must be >= 1, got {depth}')
    
    return [init_ch * (2**i) for i in range(depth)]

def build_decoder_channels(channels: list[int], bottleneck_channel: int):
    decoder_in_channels = [bottleneck_channel, *reversed(channels[1:])]
    skip_channels = list(reversed(channels))

    return list(zip(decoder_in_channels, skip_channels))

def build_norm(norm: str | None, num_ch: int):
    if norm is None: return None

    norms = {
        "batch": lambda: nn.BatchNorm2d(num_ch),
        "instance": lambda: nn.InstanceNorm2d(num_ch, affine=False),
    }

    key = norm.lower()
    if key not in norms:
        raise ValueError(f"Unsupported norm: {norm}")

    return norms[key]()

def build_out_activation(name: str | None):
    if name is None: return None

    target = name.lower()
    for attr_name in dir(nn):
        if attr_name.lower() == target:
            act_cls = getattr(nn, attr_name); break
        
    else: raise ValueError(f'Unsupported output activation: {name}')

    if not isinstance(act_cls, type) or not issubclass(act_cls, nn.Module):
        raise ValueError(f"Unsupported output activation: {name}")

    return act_cls()

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.in_ch   = in_ch
        self.skip_ch = skip_ch
        self.out_ch  = out_ch

        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, out_act: str | None=None):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=1)]

        o_act = build_out_activation(out_act)
        if o_act is not None:
            layers.append(o_act)
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.block(x)
    
class DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int=2, norm: str | None='batch'):
        super().__init__()
        use_norm = norm is not None

        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm)]

        norm_layer = build_norm(norm, out_ch)
        if norm_layer is not None:
            layers.append(norm_layer)
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1, init_ch: int=64, depth: int=4, out_act: str | None=None):
        super().__init__()
        # depth means the number of encoder/decoder blocks, excluding bottleneck and output layer.

        # Channel Schedule
        channels = build_channel_schedule(init_ch, depth)

        # Encoder
        self.encoders = nn.ModuleList()
        self.encoders.append(DoubleConv(in_ch, channels[0]))

        for i in range(1, depth):
            self.encoders.append(
                DownBlock(channels[i - 1], channels[i])
            )
        
        # Bottleneck
        self.bottleneck = DownBlock(channels[-1], channels[-1] * 2)

        # Decoder
        self.decoders = nn.ModuleList()
        for decoder_in_ch, skip_ch in build_decoder_channels(channels, self.bottleneck.out_ch):
            self.decoders.append(
                UpBlock(decoder_in_ch, skip_ch, skip_ch)
            )

        self.outc = OutConv(channels[0], out_ch, out_act)

    def forward(self, x: torch.Tensor):
        skips = []

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.outc(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1, init_ch: int=64, depth: int=4, norm: str | None='batch'):
        super().__init__()
        # depth means the number of encoder/decoder blocks, excluding bottleneck and output layer.

        # Channel Schedule
        channels = build_channel_schedule(init_ch, depth)

        # Network
        layers = []

        layers.append(DiscBlock(in_ch, channels[0], stride=2, norm=None))
        for i in range(1, depth):
            layers.append(
                DiscBlock(channels[i - 1], channels[i], stride=2, norm=norm)
            )
        
        layers.append(nn.Conv2d(channels[-1], out_ch, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
class GAN(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1, G_init_ch: int=64, D_init_ch: int=64, depth: int=4, 
                 out_act: str | None=None, norm: str | None='batch'):
        super().__init__()
        self.G = Generator(in_ch=in_ch, out_ch=out_ch, init_ch=G_init_ch, depth=depth, out_act=out_act)
        self.D = Discriminator(in_ch=in_ch, out_ch=out_ch, init_ch=D_init_ch, depth=depth, norm=norm)

class CycleGAN(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1, G_init_ch: int=64, D_init_ch: int=64, depth: int=4, 
                 out_act: str | None=None, norm: str | None='batch'):
        super().__init__()
        self.G_XtoY = Generator(in_ch=in_ch, out_ch=out_ch, init_ch=G_init_ch, depth=depth, out_act=out_act)
        self.G_YtoX = Generator(in_ch=in_ch, out_ch=out_ch, init_ch=G_init_ch, depth=depth, out_act=out_act)
        self.D_X    = Discriminator(in_ch=in_ch, out_ch=out_ch, init_ch=D_init_ch, depth=depth, norm=norm)
        self.D_Y    = Discriminator(in_ch=in_ch, out_ch=out_ch, init_ch=D_init_ch, depth=depth, norm=norm)

class SuCycleGAN(CycleGAN):
    def __init__(self, in_ch: int=1, out_ch: int=1, G_init_ch: int=64, D_init_ch: int=64, depth: int=4, 
                 out_act: str | None=None, norm: str | None='batch'):
        super().__init__(in_ch=in_ch, out_ch=out_ch, G_init_ch=G_init_ch, D_init_ch=D_init_ch, depth=depth,
                         out_act=out_act, norm=norm)