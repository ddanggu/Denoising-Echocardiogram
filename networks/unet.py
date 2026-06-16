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

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.Mish(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Mish(inplace=True),
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
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1, init_ch: int=64, depth: int=3):
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

        self.outc = OutConv(channels[0], out_ch) # Output layer

    def forward(self, x: torch.Tensor):
        skips = []

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.outc(x)