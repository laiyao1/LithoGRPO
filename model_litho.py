"""Original MiniUnet architecture and checkpoint parameter names."""

import torch
from torch import nn


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, downsample=False):
        super(DownLayer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2)
        self.in_channels = in_channels

    def forward(self, x, temb):
        res = x
        x += self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        if self.downsample:
            x = self.pool(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16, upsample=False):
        super(UpLayer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, temb):
        if self.upsample:
            x = self.upsample(x)
        res = x
        x += self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        return x


class MiddleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x
        fc_out = self.fc(temb)
        x = x + fc_out[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res
        return x


class MiniUnet(nn.Module):
    def __init__(self, base_channels=32, time_emb_dim=None):
        super(MiniUnet, self).__init__()
        if time_emb_dim is None:
            self.time_emb_dim = base_channels
        self.base_channels = base_channels
        self.conv_in = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.ModuleList(
            [
                DownLayer(
                    base_channels, base_channels * 2, time_emb_dim=self.time_emb_dim
                ),
                DownLayer(
                    base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim
                ),
            ]
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.down2 = nn.ModuleList(
            [
                DownLayer(
                    base_channels * 2, base_channels * 4, time_emb_dim=self.time_emb_dim
                ),
                DownLayer(
                    base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim
                ),
            ]
        )
        self.maxpool2 = nn.MaxPool2d(2)
        self.down3 = nn.ModuleList(
            [
                DownLayer(
                    base_channels * 4, base_channels * 8, time_emb_dim=self.time_emb_dim
                ),
                DownLayer(
                    base_channels * 8, base_channels * 8, time_emb_dim=self.time_emb_dim
                ),
            ]
        )
        self.maxpool3 = nn.MaxPool2d(2)
        self.down4 = nn.ModuleList(
            [
                DownLayer(
                    base_channels * 8,
                    base_channels * 16,
                    time_emb_dim=self.time_emb_dim,
                ),
                DownLayer(
                    base_channels * 16,
                    base_channels * 16,
                    time_emb_dim=self.time_emb_dim,
                ),
            ]
        )
        self.maxpool4 = nn.MaxPool2d(2)
        self.middle = MiddleLayer(
            base_channels * 16, base_channels * 16, time_emb_dim=self.time_emb_dim
        )
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList(
            [
                UpLayer(
                    base_channels * 32,
                    base_channels * 8,
                    time_emb_dim=self.time_emb_dim,
                ),
                UpLayer(
                    base_channels * 8, base_channels * 8, time_emb_dim=self.time_emb_dim
                ),
            ]
        )
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList(
            [
                UpLayer(
                    base_channels * 16,
                    base_channels * 4,
                    time_emb_dim=self.time_emb_dim,
                ),
                UpLayer(
                    base_channels * 4, base_channels * 4, time_emb_dim=self.time_emb_dim
                ),
            ]
        )
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.up3 = nn.ModuleList(
            [
                UpLayer(
                    base_channels * 8, base_channels * 2, time_emb_dim=self.time_emb_dim
                ),
                UpLayer(
                    base_channels * 2, base_channels * 2, time_emb_dim=self.time_emb_dim
                ),
            ]
        )
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.up4 = nn.ModuleList(
            [
                UpLayer(
                    base_channels * 4, base_channels, time_emb_dim=self.time_emb_dim
                ),
                UpLayer(base_channels, base_channels, time_emb_dim=self.time_emb_dim),
            ]
        )
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_emb(self, t, dim):
        t = t * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def label_emb(self, y, dim):
        y = y * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
        sin_emb = torch.sin(y[:, None] / freqs)
        cos_emb = torch.cos(y[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x, t, y=None):
        x = self.conv_in(x)
        temb = self.time_emb(t, self.base_channels)
        if y is not None:
            if len(y.shape) == 1:
                yemb = self.label_emb(y, self.base_channels)
                yemb[y == -1] = 0.0
                temb += yemb
        skip_connections = []
        for layer in self.down1:
            x = layer(x, temb)
        skip_connections.append(x)
        x = self.maxpool1(x)
        for layer in self.down2:
            x = layer(x, temb)
        skip_connections.append(x)
        x = self.maxpool2(x)
        for layer in self.down3:
            x = layer(x, temb)
        skip_connections.append(x)
        x = self.maxpool3(x)
        for layer in self.down4:
            x = layer(x, temb)
        skip_connections.append(x)
        x = self.maxpool4(x)
        x = self.middle(x, temb)
        x = self.upsample1(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for layer in self.up1:
            x = layer(x, temb)
        x = self.upsample2(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for layer in self.up2:
            x = layer(x, temb)
        x = self.upsample3(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for layer in self.up3:
            x = layer(x, temb)
        x = self.upsample4(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)
        for layer in self.up4:
            x = layer(x, temb)
        x = self.conv_out(x)
        return x
