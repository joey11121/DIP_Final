

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MCNNPlus(nn.Module):
    """
    改良版 Multi-column CNN (MCNN++)

    - 每個 branch: Conv + BN + ReLU
    - 使用較小 kernel + (可選) dilation
    - fuse 部分加深成 3 層 conv
    - 加入 SE channel attention
    - 輸出 density map 可上採樣
    """
    def __init__(self, upsample_factor=1, load_weights=False):
        super(MCNNPlus, self).__init__()

        # 分支1：大感受野（用 dilation，而不是大 kernel）
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # 分支2：中等感受野
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(20, 40, 3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(40, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.Conv2d(20, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
        )

        # 分支3：小感受野（細節）
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
        )

        in_channels = 8 + 10 + 12  # =30

        # channel attention
        self.se = SEBlock(in_channels)

        # fuse 部分加深
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1)
        )

        self.upsample_factor = upsample_factor

        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x1, x2, x3), dim=1)  # [B,30,H/4,W/4]
        x_cat = self.se(x_cat)

        dmap = self.fuse(x_cat)  # [B,1,H/4,W/4]

        if self.upsample_factor != 1:
            dmap = F.interpolate(
                dmap,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
        return dmap

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    img = torch.rand((1, 3, 800, 1200))
    net = MCNNPlus(upsample_factor=1)
    out = net(img)
    print("Output:", out.shape)
