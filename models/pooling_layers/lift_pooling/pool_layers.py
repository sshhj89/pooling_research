import torch
import torch.nn as nn

class LiftDownPool(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(LiftDownPool, self).__init__()
        padding = kernel_size // 2
        g1 = 1
        g2 = 1

        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=g1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, groups=g2),
            nn.Tanh()
        )

        self.updater = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=g1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, groups=g2),
            nn.Tanh()
        )

    def forward(self, x):
        # Horizontal split
        xe = x[:, :, :, ::2]
        xo = x[:, :, :, 1::2]

        d = xo - self.predictor(xe)
        s = xe + self.updater(xo)

        # Vertical split on s
        se = s[:, :, ::2, :]
        so = s[:, :, 1::2, :]

        LL = so - self.predictor(se)
        LH = se + self.updater(so)

        # Vertical split on s
        de = d[:, :, ::2, :]
        do = d[:, :, 1::2, :]

        HL = do - self.predictor(de)
        HH = de + self.updater(do)

        # Output is sum of all sub-bands from openreview comments
        out = LL + LH + HL + HH

        return out, d, s, xe, xo