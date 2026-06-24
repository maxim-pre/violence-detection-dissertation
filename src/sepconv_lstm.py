import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SepConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.gates = SeparableConv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)

        gates = self.gates(combined)

        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # candidate memory

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class SepConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell = SepConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

    def forward(self, x):
        """
        x shape: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        c = torch.zeros_like(h)

        outputs = []

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs