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
            groups=in_channels, 
            bias=False, 
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, 
            bias=False,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SepConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.x_input = SeparableConv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.h_input = SeparableConv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.b_input = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.x_forget = SeparableConv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.h_forget = SeparableConv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.b_forget = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.x_candidate = SeparableConv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.h_candidate = SeparableConv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.b_candidate = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.x_output = SeparableConv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.h_output = SeparableConv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.b_output = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))


    def forward(self, x, h_prev, c_prev):

        i = torch.sigmoid(self.x_input(x) + self.h_input(h_prev) + self.b_input)
        f = torch.sigmoid(self.x_forget(x) + self.h_forget(h_prev) + self.b_forget)
        c_tilde = torch.tanh(self.x_candidate(x) + self.h_candidate(h_prev) + self.b_candidate)
        o = torch.sigmoid(self.x_output(x) + self.h_output(h_prev) + self.b_output)

        c = f * c_prev + i * c_tilde
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
        # input shape: [B, 32, 96, 7, 7]
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        c = torch.zeros_like(h)

        outputs = []

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        # shape: [B, 32, 64, 7, 7]

        return outputs