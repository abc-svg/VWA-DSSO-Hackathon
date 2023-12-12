import torch
from TransCNN import HARTransformer
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv3d(1, channels, kernel_size=(3, 3, 3), padding=("same"))
        self.conv2 = torch.nn.Conv3d(channels, 64, kernel_size=(3, 3, 3), padding=("same"))
        self.batchnorm1 = nn.BatchNorm3d(channels)
        self.batchnorm2 = nn.BatchNorm3d(64)

    def forward(self, x):
        y = F.leaky_relu(self.batchnorm1(self.conv1(x)))
        y = self.batchnorm2(self.conv2(y))
        return F.leaky_relu(x + y)


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class HARTrans(nn.Module):

  def __init__(self, nClasses: int, lstmUnit: str = ""):
    super().__init__()

    dropoutRatio1 = 0.5
    dropoutRatio2 = 0.5

    convChannel = 32
    self.conv = nn.Sequential(
        nn.Conv3d(1, convChannel, kernel_size=3, padding="same", bias=False),
        nn.BatchNorm3d(convChannel), nn.LeakyReLU(0.2),
        nn.AvgPool3d(kernel_size=2, stride=2), nn.Dropout(dropoutRatio1))
        nn.Conv3d(convChannel, 64, kernel_size=3, padding="same", bias=False),
        nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
        nn.AvgPool3d(kernel_size=2, stride=2), nn.Dropout(dropoutRatio1),
        nn.Dropout(dropoutRatio1))

    nLstms: int = 128
    if lstmUnit == "gru":
      LSTM = nn.GRU
    else:
      LSTM = nn.LSTM

    nLstmFeature = 896
    self.lstm = nn.Sequential(
        nn.Flatten(start_dim=2),
        LSTM(
            input_size=nLstmFeature,
            hidden_size=nLstms,
            num_layers=1,
            batch_first=True))

    self.out = nn.Sequential(
        nn.Dropout(dropoutRatio2), nn.Flatten(),
        nn.Linear(64000, 16), nn.LogSoftmax(dim=1))

  def forward(self, x: torch.Tensor):
    d1 = x.size(dim=0)
    x = x.view(d1, 1, 1000, 30, 9)
    conv = self.conv(x)
    # Batch x Channel x Frame x Tone x RX -> Batch x Frame x Channel x Tone x RX
    conv = conv.permute((0, 2, 1, 3, 4))
    lstm, _ = self.lstm(conv)
    y = self.out(lstm)
    return y

