import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from pytorch_metric_learning import losses


class ResBlockDown(nn.Module):
    def __init__(self, in_ch_dim, out_ch_dim, size):
        super().__init__()
        self.in_ch_dim = in_ch_dim
        self.out_ch_dim = out_ch_dim
        self.size = size
        self.main_flow = nn.Sequential(
            nn.ReLU(),
            utils.spectral_norm(nn.Conv2d(in_ch_dim, out_ch_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1)),
            nn.ReLU(),
            utils.spectral_norm(nn.Conv2d(out_ch_dim, out_ch_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1)),
            nn.AvgPool2d(2)
        )
        self.res_flow = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_ch_dim, out_ch_dim,
                      kernel_size=(1, 1),
                      stride=(1, 1))),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        res = self.res_flow(x)
        x = self.main_flow(x)
        x += res
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = utils.spectral_norm(nn.Conv2d(in_channels, out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = utils.spectral_norm(nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main1 = nn.Sequential(
            ResBlockDown(3, 1, 256),
            ResBlockDown(1, 2, 128),
            ResBlockDown(2, 4, 64),
            ResBlockDown(4, 8, 32)
        )
        self.main2 = nn.Sequential(
            ResBlockDown(8, 8, 16),
            ResBlockDown(8, 16, 8),
            ResidualBlock(16, 16),
            nn.AvgPool2d(4)
        )
        self.linear1 = utils.spectral_norm(nn.Linear(8, 768))
        self.linear2 = utils.spectral_norm(nn.Linear(768, 16))
        self.linear3 = utils.spectral_norm(nn.Linear(16, 1))

    def forward(self, x, sent):
        output = self.main1(x)
        img_feat = self.linear1(output.permute(0, 2, 3, 1).contiguous().view(-1, 8)).view(-1, 16, 16, 768).permute(0, 3, 1, 2)
        output = self.main2(output)
        sent_prj = self.linear2(sent).view(-1, 16, 1, 1)
        output = torch.tanh(self.linear3(torch.matmul(sent_prj, output).view(64, -1)))
        return output, img_feat

#
# test_input = torch.randn(64, 3, 256, 256)
# test_sent = torch.randn(64, 768)
# test_word = torch.randn(64, 16, 768)
# model = Discriminator()
# print(model(test_input, test_sent)[0].shape)


# test_input = torch.randn(64, 16, 1, 1)
# test_sent = torch.randn(64, 16)
# test_result = torch.matmul(test_sent.view(64, 16, 1, 1), test_input)
# print(test_result.shape)

