import torch
import torch.nn as nn
import torch.nn.utils as utils


class ResBlockDown(nn.Module):
    def __init__(self, in_ch_dim, out_ch_dim, size, down = True):
        super().__init__()
        self.down = down
        self.in_ch_dim = in_ch_dim
        self.out_ch_dim = out_ch_dim
        self.size = size
        self.main_flow = nn.Sequential(
            nn.ReLU(),
            utils.spectral_norm(nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            nn.ReLU(),
            utils.spectral_norm(nn.Conv2d(out_ch_dim, out_ch_dim, kernel_size=(3, 3), stride=(1, 1), padding=1))
        )
        self.down1 = nn.AvgPool2d(2)
        self.res_flow = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=(1, 1), stride=(1, 1)))
        )
        self.down2 = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res_flow(x)
        output = self.main_flow(x)
        if self.down:
            output = self.down1(output)
            res = self.down2(res)
        output += res
        return output


class OptimizedBlock(nn.Module):
    def __init__(self, in_ch_dim, out_ch_dim):
        super(OptimizedBlock, self).__init__()
        self.main = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_ch_dim, out_ch_dim,
                                          kernel_size=(3, 3),
                                          padding=1,
                                          stride=(1, 1))),
            nn.ReLU(),
            utils.spectral_norm(nn.Conv2d(out_ch_dim, out_ch_dim,
                                          kernel_size=(3, 3),
                                          padding=1,
                                          stride=(1, 1))),
            nn.AvgPool2d(2)
        )
        self.res = nn.Sequential(
            nn.AvgPool2d(2),
            utils.spectral_norm(nn.Conv2d(in_ch_dim, out_ch_dim,
                                          kernel_size=(1, 1),
                                          stride=(1, 1)))
        )

    def forward(self, x):
        output = self.main(x)
        res = self.res(x)
        output += res
        return output


class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.main1 = nn.Sequential(
            OptimizedBlock(3, 1),
            ResBlockDown(1, 2, 128),
            ResBlockDown(2, 4, 64),
            ResBlockDown(4, 8, 32)
        )
        self.conv1 = utils.spectral_norm(nn.Conv2d(8, 768,
                                                   kernel_size=(1, 1),
                                                   stride=(1, 1)))
        self.main2 = nn.Sequential(
            ResBlockDown(8, 8, 16),
            ResBlockDown(8, 16, 8),
            ResBlockDown(16, 16, 4, False),
            nn.AvgPool2d(4)
        )
        self.conv2 = utils.spectral_norm(nn.Conv2d(16, 768,
                                                   kernel_size=(1, 1),
                                                   stride=(1, 1)))
        self.linear1 = utils.spectral_norm(nn.Linear(16, 1))
        self.linear2 = utils.spectral_norm(nn.Linear(768, 16))

    def forward(self, x, sent):
        x = self.main1(x)
        img_region_feat = self.conv1(x)
        x = self.main2(x)
        img_feat = self.conv2(x)
        x = torch.relu(x)
        x_pool = torch.sum(x, dim=(2, 3))
        output = self.linear1(x_pool)
        sent_prj = self.linear2(sent)
        output += torch.sum(x_pool*sent_prj, dim=1, keepdim=True)
        output = torch.sigmoid(output)
        return output, img_region_feat, img_feat

# test_model = Discriminator(64)
# test_input = torch.randn(64, 3, 256, 256)
# test_sent = torch.randn(64, 768)
# result, _, _ = test_model(test_input, test_sent)
# print(max(result))
# print(min(result))
