import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfModulationBatchNorm(nn.Module):
    def __init__(self, in_ch_dim, cond_dim):
        super(SelfModulationBatchNorm, self).__init__()
        self.in_ch_dim = in_ch_dim
        self.cond_dim = cond_dim
        self.gamma = nn.Linear(cond_dim, in_ch_dim)
        self.beta = nn.Linear(cond_dim, in_ch_dim)
        self.bn = nn.BatchNorm2d(in_ch_dim)

    def forward(self, x, cond):
        gamma = self.gamma(cond).view(-1, self.in_ch_dim, 1, 1)
        beta = self.beta(cond).view(-1, self.in_ch_dim, 1, 1)
        output = self.bn(x)
        output = output * (gamma + 1.0) + beta
        return output


class ResBlockUp(nn.Module):
    def __init__(self, in_ch_dim, out_ch_dim, size, cond_dim):
        super().__init__()
        self.in_ch_dim = in_ch_dim
        self.out_ch_dim = out_ch_dim
        self.size = size
        self.bn1 = SelfModulationBatchNorm(in_ch_dim, cond_dim)
        self.main_flow_1 = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(size*2),
            nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=(3, 3), padding=1, stride=(1, 1))
        )
        self.bn2 = SelfModulationBatchNorm(out_ch_dim, cond_dim)
        self.main_flow_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_ch_dim, out_ch_dim, kernel_size=(3, 3), padding=1, stride=(1, 1))
        )
        self.res_flow = nn.Sequential(
            nn.Upsample(size*2),
            nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x, cond):
        output = self.bn1(x, cond)
        output = self.main_flow_1(output)
        output = self.bn2(output, cond)
        output = self.main_flow_2(output)
        res = self.res_flow(x)
        output += res
        return output


class AttnSelfModulationBatchNorm(nn.Module):
    def __init__(self, in_ch_dim, cond_dim):
        super(AttnSelfModulationBatchNorm, self).__init__()
        self.in_ch_dim = in_ch_dim
        self.cond_dim = cond_dim
        self.gamma = nn.Conv2d(cond_dim, in_ch_dim, kernel_size=(1, 1), stride=(1, 1))
        self.beta = nn.Conv2d(cond_dim, in_ch_dim, kernel_size=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(in_ch_dim)

    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        output = self.bn(x)
        output = output * (gamma + 1.0) + beta
        return output


class AttnResBlockUp(nn.Module):
    def __init__(self, in_ch_dim, out_ch_dim, size, cond_dim):
        super().__init__()
        self.in_ch_dim = in_ch_dim
        self.out_ch_dim = out_ch_dim
        self.size = size
        self.bn1 = AttnSelfModulationBatchNorm(in_ch_dim, cond_dim)
        self.main_flow_1 = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(size*2),
            nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=(3, 3), padding=1, stride=(1, 1))
        )
        self.bn2 = AttnSelfModulationBatchNorm(out_ch_dim, cond_dim)
        self.main_flow_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_ch_dim, out_ch_dim, kernel_size=(3, 3), padding=1, stride=(1, 1))
        )
        self.res_flow = nn.Sequential(
            nn.Upsample(size*2),
            nn.Conv2d(in_ch_dim, out_ch_dim, kernel_size=(1, 1), stride=(1, 1))
        )
        self.up = nn.Upsample(size*2)

    def forward(self, x, cond0):
        cond1 = self.up(cond0)
        output = self.bn1(x, cond0)
        output = self.main_flow_1(output)
        output = self.bn2(output, cond1)
        output = self.main_flow_2(output)
        res = self.res_flow(x)
        output += res
        return output, cond1


def get_contexts(word, img, word_len=None, gamma=1):
    spatial_region_size = img.size(-1)
    total_region_size = spatial_region_size * spatial_region_size
    total_word_len = word.size(1)

    word = l2_normalize(word, -1)
    img = l2_normalize(img.permute(0, 2, 3, 1).contiguous().view(-1, 256, 768), -1)

    attn_matrix = torch.matmul(img, word.permute(0, 2, 1).contiguous())
    attn_matrix = attn_matrix * gamma

    if word_len is not None:
        mask = torch.arange(total_word_len).cuda()[None, :] >= word_len.cuda()[:, None]
        mask = mask * -1e9
        mask = mask.view(-1, 1, total_word_len)
        mask = torch.tile(mask, [1, total_region_size, 1])
        attn_matrix = attn_matrix + mask
    attn = F.softmax(attn_matrix, -1)
    region_context = torch.matmul(attn, word)
    return region_context.view(-1, 16, 16, 768)


def l2_normalize(x, axis=None, epsilon=1e-12):
    square = torch.square(x)
    square_sum = torch.sum(square, dim=axis, keepdims=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon)))
    return torch.multiply(x, x_inv_norm)


class Generator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(256,4*4*16)
        self.res_up1 = ResBlockUp(16, 16, 4, 256)
        self.res_up2 = ResBlockUp(16, 8, 8, 256)
        self.conv1 = nn.Conv2d(8, 768, kernel_size=(1, 1), stride=(1, 1))
        self.attn_res_up1 = AttnResBlockUp(8, 8, 16, 1024)
        self.attn_res_up2 = AttnResBlockUp(8, 4, 32, 1024)
        self.attn_res_up3 = AttnResBlockUp(4, 2, 64, 1024)
        self.attn_res_up4 = AttnResBlockUp(2, 1, 128, 1024)
        self.attn_batch_norm = AttnSelfModulationBatchNorm(1, 1024)
        self.conv2 = nn.Conv2d(1, 3, kernel_size=(3, 3), padding=1, stride=(1, 1))

    def forward(self, noise, sent, word, max_len):
        sent = self.linear1(sent)
        cond = torch.cat([sent, noise], dim=1)
        x = self.linear2(cond).view(-1, 16, 4, 4)
        x = self.res_up1(x, cond)
        x = self.res_up2(x, cond)
        x_cond = self.conv1(x)
        context = get_contexts(word, x_cond, max_len)
        attn_cond = torch.tile(cond.view(-1, 1, 1, 256), [1, 16, 16, 1])
        attn_cond = torch.cat([context, attn_cond], dim=-1).permute(0, 3, 1, 2).contiguous()
        x, attn_cond1 = self.attn_res_up1(x, attn_cond)
        x, attn_cond2 = self.attn_res_up2(x, attn_cond1)
        x, attn_cond3 = self.attn_res_up3(x, attn_cond2)
        x, attn_cond4 = self.attn_res_up4(x, attn_cond3)
        x = self.attn_batch_norm(x, attn_cond4)
        x = torch.relu(x)
        x = torch.tanh(self.conv2(x))
        x = (x + 1.0) / 2.0
        return x

