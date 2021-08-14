import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


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


class Attention():
    def __init__(self):
        self.cosine_similarity = losses.NTXentLoss().get_distance()

    def get_contexts(self, word, img, rho = 1):
        sims = []
        for w, i in zip(word, img.permute(0, 2, 3, 1).contiguous().view(-1, 256, 768)):
            sim = self.cosine_similarity(i, w)
            sims.append(sim)
        result = torch.stack(sims, 0)
        exp_x = torch.exp(result * rho)
        sum_exp = torch.sum(exp_x, dim=-1).view(-1, 256, 1)
        attentions = torch.div(exp_x, sum_exp)
        return (torch.matmul(attentions, word)).view(-1, 16, 16, 768)


# class SelfModulation(nn.Module):
#     def __init__(self, in_ch_dim, kernel_size, cond_dim):
#         super().__init__()
#         self.in_ch_dim = in_ch_dim
#         self.kernel_size = kernel_size
#         self.linear1 = nn.Linear(cond_dim, in_ch_dim*kernel_size*kernel_size)
#         self.linear2 = nn.Linear(cond_dim, in_ch_dim*kernel_size*kernel_size)
#         self.bn1 = nn.BatchNorm2d(in_ch_dim)
#
#     def forward(self, h, cond):
#         x = self.linear1(cond).view(-1, self.in_ch_dim, self.kernel_size, self.kernel_size) * (self.bn1(h))
#         x += self.linear2(cond).view(-1, self.in_ch_dim, self.kernel_size, self.kernel_size)
#         return x


class Generator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.attention = Attention()
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

    def forward(self, noise, sent, word):
        sent = self.linear1(sent)
        cond = torch.cat([noise, sent], dim=1)
        x = self.linear2(cond).view(-1, 16, 4, 4)
        x = self.res_up1(x, cond)
        x = self.res_up2(x, cond)
        x_cond = self.conv1(x) # (b, 768, 16, 16)
        context = self.attention.get_contexts(word, x_cond)
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
