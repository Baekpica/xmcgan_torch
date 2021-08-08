import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


class ResBlockUp(nn.Module):
    def __init__(self, in_ch_dim, out_ch_dim, size, cond_dim):
        super().__init__()
        self.in_ch_dim = in_ch_dim
        self.out_ch_dim = out_ch_dim
        self.size = size
        flatten_dim_in = in_ch_dim * size * size
        flatten_dim_out = out_ch_dim * size*2 * size*2
        self.main_flow_1 = nn.Sequential(
            nn.BatchNorm2d(in_ch_dim),
            nn.ReLU(),
            nn.Upsample(size*2),
            nn.Conv2d(in_ch_dim, out_ch_dim,
                      kernel_size=3,
                      padding=1,
                      stride=1)
        )
        self.main_flow_2 = nn.Sequential(
            nn.BatchNorm2d(out_ch_dim),
            nn.ReLU(),
            nn.Conv2d(out_ch_dim, out_ch_dim,
                      kernel_size=3,
                      padding=1,
                      stride=1)
        )
        self.res_flow = nn.Sequential(
            nn.Upsample(size*2),
            nn.Conv2d(in_ch_dim, out_ch_dim,
                      kernel_size=1,
                      stride=1,
                      bias=False)
        )
        self.long_cond = nn.Linear(cond_dim, flatten_dim_in)   # '256' is constant (because this is global condition)
        self.short_cond = nn.Linear(cond_dim, flatten_dim_out)

    def forward(self, x, cond):
        output = self.main_flow_2(self.main_flow_1(x))
        res = self.res_flow(x)
        cond_long = torch.reshape(self.long_cond(cond), (-1, self.in_ch_dim, self.size, self.size))
        cond_long = self.main_flow_2(self.main_flow_1(cond_long))
        cond_short = torch.reshape(self.short_cond(cond), (-1, self.out_ch_dim, self.size*2, self.size*2))
        cond_short = self.main_flow_2(cond_short)
        output += (res + cond_long + cond_short)
        return output


class Attention():
    def __init__(self):
        self.cosine_similarity = losses.NTXentLoss().get_distance()

    def get_contexts(self, word, img, rho = 1):
        sims = []
        for w, i in zip(word, img.view(64, -1, 768)):
            sim = self.cosine_similarity(i, w)
            sims.append(sim)
        result = torch.stack(sims, 0)
        exp_x = torch.exp(result * rho)
        sum_exp = torch.sum(exp_x)
        attentions = exp_x / sum_exp
        return (torch.matmul(attentions, word)).view(64, 16, 16, -1)


class SelfModulation(nn.Module):
    def __init__(self, in_ch_dim, kernel_size, cond_dim):
        super().__init__()
        self.in_ch_dim = in_ch_dim
        self.kernel_size = kernel_size
        self.linear1 = nn.Linear(cond_dim, in_ch_dim*kernel_size*kernel_size)
        self.linear2 = nn.Linear(cond_dim, in_ch_dim*kernel_size*kernel_size)
        self.bn1 = nn.BatchNorm2d(in_ch_dim)

    def forward(self, h, cond):
        x = self.linear1(cond).view(-1, self.in_ch_dim, self.kernel_size, self.kernel_size) * (self.bn1(h))
        x += self.linear2(cond).view(-1, self.in_ch_dim, self.kernel_size, self.kernel_size)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(256,4*4*16)
        self.self_mod1 = SelfModulation(16, 4, 256)
        self.res_up1 = ResBlockUp(16, 16, 4, 256)
        self.self_mod2 = SelfModulation(16, 8, 256)
        self.res_up2 = ResBlockUp(16, 8, 8, 256)
        self.linear3 = nn.Linear(8, 768)
        self.linear4 = nn.Linear(256*768, 128)
        self.linear5 = nn.Linear(256+128, 16*16*8)
        self.self_mod3 = SelfModulation(8, 16, 384)
        self.attn_res_up1 = ResBlockUp(8, 8, 16, 384)
        self.self_mod4 = SelfModulation(8, 32, 384)
        self.attn_res_up2 = ResBlockUp(8, 4, 32, 384)
        self.self_mod5 = SelfModulation(4, 64, 384)
        self.attn_res_up3 = ResBlockUp(4, 2, 64, 384)
        self.self_mod6 = SelfModulation(2, 128, 384)
        self.attn_res_up4 = ResBlockUp(2, 1, 128, 384)
        self.self_mod7 = SelfModulation(1, 256, 384)
        self.conv1 = nn.Conv2d(1, 3,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=(1, 1))

    def forward(self, noise, sent, word):
        sent = self.linear1(sent)
        cond = torch.cat([noise, sent], dim=1)
        x = self.linear2(cond).view(64, 16, 4, 4)
        x = self.res_up1(self.self_mod1(x, cond), cond)
        x = self.res_up2(self.self_mod2(x, cond), cond).view(-1, 8)
        x = self.linear3(x).view(64, 768, 16, 16)
        context = self.attention.get_contexts(word, x).view(64, -1)
        x = self.linear4(context)
        attn_cond = torch.cat([cond, x], dim=1)
        x = self.linear5(attn_cond).view(64, 8, 16, 16)
        x = self.attn_res_up1(self.self_mod3(x, attn_cond), attn_cond)
        x = self.attn_res_up2(self.self_mod4(x, attn_cond), attn_cond)
        x = self.attn_res_up3(self.self_mod5(x, attn_cond), attn_cond)
        x = F.relu(self.attn_res_up4(self.self_mod6(x, attn_cond), attn_cond))
        x = F.tanh(self.conv1(self.self_mod7(x, attn_cond)))
        return x


# test_noise = torch.randn(64, 128)
# test_sent = torch.randn(64, 768)
# test_word = torch.randn(64, 16, 768)
# model = Generator()
# test_result = model(test_noise, test_sent, test_word)
# print(test_result.shape)
# print(model)
