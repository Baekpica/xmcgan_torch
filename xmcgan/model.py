import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(256,4*4*16)
        self.self_res_up1 = nn.x
        self.self_res_up2 =
        self.linear3 = nn.Linear(8, 768)

    def forward(self, noises, sents):


        return x


class Discriminator(nn.Module):
    def __init__(self, parmas):
        super().__init__()

    def forward(self, images, sents):

        return x


