import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(256,4*4*16)
        self.self_res_up1 = nn.
        self.self_res_up2 =
        self.linear3 = nn.Linear(8, 768)



class XmcGan(nn.Module):
    def __init__(self):
        super(XmcGan, self).__init__()


    def generator(self, ):


    def discriminator(self):
