from pytorch_metric_learning import losses
import torch
from torch.autograd import Variable


def get_contrastive_loss(pair1, pair2, loss_func):
    embeddings = Variable(torch.cat([pair1, pair2], dim=0), requires_grad=True)
    indices = torch.arange(0, pair1.size(0))
    labels = torch.cat((indices, indices))
    loss = loss_func(embeddings, labels)
    return loss


def hinge_loss_d():
    return loss


def hinge_loss_g():
    return loss


# TBD: contrastive loss => soft attention 적용한 버전.
