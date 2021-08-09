from pytorch_metric_learning import losses
import torch
from torch.autograd import Variable


def get_contrastive_loss(pair1, pair2, loss_func):
    embeddings = torch.cat([pair1, pair2], dim=1)
    indices = torch.zeros(pair1.size())
    labels = torch.cat((indices, indices))
    loss = loss_func(embeddings, labels)
    loss = Variable(loss, requires_grad=True)
    return loss


class ContrastiveLoss():
    def __init__(self):
        self.similarity_func = losses.NTXentLoss().get_distance()

    def contrastive_loss(self, pair1, pair2, temperature=0.1):
        # This is used for sent_contrastive loss or img_contrastive_loss
        mat_similarity = torch.exp(self.similarity_func(pair1, pair2)/temperature)
        sum_similarity = torch.sum(mat_similarity, dim=-1).view(-1, 1)
        loss = -torch.log(torch.div(mat_similarity, sum_similarity))
        losses = []
        for idx in range(loss.size(0)):
            element_loss = loss[idx][idx]
            losses.append(element_loss)
        losses = torch.stack(losses, 0)
        # print(losses.shape)
        n = loss.size(0)
        sum_losses = torch.sum(losses)
        output = sum_losses / n
        return output
    #
    # def image_contrastive_loss(self, pair1, pair2, temperature = 0.1):
    #
    #
    # def word_contrastive_loss(self, pair1, pair2, temperature=0.1):
    #
    #     return output

#
# test_img = torch.randn(64, 768)
# test_sent = torch.randn(64, 768)
#
# # test_img = test_img.permute(0, 2, 3, 1).contiguous().view(-1, 256, 768)
# loss_func = SentContrastiveLoss()
# loss_func2 = SentContrastiveLoss()
# result = loss_func.contrastive_loss(test_img, test_sent)
# result2 = loss_func2.contrastive_loss(test_img, test_sent)
# print(result)
# print(result2)


def hinge_loss_d(out_d_real, out_d_fake, device):
    real_loss = torch.relu(1.0 - out_d_real)
    fake_loss = torch.relu(1.0 + out_d_fake)
    loss = torch.mean(real_loss + fake_loss)
    # real_loss = -torch.mean(torch.minimum(torch.zeros(out_d_real.shape).to(device), -torch.ones(out_d_real.shape).to(device) + out_d_real))
    # fake_loss = -torch.mean(torch.minimum(torch.zeros(64, 1).to(device), -torch.ones(out_d_real.shape).to(device) - out_d_fake))
    # loss = real_loss + fake_loss
    return loss


def hinge_loss_g(out_d_fake):
    loss = -torch.mean(out_d_fake)
    return loss

# def hinge_loss(real_logit: jnp.ndarray, fake_logit: jnp.ndarray) -> jnp.ndarray:
#   generator_loss = -jnp.mean(fake_logit)
#   real_loss = jax.nn.relu(1.0 - real_logit)
#   fake_loss = jax.nn.relu(1.0 + fake_logit)
#   discriminator_loss = jnp.mean(real_loss + fake_loss)
#   return discriminator_loss, generator_loss



# TBD: contrastive loss => soft attention 적용한 버전.
