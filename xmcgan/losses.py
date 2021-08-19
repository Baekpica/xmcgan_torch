from pytorch_metric_learning import losses
import torch

# def get_contrastive_loss(pair1, pair2, loss_func):
#     embeddings = torch.cat([pair1, pair2], dim=1)
#     indices = torch.zeros(pair1.size())
#     labels = torch.cat((indices, indices))
#     loss = loss_func(embeddings, labels)
#     loss = Variable(loss, requires_grad=True)
#     return loss


class ContrastiveLoss():
    def __init__(self):
        self.similarity_func = losses.NTXentLoss().get_distance()

    def contrastive_loss(self, pair1, pair2, temperature=0.1):
        # This is used for sent_contrastive loss or img_contrastive_loss
        pair1 = self.l2_normalize(pair1, axis=1)
        pair2 = self.l2_normalize(pair2, axis=1)
        mat_similarity = torch.exp(self.similarity_func(pair1, pair2)/temperature)
        sum_similarity = torch.sum(mat_similarity, dim=1).view(-1, 1)
        loss = -torch.log(torch.div(mat_similarity, sum_similarity))
        losses = []
        for idx in range(loss.size(0)):
            element_loss = loss[idx][idx]
            losses.append(element_loss)
        losses = torch.stack(losses, 0)
        n = loss.size(0)
        sum_losses = torch.sum(losses)
        output = sum_losses / n
        return output

    def get_attentional_score(self, word, region, rho=1, temperature=0.1):
        word = self.l2_normalize(word, -1)
        region_feat = region.permute(1, 2, 0).contiguous().view(-1, 768)
        region_feat = self.l2_normalize(region_feat, -1)
        attentions = torch.exp(rho * self.similarity_func(word, region_feat))
        sum_attentions = torch.sum(attentions, dim=-1).view(-1, 1)
        attentions = torch.div(attentions, sum_attentions)
        # print(attentions.shape)
        contexts = torch.matmul(attentions, region_feat)
        # print(contexts.shape)
        scores_mat = torch.exp(self.similarity_func(word, contexts))
        score = 0
        for idx in range(scores_mat.size(0)):
            score += scores_mat[idx][idx]
        score = torch.log(score) / temperature
        return score

    def attentional_contrastive_loss(self, batch_word, batch_region):
        total_scores = []
        for word_idx in range(batch_word.size(0)):
            word_feat = batch_word[word_idx].squeeze()
            one_sent_scores = []
            for region_idx in range(batch_region.size(0)):
                region_feat = batch_region[region_idx].squeeze()
                score = self.get_attentional_score(word_feat, region_feat)
                one_sent_scores.append(score)
            one_sent_scores = torch.stack(one_sent_scores, dim=0)
            total_scores.append(one_sent_scores)
        total_scores = torch.stack(total_scores, dim=0)
        loss_mat = torch.exp(total_scores)
        sum_loss = torch.sum(loss_mat, dim=-1).view(-1, 1)
        loss = -torch.log(torch.div(loss_mat, sum_loss))
        losses = []
        for idx in range(loss.size(0)):
            element_loss = loss[idx][idx]
            losses.append(element_loss)
        losses = torch.stack(losses, 0)
        n = loss.size(0)
        sum_losses = torch.sum(losses)
        output = sum_losses / n
        return output

    def l2_normalize(self, x, axis=None, epsilon=1e-12):
        square = torch.square(x)
        square_sum = torch.sum(square, dim=axis, keepdims=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon)))
        return torch.multiply(x, x_inv_norm)


def hinge_loss_d(out_d_real, out_d_fake):
    real_loss = torch.relu(1.0 - out_d_real)
    fake_loss = torch.relu(1.0 + out_d_fake)
    loss = torch.mean(real_loss + fake_loss)
    return loss


def hinge_loss_g(out_d_fake):
    loss = -torch.mean(out_d_fake)
    return loss


def hinge_loss_d_revised(out_d_real, out_d_fake):
    """"exactly same with hinge_loss_d"""
    real_loss = torch.minimum(torch.zeros_like(out_d_real), -1.0 + out_d_real)
    real_loss = -torch.mean(real_loss)
    fake_loss = torch.minimum(torch.zeros_like(out_d_fake), -1.0 - out_d_fake)
    fake_loss = -torch.mean(fake_loss)
    loss = real_loss + fake_loss
    return loss

# x = torch.sigmoid(torch.randn(64, 1))
# y = torch.sigmoid(torch.randn(64, 1))
# print(hinge_loss_d_revised(x, y))
# print(hinge_loss_d(x, y))

class NTXentLoss():
    def __init__(self, temperature = 0.1):
        self.loss_func = losses.NTXentLoss(temperature)

    def get_contrastive_loss(self, pair1, pair2):
        embeddings = torch.cat([pair1, pair2])
        indices = torch.arange(0, pair1.size(0))
        labels = torch.cat([indices, indices])
        loss = self.loss_func(embeddings, labels)
        return loss




# batch_size = 64
# test_img_feat = torch.randn(batch_size, 768, 1, 1)
# test_sent_feat = torch.randn(batch_size, 768)
# test_region_feat = torch.randn(batch_size, 768, 16, 16)
# test_word_feat = torch.randn(batch_size, 16, 768)
#
# loss = ContrastiveLoss()
# result = loss.attentional_contrastive_loss(test_word_feat, test_region_feat)
# result.backward()