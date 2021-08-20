from pytorch_metric_learning import losses, distances
import torch
from torch import vmap
import torch.nn.functional as F

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


class AttentionalContrastiveLoss():
    def __init__(self):
        a = 1

    def contrastive_loss(self, image_feat, cond_feat, l2_norm = True, temperature = 0.1):
        if l2_norm:
            image_feat = self.l2_normalize(image_feat, -1)
            cond_feat = self.l2_normalize(cond_feat, -1)
        local_batch_size = image_feat.size(0)

        image_feat_large = image_feat
        cond_feat_large = cond_feat

        labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).cuda()
        logits_img2cond = torch.matmul(image_feat,
                                       cond_feat_large.permute(1, 0).contiguous()) / temperature
        logits_cond2img = torch.matmul(cond_feat,
                                       image_feat_large.permute(1, 0).contiguous()) / temperature
        loss_img2cond = self.cross_entropy_loss_with_logits(labels, logits_img2cond)
        loss_cond2img = self.cross_entropy_loss_with_logits(labels, logits_cond2img)
        loss_img2cond = torch.mean(logits_img2cond)
        loss_cond2img = torch.mean(logits_cond2img)
        loss = loss_img2cond + loss_cond2img
        return loss

    def word_loss(self, image_feat, word_feat, max_len, gamma1=5, gamma2=5, gamma3=50):
        batch_size, region_num, _ = image_feat.shape
        total_len = word_feat.shape[1]

        def element_attention(word_feat_i, max_len_i):
            word_feat_i = word_feat_i[None, :]
            word_feat_i = torch.tile(word_feat_i, [batch_size, 1, 1])
            max_len_i = torch.tile(max_len_i, [region_num])
            mask = torch.arange(total_len, dtype=torch.float32).cuda()[None, :] >= max_len_i.cuda()[:, None]
            mask = mask[None, :]
            mask = torch.tile(mask, (batch_size, 1, 1))
            mask_2 = mask[:, 0, :]
            region_context = self.attention(image_feat, word_feat_i, gamma1, mask)
            row_sim = self.cosine_similarity(word_feat_i, region_context)
            row_sim = row_sim * gamma2
            row_sim = row_sim + mask_2 * (-1e9)
            row_sim = torch.logsumexp(row_sim, dim=-1, keepdim=True)
            row_sim = row_sim / gamma2
            return row_sim

        similarities = vmap(element_attention)(word_feat, max_len)
        similarities = similarities * gamma3
        similarities = torch.squeeze(similarities)
        similarities_transpose = similarities
        similarities = similarities_transpose.permute(1, 0)

        labels = F.one_hot(torch.arange(batch_size), batch_size).cuda()
        loss_0 = self.cross_entropy_loss_with_logits(labels, similarities)
        loss_1 = self.cross_entropy_loss_with_logits(labels, similarities_transpose)
        loss_0 = torch.mean(loss_0)
        loss_1 = torch.mean(loss_1)
        matching_loss = loss_0 + loss_1
        return matching_loss

    def attention(self, region_feat, word_feat, gamma, mask=None):
        region_feat = self.l2_normalize(region_feat, -1)
        word_feat = self.l2_normalize(word_feat, -1)
        attn_matrix = torch.matmul(region_feat, word_feat.permute(0, 2, 1))
        attn_matrix = attn_matrix * gamma
        if mask is not None:
            attn_matrix = attn_matrix + mask*(-1e9)
        alpha = torch.softmax(attn_matrix, dim=-2)
        region_context = torch.matmul(alpha.permute(0, 2, 1), region_feat)
        return region_context

    def cross_entropy_loss_with_logits(self, labels, logits):
        logp = torch.log_softmax(logits, -1)
        loss = - torch.sum(torch.multiply(labels, logp), dim=-1)
        return loss

    def l2_normalize(self, x, axis=None, epsilon=1e-12):
        square = torch.square(x)
        square_sum = torch.sum(square, dim=axis, keepdims=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon)))
        return torch.multiply(x, x_inv_norm)

    def cosine_similarity(self, x1, x2):
        dist = torch.sum(torch.multiply(x1, x2), -1)
        dist = dist / (torch.linalg.norm(x1, dim=-1) * torch.linalg.norm(x2, dim=-1))
        return dist



# model = AttentionalContrastiveLoss()
# test_word = torch.randn(64, 16, 768).cuda()
# test_region = torch.randn(64, 256, 768).cuda()
# test_maxlen = torch.tensor([13, 16, 14, 12, 15, 13, 15, 16, 16, 12, 13, 11, 14, 15, 12, 10, 13, 14,
#                             12, 10, 11, 12, 14, 12, 14, 14, 16, 11, 13, 11, 16, 13, 12, 16, 14, 14,
#                             16, 16, 12, 14, 14, 16, 13, 11, 16, 13, 11, 16, 13, 13, 12, 13, 12, 16,
#                             13, 12, 16, 14, 13, 15, 12, 10, 11, 16]).cuda()
# result = model.word_loss(test_region, test_word, test_maxlen)
# print(result)