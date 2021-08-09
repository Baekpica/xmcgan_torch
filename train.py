# TBD: module import
import torch
from torch import optim
from pytorch_metric_learning import losses
from xmcgan import dataset, generator, discriminator
import xmcgan.losses as xmc_losses
import torch.nn as nn

# TBD - set params
num_epochs = 100
batch_size = 64
d_iter_per_g = 2
loss_coef_1 = 1.0
loss_coef_2 = 1.0
loss_coef_3 = 1.0
lr_d = 4e-4
lr_g = 1e-4
beta1 = 0.5
beta2 = 0.999
temperature = 0.1

device = 'cuda'

model_d = discriminator.Discriminator().to(device)
# model_d = nn.DataParallel(model_d, [0, 1])
model_g = generator.Generator().to(device)
# model_g = nn.DataParallel(model_g, [0, 1])

data_class = dataset.COCO_Dataset(set_name='train2014')
data_loader = dataset.DataLoader(data_class,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

bert = dataset.BertEmbeddings()
resnet = dataset.ResNetEmbedding().to(device)

# loss func
contrastive_loss = losses.NTXentLoss(temperature)
# sent_contrastive = xmc_losses.SentContrastiveLoss()

# # initialize weights
# model_d.apply(initialize_weights)
# model_g.apply(initialize_weights)

# optimizer
opt_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(beta1, beta2))
opt_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(beta1, beta2))

d_losses = []
g_losses = []
for epoch in range(num_epochs):
    print('epoch:', epoch)
    # for d_iter in range(d_iter_per_g):
    for idx, (images, sents) in enumerate(data_loader):
        images = images.to(device)
        word, sents, max_len = bert.get_bert_for_caption(sents)
        word = word.to(device)
        sents = sents.to(device)
        max_len = max_len.to(device)
        model_d.zero_grad()
        noises = torch.randn(batch_size, 128).to(device)
        out_g = model_g(noises, sents, word).to(device)
        out_d_real, img_feat_real = model_d(images, sents)
        out_d_fake, img_feat_fake = model_d(out_g, sents)

        # real_sent_c_loss = sent_contrastive.contrastive_loss(images, sents)
        # real_sent_c_loss.backward()
        # real_word_c_loss = word_loss_func(images, sents)
        # real_word_c_loss.backward()
        d_gan_loss = xmc_losses.hinge_loss_d(out_d_real, out_d_fake, device)
        d_gan_loss.backward()
        # d_loss = d_gan_loss + loss_coef_1*real_sent_c_loss + loss_coef_2*real_word_c_loss
        d_losses.append(d_gan_loss)
        opt_d.step()

    # for images, sents in data_loader:
    #     word, sents, max_len = bert.get_bert_for_caption(sents)
    #     word = word.to(device)
    #     sents = sents.to(device)
    #     max_len = max_len.to(device)
    #     noises = torch.randn(batch_size, 128)
        model_g.zero_grad()
        out_g = model_g(noises, sents, word)
        out_d_fake, img_feat_fake = model_d(out_g, sents)
        # fake_sent_c_loss = sent_contrastive.contrastive_loss(img_feat_fake, sents)
        # fake_sent_c_loss.backward()
        # fake_word_c_loss = word_loss_func(out_g, sents)
        # fake_word_c_loss.backward()
        # img_c_loss = xmc_losses.get_contrastive_loss(out_g.view(batch_size,-1), images.view(batch_size, -1), contrastive_loss)
        # img_c_loss.backward()

        g_gan_loss = xmc_losses.hinge_loss_g(out_d_fake)
        g_gan_loss.backward()
        g_losses.append(g_gan_loss)
        # g_loss = g_gan_loss + loss_coef_1*fake_sent_c_loss + loss_coef_2*fake_word_c_loss + loss_coef_3*img_c_loss
        opt_g.step()

        if idx % 100 == 0:
            print('d:', d_gan_loss)
            print('g:', g_gan_loss)
