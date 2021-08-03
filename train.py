# TBD: module import
import torch
from torch import optim

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

model_d = Discriminator(params_d).to(device)
model_g = Genrator(params_g).to(device)

model_d.apply(initialize_weights)
model_g.apply(initialize_weights)

#optimizer
opt_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(beta1, beta2))
opt_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(beta1, beta2))

for epoch in range(num_epochs):
    for d_iter in range(d_iter_per_g):
        for images, sents in train_dataloader:
            model_d.zero_grad()
            noises = torch.randn()
            out_g = model_g(noises, sents)
            out_d_real = model_d(images, sents)
            out_d_fake = model_d(out_g, sents)
            real_sent_c_loss = sent_loss_func(images, sents)
            real_sent_c_loss.backward()
            real_word_c_loss = word_loss_func(images, sents)
            real_word_c_loss.backward()
            d_gan_loss = d_loss_func(out_d_real, out_d_fake)
            d_gan_loss.backward()
            d_loss = d_gan_loss + loss_coef_1*real_sent_c_loss + loss_coef_2*real_word_c_loss
            opt_d.step()
    for images, sents in train_dataloader:
        model_g.zero_grad()
        noises = torch.randn()
        out_g = model_g(noises, sents)
        fake_sent_c_loss = sent_loss_func(out_g, sents)
        fake_sent_c_loss.backward()
        fake_word_c_loss = word_loss_func(out_g, sents)
        fake_word_c_loss.backward()
        img_c_loss = img_loss_func(out_g, images)
        img_c_loss.backward()
        g_gan_loss = g_loss_func(model_d(out_g), sents)
        g_gan_loss.backward()
        g_loss = g_gan_loss + loss_coef_1*fake_sent_c_loss + loss_coef_2*fake_word_c_loss + loss_coef_3*img_c_loss
        opt_g.step()
