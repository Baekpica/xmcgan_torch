# TBD: module import
import matplotlib.pyplot as plt
import torch
from torch import optim
from xmcgan import dataset, generator, discriminator
import xmcgan.losses as xmc_losses
from torch.autograd import Variable
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

# def setup(rank, world_size):
#     dist.init_process_group(
#         backend='nccl',
#         init_method='tcp://127.0.0.1:61105',
#         world_size=world_size,
#         rank=rank
#     )
#
#
# def cleanup():
#     dist.destroy_process_group()
#
#
# def main():
#     n_gpus = torch.cuda.device_count()
#     torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus,))
#
#
# def main_worker(gpu, n_gpus):
#     device = 'cuda'
#     batch_size = 6 * 3
#     num_worker = 4 * n_gpus
#     epochs = 100
#     d_iter_per_g = 2
#     loss_coef_1 = 1.0
#     loss_coef_2 = 1.0
#     loss_coef_3 = 1.0
#     lr_d = 4e-4
#     lr_g = 1e-4
#     beta1 = 0.5
#     beta2 = 0.999
#     temperature = 0.1
#
#     batch_size = int(batch_size/n_gpus)
#     num_worker = int(num_worker/n_gpus)
#
#     torch.distributed.init_process_group(
#         backend='nccl',
#         init_method='tcp://127.0.0.1:61105',
#         world_size=n_gpus,
#         rank=gpu
#     )
#
#     model_d = discriminator.Discriminator(batch_size)
#     model_g = generator.Generator(batch_size)
#
#     torch.cuda.set_device(gpu)
#     model_d = model_d.cuda(gpu)
#     model_d = torch.nn.parallel.DistributedDataParallel(model_d, device_ids=[gpu])
#     model_g = model_g.cuda(gpu)
#     model_g = torch.nn.parallel.DistributedDataParallel(model_g, device_ids=[gpu])
#
#     dist.barrier()
#     data_class = dataset.COCO_Dataset(set_name='train2014')
#     data_sampler = dataset.DistributedSampler(data_class)
#     data_loader = dataset.DataLoader(data_class,
#                                      batch_size=batch_size,
#                                      shuffle=False,
#                                      num_workers=num_worker,
#                                      pin_memory=True,
#                                      sampler=data_sampler)
#
#     bert = dataset.BertEmbeddings()
#
#     bert = dataset.BertEmbeddings()
#     # resnet = dataset.ResNetEmbedding().to(device)
#
#     # loss func
#     contrastive_loss = xmc_losses.ContrastiveLoss()
#
#     # # initialize weights
#     # model_d.apply(initialize_weights)
#     # model_g.apply(initialize_weights)
#
#     # optimizer
#     opt_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(beta1, beta2))
#     opt_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(beta1, beta2))
#
#     d_losses = []
#     g_losses = []
#     for epoch in range(epochs):
#         print('epoch:', epoch)
#         # for d_iter in range(d_iter_per_g):
#         with torch.autograd.set_detect_anomaly(True):
#             for idx, (images, sents) in enumerate(data_loader):
#                 images = images.to(device)
#                 word, sents, max_len = bert.get_bert_for_caption(sents)
#                 word = word.to(device)
#                 sents = sents.to(device)
#                 max_len = max_len.to(device)
#                 model_d.zero_grad()
#                 noises = torch.randn(batch_size, 128).to(device)
#                 out_g = model_g(noises, sents, word).to(device)
#                 out_d_real, region_feat_real, img_feat_real = model_d(images, sents)
#                 out_d_fake, region_feat_fake, img_feat_fake = model_d(out_g, sents)
#
#                 real_sent_c_loss = contrastive_loss.contrastive_loss(img_feat_real.view(batch_size, -1), sents)
#                 real_sent_c_loss.backward(retain_graph=True)
#                 real_word_c_loss = contrastive_loss.attentional_contrastive_loss(word, region_feat_real)
#                 real_word_c_loss.backward(retain_graph=True)
#                 d_gan_loss = xmc_losses.hinge_loss_d(out_d_real, out_d_fake, device)
#                 d_gan_loss.backward()
#                 d_loss = d_gan_loss + loss_coef_1 * real_sent_c_loss + loss_coef_2 * real_word_c_loss
#                 d_losses.append(d_loss)
#                 opt_d.step()
#
#                 if idx % 2 == 0:
#                     model_g.zero_grad()
#                     out_g = model_g(noises, sents, word)
#                     out_d_fake, region_feat_fake, img_feat_fake = model_d(out_g, sents)
#                     out_d_real, region_feat_real, img_feat_real = model_d(images, sents)
#                     fake_sent_c_loss = contrastive_loss.contrastive_loss(img_feat_fake.view(batch_size, -1), sents)
#                     fake_sent_c_loss.backward(retain_graph=True)
#                     fake_word_c_loss = contrastive_loss.attentional_contrastive_loss(word, region_feat_fake)
#                     fake_word_c_loss.backward(retain_graph=True)
#                     img_c_loss = contrastive_loss.contrastive_loss(img_feat_real.view(batch_size, -1),
#                                                                    img_feat_fake.view(batch_size, -1))
#                     img_c_loss.backward(retain_graph=True)
#
#                     g_gan_loss = xmc_losses.hinge_loss_g(out_d_fake)
#                     g_gan_loss.backward()
#                     g_loss = g_gan_loss + loss_coef_1 * fake_sent_c_loss + loss_coef_3 * img_c_loss + loss_coef_2*fake_word_c_loss
#                     g_losses.append(Variable(g_loss, requires_grad=False))
#                     opt_g.step()
#
#                 print(f'\r{(idx + 1) * batch_size}/{data_class.__len__()}'
#                       f'\t{round(((idx + 1) * batch_size / data_class.__len__()) * 100, 1)}%',
#                       end='')
#
#                 if idx % 100 == 0:
#                     print('\nDiscriminator Loss:', d_loss,
#                           '\n\tImg-Sent Real Contrastive Loss:', real_sent_c_loss,
#                           '\n\tWord-Region Real Contastive Loss:', real_word_c_loss,
#                           '\n\tDiscriminator Hinge Loss:', d_gan_loss)
#                     print('Generator Loss:', g_loss,
#                           '\n\tImg-Sent Fake Contrastive Loss:', fake_sent_c_loss,
#                           '\n\tWord-Region Fake Contrastive Loss:', fake_word_c_loss,
#                           '\n\tImg-Img Contrastive Loss:', img_c_loss,
#                           '\n\tGenerator Hinge Loss:', g_gan_loss)
#
# if __name__ == "__main__":
#     main()
#
#     # if cfg['load_path']:

# TBD - set params
num_epochs = 100
batch_size = 18 * 3
num_gpu = 3
device = 'cuda'
d_iter_per_g = 2
loss_coef_1 = 1.0
loss_coef_2 = 1.0
loss_coef_3 = 1.0
lr_d = 4e-4
lr_g = 1e-4
beta1 = 0.5
beta2 = 0.999
temperature = 0.1

model_d = discriminator.Discriminator(int(batch_size/num_gpu)).to(device)
model_d = nn.DataParallel(model_d, [0, 1, 2])
model_g = generator.Generator(int(batch_size/num_gpu)).to(device)
model_g = nn.DataParallel(model_g, [0, 1, 2])

data_class = dataset.COCO_Dataset(set_name='train2014')
data_loader = dataset.DataLoader(data_class,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

bert = dataset.BertEmbeddings()
# resnet = dataset.ResNetEmbedding().to(device)

# loss func
contrastive_loss = xmc_losses.ContrastiveLoss()

# # initialize weights
# model_d.apply(initialize_weights)
# model_g.apply(initialize_weights)

# optimizer
opt_d = optim.Adam(model_d.parameters(), lr=lr_d, betas=(beta1, beta2))
opt_g = optim.Adam(model_g.parameters(), lr=lr_g, betas=(beta1, beta2))

img_list = []
d_losses = []
g_losses = []
for epoch in range(num_epochs):
    print('epoch:', epoch)
    for idx, (images, sents) in enumerate(data_loader):
        images = images.to(device)
        word, sents, max_len = bert.get_bert_for_caption(sents)
        word = word.to(device)
        sents = sents.to(device)
        max_len = max_len.to(device)
        model_d.zero_grad()
        noises = torch.randn(batch_size, 128).to(device)
        out_g = model_g(noises, sents, word).to(device)
        out_d_real, region_feat_real, img_feat_real = model_d(images, sents)
        out_d_fake, region_feat_fake, img_feat_fake = model_d(out_g, sents)

        real_sent_c_loss = contrastive_loss.contrastive_loss(img_feat_real.view(batch_size, -1), sents)
        real_sent_c_loss.backward(retain_graph=True)
        real_word_c_loss = contrastive_loss.attentional_contrastive_loss(word, region_feat_real)
        real_word_c_loss.backward(retain_graph=True)
        d_gan_loss = xmc_losses.hinge_loss_d(out_d_real, out_d_fake, device)
        d_gan_loss.backward()
        d_loss = d_gan_loss + loss_coef_1*real_sent_c_loss + loss_coef_2*real_word_c_loss
        # d_losses.append(d_loss)
        opt_d.step()

        if idx % 2 == 0:
            model_g.zero_grad()
            out_g = model_g(noises, sents, word)
            out_d_fake, region_feat_fake, img_feat_fake = model_d(out_g, sents)
            out_d_real, region_feat_real, img_feat_real = model_d(images, sents)
            fake_sent_c_loss = contrastive_loss.contrastive_loss(img_feat_fake.view(batch_size, -1), sents)
            fake_sent_c_loss.backward(retain_graph=True)
            fake_word_c_loss = contrastive_loss.attentional_contrastive_loss(word, region_feat_fake)
            fake_word_c_loss.backward(retain_graph=True)
            img_c_loss = contrastive_loss.contrastive_loss(img_feat_real.view(batch_size, -1),
                                                           img_feat_fake.view(batch_size, -1))
            img_c_loss.backward(retain_graph=True)

            g_gan_loss = xmc_losses.hinge_loss_g(out_d_fake)
            g_gan_loss.backward()
            g_loss = g_gan_loss + loss_coef_1*fake_sent_c_loss + loss_coef_3*img_c_loss + loss_coef_2 * fake_word_c_loss
            # g_losses.append(Variable(g_loss, requires_grad=False))
            opt_g.step()

        print(f'\r{(idx + 1) * batch_size}/{data_class.__len__()}'
              f'\t{round(((idx + 1) * batch_size / data_class.__len__()) * 100, 1)}%',
              end='')

        if idx % 100 == 0:
            print('\nDiscriminator Loss:', d_loss,
                  '\n\tImg-Sent Real Contrastive Loss:', real_sent_c_loss,
                  '\n\tWord-Region Real Contastive Loss:', real_word_c_loss,
                  '\n\tDiscriminator Hinge Loss:', d_gan_loss)
            print('Generator Loss:', g_loss,
                  '\n\tImg-Sent Fake Contrastive Loss:', fake_sent_c_loss,
                  '\n\tWord-Region Fake Contrastive Loss:', fake_word_c_loss,
                  '\n\tImg-Img Contrastive Loss:', img_c_loss,
                  '\n\tGenerator Hinge Loss:', g_gan_loss)

        if idx % 10 == 0:
            # tf = transforms.ToPILImage()
            # for ii in range(16):
            #     plt.subplot(4, 4, ii+1)
            #     plt.imshow(tf(out_g[ii]))
            #     plt.axis('off')
            # plt.show()
            # plt.clf()
            # for ii in range(16):
            #     plt.subplot(4, 4, ii+1)
            #     plt.imshow(tf(images[ii]))
            #     plt.axis('off')
            # plt.show()
            # plt.clf()

            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(images[:16].to('cpu'), normalize=True), (1,2,0)))

            # Plot the fake images from the last epoch
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(vutils.make_grid(out_g[:16].to('cpu'), normalize=True), (1,2,0)))
            plt.savefig(f'./exp/20210812/{epoch}_{idx}.png')
            plt.clf()