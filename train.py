from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import define_G, define_D, print_network
# from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

from blur_dataset import BlurDataset, GaussianBlur, INVERSE_NORMALIZE

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', type=str, default=r"/mnt/SSD_Data/mirflickr/",
                    help='dataset path')
parser.add_argument('--batchSize', type=int, default=1,
                    help='training batch size')
parser.add_argument('--testBatchSize', type=int,
                    default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3,
                    help='input image channels')
parser.add_argument('--output_nc', type=int, default=3,
                    help='output image channels')
parser.add_argument('--ngf', type=int, default=64,
                    help='generator filters in first conv layer')
parser.add_argument('--Diters', type=int, default=5,
                    help='number of D iters per each G iter')
parser.add_argument('--ndf', type=int, default=64,
                    help='discriminator filters in first conv layer')
parser.add_argument('--lrD', type=float, default=0.00005,
                    help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005,
                    help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10,
                    help='weight on L1 term in objective')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = BlurDataset(opt.dataset + "train/", GaussianBlur(5, 1.5))
test_set = BlurDataset(opt.dataset + "val/", GaussianBlur(5, 1.5))
training_data_loader = DataLoader(
    dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, [0])
netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'batch', False, [0])


criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG,
#                         betas=(opt.beta1, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD,
#                         betas=(opt.beta1, 0.999))

optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)

print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)

TO_PIL = transforms.ToPILImage()


def save_debug_image(tensor_real, tensor_blur, tensor_recovered, filename):
    if not os.path.exists("debug/"):
        os.mkdir("debug/")
    assert tensor_real.size() == tensor_recovered.size()
    recovered = TO_PIL(
        (INVERSE_NORMALIZE(tensor_recovered.cpu()) * 255).clamp(0, 255).byte())
    real = TO_PIL((INVERSE_NORMALIZE(tensor_real.cpu())
                   * 255).clamp(0, 255).byte())
    blur = TO_PIL((INVERSE_NORMALIZE(tensor_blur.cpu())
                   * 255).clamp(0, 255).byte())
    # recovered = TO_PIL(
    #     (tensor_recovered.cpu() * 255).round().byte())
    # real = TO_PIL((tensor_real.cpu() * 255).round().byte())
    # blur = TO_PIL((tensor_blur.cpu() * 255).round().byte())
    new_im = Image.new('RGB', (real.size[0] * 3 + 10, real.size[1]))
    new_im.paste(blur, (0, 0))
    new_im.paste(recovered, (real.size[0] + 5, 0))
    new_im.paste(real, (real.size[0] * 2 + 10, 0))
    new_im.save(os.path.join("debug/", filename))


def train(epoch):
    one = torch.cuda.FloatTensor([1])
    mone = one * -1
    n_critic = 50 if epoch == 1 else opt.Diters
    loss_d = []
    loss_g = []
    gen_iter_cnt = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        # Prepare tensor
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        # Generate Fake
        fake_b = netG(real_a)
        if n_critic:
            ############################
            # (1) Update D network:
            ###########################

            optimizerD.zero_grad()

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            D_real = netD.forward(real_ab)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            D_fake = netD.forward(fake_ab.detach())
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # Combined loss
            loss_d.append((D_real - D_fake).data)
            optimizerD.step()
            n_critic -= 1

        else:
            ############################
            # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
            ##########################
            optimizerG.zero_grad()
            # noise = Variable(torch.torch.randn(real_a.size()[0], 2))

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            D_fake = netD.forward(fake_ab)
            D_fake = D_fake.mean()
            D_fake.backward(mone, retain_graph=True)
            G_cost = -D_fake

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
            loss_g_l1.backward()
            loss_g.append((G_cost + loss_g_l1).data)

            optimizerG.step()
            n_critic = opt.Diters
            gen_iter_cnt += 1
            if gen_iter_cnt == 20:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, iteration, len(training_data_loader),
                    torch.cat(loss_d).mean(), torch.cat(loss_g).mean()))
                # print(fake_b.data.mean())
                loss_d = []
                loss_g = []
                gen_iter_cnt = 0

        if iteration and iteration % 100 == 0:
            save_debug_image(
                real_b.data[0], real_a.data[0], fake_b.data[0],
                "{}_{}.png".format(epoch, iteration))


def test():
    avg_psnr, avg_loss = 0, 0
    for batch in testing_data_loader:
        x, target = Variable(batch[0], volatile=True), Variable(
            batch[1], volatile=True)
        if opt.cuda:
            x = x.cuda()
            target = target.cuda()

        prediction = netG(x)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr

        # train with fake
        fake_ab = torch.cat((x, prediction), 1)
        D_fake = netD.forward(fake_ab.detach())
        D_fake = D_fake.data.mean()
        # train with real
        real_ab = torch.cat((x, target), 1)
        D_real = netD.forward(real_ab)
        D_real = D_real.data.mean()
        # Combined loss
        avg_loss += D_real - D_fake

    print("===> Avg. PSNR: {:.4f} dB Avg. Loss: {:.4f}".format(
        avg_psnr / len(testing_data_loader), avg_loss / len(testing_data_loader)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint/"):
        os.mkdir("checkpoint/")
    net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(
        epoch)
    net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(
        epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint/"))


# Train Generator First
print("Training generator...")
for iteration, batch in tqdm(
        enumerate(training_data_loader, 1), total=len(training_data_loader)):
    # Prepare tensor
    real_a_cpu, real_b_cpu = batch[0], batch[1]
    real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
    real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
    optimizerG.zero_grad()
    # Generate Fake
    fake_b = netG(real_a)
    # G(A) = B
    loss_g_l1 = criterionL1(fake_b, real_b)
    loss_g_l1.backward()
    optimizerG.step()
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

# GAN Training
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    if epoch % 2 == 0:
        checkpoint(epoch)
