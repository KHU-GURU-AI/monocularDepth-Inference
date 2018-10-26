import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util as util
from .base_model import BaseModel
from . import networks
from util import ReplayBuffer, LambdaLR, weights_init_normal
import sys

class CycleGanModel(BaseModel):
    def name(self):
        return 'TrainCycleGanModel'

    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.input_A = self.Tensor(args.batchSize, 3, 1024, 256)
        self.input_B = self.Tensor(args.batchSize, 3, 1024, 256)

        self.fake_A_Buffer = ReplayBuffer()
        self.fake_B_Buffer = ReplayBuffer()


        self.netG_AtoB = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, args.init_type, self.gpu_ids)
        self.netG_BtoA = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, args.init_type, self.gpu_ids)
        self.netD_A = networks.define_D(3, 64, 'basic', norm='instance', use_sigmoid=False, gpu_ids=args.gpu_ids)
        self.netD_B = networks.define_D(3, 64, 'basic', norm='instance', use_sigmoid=False, gpu_ids=args.gpu_ids)

        self.netG_AtoB.apply(weights_init_normal)
        self.netG_BtoA.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        checkpoint_AtoB_filename = 'netG_A2B.pth'
        checkpoint_BtoA_filename = 'netG_B2A.pth'

        checkpoint_D_A_filename = 'netD_A.pth'
        checkpoint_D_B_filename = 'netD_B.pth'

        checkpoint_path_AtoB = os.path.join(args.checkpoints_dir, checkpoint_AtoB_filename)
        checkpoint_path_BtoA = os.path.join(args.checkpoints_dir, checkpoint_BtoA_filename)

        checkpoint_path_D_A = os.path.join(args.checkpoints_dir, checkpoint_D_A_filename)
        checkpoint_path_D_B = os.path.join(args.checkpoints_dir, checkpoint_D_B_filename)

        # Load checkpoint
        # self.netG_AtoB.load_state_dict(torch.load(checkpoint_path_AtoB))
        # self.netG_BtoA.load_state_dict(torch.load(checkpoint_path_BtoA))
        # self.netD_A.load_state_dict(torch.load(checkpoint_path_D_A))
        # self.netD_B.load_state_dict(torch.load(checkpoint_path_D_B))
        
        # define loss
        # self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionGAN = torch.nn.MSELoss().cuda()
        self.criterionCycle = torch.nn.L1Loss().cuda()
        self.criterionIdentity = torch.nn.L1Loss().cuda()

        # init optimizer
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AtoB.parameters(), self.netG_BtoA.parameters())
                                            , lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D_a = torch.optim.Adam(self.netD_A.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D_b = torch.optim.Adam(self.netD_B.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        self.lr_scheduler_D_a = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_a, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        self.lr_scheduler_D_b = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_b, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)


    def set_input(self, input_real, input_fake):
        self.image_real_sizes = input_real['A_sizes']

        input_A = input_real['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_real_paths = input_real['A_paths']

        # self.size_real = (int(self.image_real_sizes[0]), int(self.image_real_sizes[1]))

        self.image_fake_sizes = input_fake['B_sizes']

        input_B = input_fake['B']
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_fake_paths = input_fake['B_paths']

        # self.size_fake = (int(self.image_fake_sizes[0]), int(self.image_fake_sizes[1]))

    def train(self):
        real_A = Variable(self.input_A)
        real_B = Variable(self.input_B)
        target_real = Variable(self.Tensor(real_B.size(0), 1, 14, 62).fill_(1.0), requires_grad=False)
        target_fake = Variable(self.Tensor(real_B.size(0), 1, 14, 62).fill_(0.0), requires_grad=False)
        loss_gan = self.criterionGAN
        loss_cycle = self.criterionCycle
        loss_identity = self.criterionIdentity

        self.optimizer_G.zero_grad()

        i_b = self.netG_AtoB(real_B)
        loss_identity_B = loss_identity(i_b, real_B) * 0.5
        i_a = self.netG_BtoA(real_A)
        loss_identity_A = loss_identity(i_a, real_A) * 0.5

        fake_B = self.netG_AtoB(real_A)
        pred_fake = self.netD_B(fake_B)
        loss_gan_A2B = loss_gan(pred_fake, target_real)
        fake_A = self.netG_BtoA(real_B)
        pred_fake = self.netD_A(fake_A)
        loss_gan_B2A = loss_gan(pred_fake, target_real)

        recovered_a = self.netG_BtoA(fake_B)
        loss_cycle_A = loss_cycle(recovered_a, real_A) * 10.0
        recovered_b = self.netG_AtoB(fake_A)
        loss_cycle_B = loss_cycle(recovered_b, real_B) * 10.0

        loss_G = loss_identity_A + loss_identity_B + loss_gan_A2B + loss_gan_B2A + loss_cycle_A + loss_cycle_B
        loss_G.backward()

        self.optimizer_G.step()


        self.optimizer_D_a.zero_grad()

        pred_real = self.netD_A(real_A)
        loss_d_real = loss_gan(pred_real, target_real)
        fake_A = self.fake_A_Buffer.push_and_pop(fake_A)
        pred_fake = self.netD_A(fake_A.detach())
        loss_d_fake = loss_gan(pred_fake, target_fake)

        loss_d_a = (loss_d_real + loss_d_fake) * 0.5
        loss_d_a.backward()

        self.optimizer_D_a.step()


        self.optimizer_D_b.zero_grad()

        pred_real = self.netD_B(real_B)
        loss_d_real = loss_gan(pred_real, target_real)
        fake_B = self.fake_B_Buffer.push_and_pop(fake_B)
        pred_fake = self.netD_B(fake_B.detach())
        loss_d_fake = loss_gan(pred_fake, target_fake)

        loss_d_b = (loss_d_real + loss_d_fake) * 0.5
        loss_d_b.backward()

        self.optimizer_D_b.step()

        print('Generator Total Loss : {a:.3f},   Generator Identity Loss : {b:.3f},   Generator GAN Loss : {c:.3f},   '
              'Generator Cycle Loss : {d:.3f}'.format(a=loss_G, b=loss_identity_A + loss_identity_B,
                                                      c=loss_gan_A2B + loss_gan_B2A, d=loss_cycle_A + loss_cycle_B))
        print('Discriminator Loss : {a:.3f}'.format(a=loss_d_a + loss_d_b))

    def update_learning_rate(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_a.step()
        self.lr_scheduler_D_b.step()

    def save_checkpoint(self):
        torch.save(self.netG_AtoB.state_dict(), './checkpoints/netG_A2B.pth')
        torch.save(self.netG_BtoA.state_dict(), './checkpoints/netG_B2A.pth')
        torch.save(self.netD_A.state_dict(), './checkpoints/netD_A.pth')
        torch.save(self.netD_B.state_dict(), './checkpoints/netD_B.pth')

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG_AtoB(self.real_A)

    def get_image_paths(self):
        return self.image_real_paths, self.image_fake_paths

    def get_image_sizes(self):
        return self.size_real, self.size_fake

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)

        return OrderedDict([('original', real_A), ('restyled', fake_B)])
