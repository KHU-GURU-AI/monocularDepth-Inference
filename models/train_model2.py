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
from torchvision.utils import save_image
import sys


class GanModel(BaseModel):
    def name(self):
        return 'TrainGanModel'

    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.input_B = self.Tensor(args.batchSize, 3, 1024, 256)
        self.input_C = self.Tensor(args.batchSize, 1, 1024, 256)

        self.fake_Buffer = ReplayBuffer()

        self.netG_BtoC = networks.define_G(3, 1, 64, 'unet_128', 'batch', False, args.init_type, self.gpu_ids)
        self.netD_C = networks.define_D(1, 64, 'basic', norm='batch', use_sigmoid=False, gpu_ids=args.gpu_ids)

        self.netG_BtoC.apply(weights_init_normal)
        self.netD_C.apply(weights_init_normal)

        checkpoint_BtoC_filename = 'netG_B2C.pth'
        checkpoint_D_C_filename = 'netD_C.pth'

        checkpoint_path_BtoC = os.path.join(args.checkpoints_dir, checkpoint_BtoC_filename)
        checkpoint_path_D_C = os.path.join(args.checkpoints_dir, checkpoint_D_C_filename)

        # Load checkpoint
        # self.netG_BtoC.load_state_dict(torch.load(checkpoint_path_BtoC))
        # self.netD_C.load_state_dict(torch.load(checkpoint_path_D_C))

        # define loss
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionReconstruction = torch.nn.L1Loss().cuda()

        # init optimizer
        self.optimizer_G = torch.optim.Adam(self.netG_BtoC.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD_C.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch,
                                                                                                     args.decay_epoch).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=LambdaLR(args.n_epochs, args.epoch,
                                                                                                     args.decay_epoch).step)

    def set_input(self, input):
        self.image_syn_sizes = input['B_sizes']

        input_B = input['B']
        save_image(input_B[0], './input_check/rgb.jpg')
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_syn_paths = input['B_paths']

        # self.size_syn = (int(self.image_syn_sizes[0]), int(self.image_syn_sizes[1]))

        self.image_dep_sizes = input['C_sizes']

        input_C = input['C']
        save_image(input_C[0], './input_check/depth.jpg')
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_dep_paths = input['C_paths']

        # self.size_dep = (int(self.image_dep_sizes[0]), int(self.image_dep_sizes[1]))

    def train(self):
        syn_data = Variable(self.input_B)
        dep_data = Variable(self.input_C)
        target_real = Variable(self.Tensor(syn_data.size(0), 1, 14, 62).fill_(1.0), requires_grad=False)
        target_fake = Variable(self.Tensor(syn_data.size(0), 1, 14, 62).fill_(0.0), requires_grad=False)
        loss_gan = self.criterionGAN
        loss_rec = self.criterionReconstruction

        self.optimizer_G.zero_grad()

        fake_dep = self.netG_BtoC(syn_data)
        loss_r = loss_rec(fake_dep, dep_data)
        loss_g = loss_gan(self.netD_C(fake_dep), target_real)
        loss_G = 0.01 * loss_g + 0.99 * loss_r
        # loss_G = loss_g
        loss_G.backward()

        self.optimizer_G.step()


        self.optimizer_D.zero_grad()

        pred_real = self.netD_C(dep_data)
        loss_real = loss_gan(pred_real, target_real)
        fake_A = self.fake_Buffer.push_and_pop(fake_dep)
        pred_fake = self.netD_C(fake_A)
        loss_fake = loss_gan(pred_fake, target_fake)

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()

        self.optimizer_D.step()

        print('Generator Loss : {loss_G:.5f}, Discriminator Loss : {loss_D:.5f}'.format(loss_G=loss_G, loss_D=loss_D))

    def update_learning_rate(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()

    def save_checkpoint(self):
        torch.save(self.netG_BtoC.state_dict(), './checkpoints/netG_B2C.pth')
        torch.save(self.netD_C.state_dict(), './checkpoints/netD_C.pth')

    def forward(self):
        self.syn_data = Variable(self.input_B)
        self.pred_depth = self.netG_BtoC(self.syn_data)

    def get_image_paths(self):
        return self.image_syn_paths, self.image_dep_paths

    def get_image_sizes(self):
        return self.size_syn, self.size_dep

    def get_current_visuals(self):
        syn_d = util.tensor2im(self.syn_data.data)
        pred_d = util.tensor2im(self.pred_depth.data)

        return OrderedDict([('original', syn_d), ('depth', pred_d)])
