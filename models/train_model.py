import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util as util
from .base_model import BaseModel
from . import networks
import sys

class TrainModel(BaseModel):
    def name(self):
        return 'TrainModel'
    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.input_A = self.Tensor(1, 3, 1024, 256)

        self.netG_AtoB = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, args.init_type, self.gpu_ids)
        self.netG_BtoA = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, args.init_type, self.gpu_ids)
        self.netG_BtoC = networks.define_G(3, 1, 64, 'unet_256', 'batch', False, args.init_type, self.gpu_ids)

        self.netD_A = networks.define_D(3, 64, 'pixel', norm='instance', use_sigmoid=False, gpu_ids=args.gpu_ids)
        self.netD_B = networks.define_D(3, 64, 'pixel', norm='instance', use_sigmoid=False, gpu_ids=args.gpu_ids)
        self.netD_C = networks.define_D(3, 64, 'pixel', norm='batch', use_sigmoid=False, gpu_ids=args.gpu_ids) 

        checkpoint_AtoB_filename = 'netG_A2B.pth'
        checkpoint_BtoA_filename = 'netG_B2A.pth'
        checkpoint_BtoC_filename = 'netG_B2C.pth'

        checkpoint_D_A_filename = 'netD_A.pth'
        checkpoint_D_B_filename = 'netD_B.pth'
        checkpoint_D_C_filename = 'netD_C.pth'

        checkpoint_path_AtoB = os.path.join(args.checkpoints_dir, checkpoint_AtoB_filename)
        checkpoint_path_BtoA = os.path.join(args.checkpoints_dir, checkpoint_BtoA_filename)
        checkpoint_path_BtoC = os.path.join(args.checkpoints_dir, checkpoint_BtoC_filename)

        checkpoint_path_D_A = os.path.join(args.checkpoints_dir, checkpoint_D_A_filename)
        checkpoint_path_D_B = os.path.join(args.checkpoints_dir, checkpoint_D_B_filename)
        checkpoint_path_D_C = os.path.join(args.checkpoints_dir, checkpoint_D_C_filename)

        # Load checkpoint
        # self.netG_AtoB.load_state_dict(torch.load(checkpoint_path_AtoB))
        # self.netG_BtoC.load_state_dict(torch.load(checkpoint_path_BtoC))

        self.fake_A_pool = utill.ImagePool(opt.pool_size)
        self.fake_B_pool = utill.ImagePool(opt.pool_size)
        
        # define loss
        self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionRec = torch.nn.L1loss()

        # init optimizer
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AtoB.parameters(),  \
                                                            self.netG_BtoA.parameters(),  \
                                                            self.netG_BtoC.parameters()))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), \
                                                            self.netD_B.parameters(), \
                                                            self.netD_C.parameters()))
                                                            # TODO conti...

    def set_input(self, input):
        self.image_sizes = input['A_sizes']

        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

        self.size = (int(self.image_sizes[0]), int(self.image_sizes[1]))

    def train(self):
        self.real_A = Variable(self.input_A)

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG_AtoB(self.real_A)
        self.fake_C = self.netG_BtoC(self.fake_B)

    def get_image_paths(self):
        return self.image_paths

    def get_image_sizes(self):
        return self.size

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        fake_C = util.tensor2im(self.fake_C.data)

        return OrderedDict([('original', real_A), ('restyled', fake_B), ('depth', fake_C)])