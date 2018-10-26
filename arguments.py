import argparse
import os
import util
import torch


class Arguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_directory', default="./Examples", help='path to the directory containing the images')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='the directory that contains the checkpoints')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--style_results_dir', type=str, default='./style_results/', help='saves style results here.')
        self.parser.add_argument('--depth_results_dir', type=str, default='./depth_results/', help='saves depth results here.')
        self.parser.add_argument('--how_many', type=int, default=5000, help='how many test images to run')
        self.parser.add_argument('--batchSize', type=int, default=480, help='how many test images to run')
        self.parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
        self.parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
        self.parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()

        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

        # set gpu ids
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])

        args = vars(self.args)


        print('------------ Arguments -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------------------------------')

        return self.args
