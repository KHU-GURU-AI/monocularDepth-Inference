import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2

class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.data_directory
        self.dir_A = os.path.join(opt.data_directory)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_size = A_img.size

        A = self.transform(A_img)
        input_nc = 3

        return {'A': A, 'A_paths': A_path, 'A_sizes': A_size}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'TestDataset'


class RealDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.data_directory
        self.dir_real = os.path.join(opt.data_directory, "real")

        self.A_paths = sorted(make_dataset(self.dir_real))

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_size = A_img.size

        A = self.transform(A_img)

        return {'A': A, 'A_paths': A_path, 'A_sizes': A_size}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'RealDataset'


class SyntheticDataset(BaseDataset):
    # def initialize(self, opt):
    #     self.opt = opt
    #     self.root = opt.data_directory
    #     self.dir_syn = os.path.join(opt.data_directory, "fifa")
    #     self.depth_postfix = '_d'
    #
    #     self.B_paths = []
    #     self.C_paths = []
    #
    #     syn_paths = make_dataset(self.dir_syn)
    #
    #     # split Syn and Depth
    #     for path in syn_paths:
    #         if path.split('/')[-1].split('.')[-2][-2:] == self.depth_postfix:
    #             self.C_paths.append(path)
    #         else:
    #             self.B_paths.append(path)
    #
    #     self.B_paths = sorted(self.B_paths)
    #     self.C_paths = sorted(self.C_paths)
    #
    #     self.transform = get_transform(opt)
    #
    # def __getitem__(self, index):
    #     B_path = self.B_paths[index]
    #     B_img = Image.open(B_path).convert('RGB')
    #     B_size = B_img.size
    #
    #     B = self.transform(B_img)
    #
    #     C_path = self.C_paths[index]
    #     C_img = Image.open(C_path).convert('L')
    #     C_size = C_img.size
    #
    #     C = self.transform(C_img)
    #
    #     return {'B': B, 'B_paths': B_path, 'B_sizes': B_size, \
    #             'C': C, 'C_paths': C_path, 'C_sizes': C_size}
    #
    # def __len__(self):
    #     return len(self.B_paths)
    #
    # def name(self):
    #     return 'SyntheticDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.data_directory
        self.dir_B = os.path.join(opt.data_directory, "vkitti_1.3.1_rgb")
        self.dir_C = os.path.join(opt.data_directory, "vkitti_1.3.1_depthgt")

        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)

        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        B_size = B_img.size

        B = self.transform(B_img)

        C_path = self.C_paths[index]
        C_img = Image.open(C_path).point(lambda i: i*(1./256)).convert('L')
        C_size = C_img.size

        # C_img.save("./input_check/1.jpg")

        C = self.transform(C_img)

        return {'B': B, 'B_paths': B_path, 'B_sizes': B_size, \
                'C': C, 'C_paths': C_path, 'C_sizes': C_size}

    def __len__(self):
        return len(self.B_paths)

    def name(self):
        return 'TestDataset'
