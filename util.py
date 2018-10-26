from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import ntpath
import cv2
import random
from torch.autograd import Variable


def tensor2im(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(np.uint8)

    elif image_numpy.shape[0] == 1:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 65535.0
        return image_numpy.astype(np.uint16)

def save_image_color(image_numpy, image_path):
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_image_depth(image_numpy, image_path):
    cv2.imwrite(image_path,image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_images(results_dir, visuals, image_path, size=None):
    image_dir = results_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im in visuals.items():

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if size!=None:
            im = cv2.resize(im, size)

        if label == 'depth':
            save_image_depth(im, save_path)
        else:
            save_image_color(im, save_path)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
