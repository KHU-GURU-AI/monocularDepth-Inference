import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []

    transform_list.append(transforms.Lambda(
    lambda img: __scale_to_256_factor(img)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_to_256_factor(img):
    # return img.resize((412, 120), Image.BICUBIC)
    # return img.resize((1024, 256), Image.BICUBIC)
    return img.resize((512, 128), Image.BICUBIC)