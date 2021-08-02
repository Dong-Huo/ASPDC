import math

from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
# from skimage import metrics
from torch.nn import init
import cv2
import torch.nn.functional as F
import albumentations as albu


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg):
        super(TrainSetLoader, self).__init__()

        self.transforms = get_transforms(cfg.patch_size)
        self.normalize = get_normalize()

        folder_path = os.listdir(dataset_dir)

        self.file_list = []

        for folder in folder_path:
            if "GOPR" in folder:
                image_path = os.listdir(os.path.join(dataset_dir, folder, "sharp"))
                for image in image_path:
                    if "png" in image:
                        self.file_list.append(os.path.join(dataset_dir, folder, "sharp", image))

    def __getitem__(self, index):

        img_hr = cv2.imread(self.file_list[index])
        img_blur = cv2.imread(self.file_list[index].replace("sharp", "blur"))
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)

        img_hr, img_blur = self.transforms(img_hr, img_blur)
        img_hr, img_blur = self.normalize(img_hr, img_blur)

        return toTensor(img_hr), toTensor(img_blur)

    def __len__(self):
        return len(self.file_list)


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TestSetLoader, self).__init__()
        folder_path = os.listdir(dataset_dir)

        self.file_list = []

        self.normalize = get_normalize()

        for folder in folder_path:

            if "GOPR" in folder:
                HR_path = os.path.join(dataset_dir, folder, "sharp")

                HR_im = os.listdir(HR_path)

                for im in HR_im:

                    if "png" in im:
                        image_path = os.path.join(dataset_dir, folder, "sharp", im)
                        self.file_list.append(image_path)

        print(len(self.file_list))

    def __getitem__(self, index):

        img_hr = cv2.imread(self.file_list[index])
        img_blur = cv2.imread(self.file_list[index].replace("sharp", "blur"))

        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)

        img_hr, img_blur = self.normalize(img_hr, img_blur)

        return toTensor(img_hr), toTensor(img_blur)

    def __len__(self):
        return len(self.file_list)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float()


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def get_transforms(size: int, scope: str = 'geometric', crop='random'):
    augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                    albu.ElasticTransform(),
                                    albu.OpticalDistortion(),
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.5),
                                    albu.OneOf([
                                        albu.RGBShift(),
                                        albu.HueSaturationValue(),
                                    ], p=0.5),
                                    ]),
            'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                     albu.ShiftScaleRotate(always_apply=True),
                                     albu.Transpose(always_apply=True),
                                     albu.OpticalDistortion(always_apply=True),
                                     albu.ElasticTransform(always_apply=True),
                                     ])
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]
    pad = albu.PadIfNeeded(size, size)

    pipeline = albu.Compose([aug_fn, crop_fn, pad], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process
