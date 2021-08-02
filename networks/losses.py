import torch
import torch.nn as nn

import torchvision.transforms as transforms


class ContentLoss:
    def __init__(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class PerceptualLoss():

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0

        for i in range(fakeIm.shape[0]):
            fakeIm[i, :, :, :] = self.transform(fakeIm[i, :, :, :])
            realIm[i, :, :, :] = self.transform(realIm[i, :, :, :])

        pixel_loss = nn.MSELoss()(fakeIm, realIm)

        return pixel_loss

        # return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)
