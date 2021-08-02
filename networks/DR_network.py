import functools


import torch

from collections import OrderedDict

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from networks.losses import PerceptualLoss
from networks.sub_networks import DeblurringNet, BlurringNet
from utils.data_processing import save_ckpt, weights_init_xavier


class Whole_Network:

    def __init__(self, cfg):
        self.isTrain = cfg.isTrain


        self.deblurring_net = DeblurringNet(norm_layer=self.get_norm_layer()).to(cfg.device)
        self.blurring_net = BlurringNet(norm_layer=self.get_norm_layer()).to(cfg.device)


        self.cfg = cfg
        self.Tensor = torch.cuda.FloatTensor if 'cuda' in cfg.device else torch.Tensor

        if self.isTrain:
            # initialize optimizers
            self.optimizer = torch.optim.Adam(list(self.deblurring_net.parameters()), lr=cfg.lr)

            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.n_steps, cfg.gamma)


            self.perceptual_loss = PerceptualLoss()
            self.perceptual_loss.initialize(nn.MSELoss())

        print('---------- Networks initialized -------------')
        print(sum(p.numel() for p in self.deblurring_net.parameters()))
        print(sum(p.numel() for p in self.blurring_net.parameters()))

        self.weight_initialization()

        print('-----------------------------------------------')

    def weight_initialization(self):
        self.deblurring_net.apply(weights_init_xavier)
        self.blurring_net.apply(weights_init_xavier)

    def log_initialization(self):
        self.writer = SummaryWriter(self.cfg.log_dir)

    def set_input(self, HR_sharp_patch, HR_blurry_patch):
        self.HR_sharp_patch = HR_sharp_patch
        self.HR_blurry_patch = HR_blurry_patch

    def forward(self):
        self.deblurred_rgb, _ = self.deblurring_net.forward(self.HR_blurry_patch)

        self.reblurred_rgb = self.blurring_net.forward(self.HR_blurry_patch, self.deblurred_rgb)

    # no backprop gradients
    def test(self):
        self.deblurred_rgb, _ = self.deblurring_net.forward(self.HR_blurry_patch)

        return self.deblurred_rgb

    def backward(self):

        # Second, G(A) = B
        self.loss_deblurring_pixel = self.perceptual_loss.get_loss(self.deblurred_rgb, self.HR_sharp_patch)

        self.loss_blurring_pixel = self.perceptual_loss.get_loss(self.HR_blurry_patch, self.reblurred_rgb)

        self.loss = self.loss_deblurring_pixel * self.cfg.lambda_A + self.loss_blurring_pixel * self.cfg.lambda_B

        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_norm_layer(self, norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    def get_current_errors(self):
        return OrderedDict([('loss_deblurring_pixel', self.loss_deblurring_pixel.item()),
                            ('loss_blurring_pixel', self.loss_blurring_pixel.item()),
                            ])

    def write_training_log(self, idx_epoch, idx_iter, epoch_size):

        self.writer.add_scalar('deblurring/loss_deblurring_pixel', self.loss_deblurring_pixel.item(),
                               idx_epoch * epoch_size + idx_iter)

        self.writer.add_scalar('blurring/loss_blurring_pixel', self.loss_blurring_pixel.item(),
                               idx_epoch * epoch_size + idx_iter)

        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'],
                               idx_epoch * epoch_size + idx_iter)

    def write_validating_log(self, psnr, idx_epoch):
        self.writer.add_scalar('psnr', psnr, idx_epoch + 1)

    def close_writer(self):
        self.writer.close()

    def save(self, idx_epoch):
        save_ckpt({
            'epoch': idx_epoch + 1,
            'deblurring_state_dict': self.deblurring_net.state_dict(),
        }, save_path='deblurring_models/', filename='DeblurringNet_FT.pth')

    def load_state(self):

        pretrained_dict = torch.load('deblurring_models/DeblurringNet_NF.pth')
        self.deblurring_net.load_state_dict(pretrained_dict['deblurring_state_dict'])

        pretrained_dict2 = torch.load('blurring_models/ReBlurringNet.pth')
        self.blurring_net.load_state_dict(pretrained_dict2['blurring_state_dict'])
