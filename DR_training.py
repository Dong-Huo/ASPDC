import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from networks.DR_network import Whole_Network
from networks.losses import PerceptualLoss

import argparse
import time

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from utils.data_processing import *
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--num_iteration', type=int, default=1, help='')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--lambda_A', type=float, default=1.0, help='weight of pixel loss')
    parser.add_argument('--lambda_B', type=float, default=0.1, help='weight of cycle loss')
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='deblurring_log')
    parser.add_argument('--trainset_dir', type=str, default='/home/dong/exp_data/GOPRO_Large/train/')

    return parser.parse_args()


def train(train_loader, cfg):
    cudnn.benchmark = True

    perceptual_loss = PerceptualLoss()
    perceptual_loss.initialize(nn.MSELoss())
    w_net = Whole_Network(cfg)

    previous_epoch = 0
    w_net.load_state(previous_epoch)

    w_net.log_initialization()

    for idx_epoch in range(previous_epoch, cfg.n_epochs):
        start_time = 0
        torch.cuda.empty_cache()

        for idx_iter, (img_hr_sharp, img_hr_blur) in enumerate(train_loader):
            img_hr_sharp = img_hr_sharp.to(cfg.device)
            img_hr_blur = img_hr_blur.to(cfg.device)

            w_net.set_input(img_hr_sharp, img_hr_blur)
            w_net.optimize_parameters()

            if idx_iter % 10 == 0:
                w_net.write_training_log(idx_epoch, idx_iter, len(train_loader))

            print("Epochs: " + str(idx_epoch + 1) + ", step: " + str(idx_iter))

            pixel_loss = perceptual_loss.get_loss(img_hr_blur, img_hr_sharp)

            print("loss_di_deblur:" + str(pixel_loss.data.cpu().numpy().tolist()))

            error_dict = w_net.get_current_errors()
            for item in error_dict:
                print(item + ": " + str(error_dict[item]))

            print(str(round(time.time() - start_time, 3)) + "s")
            start_time = time.time()
        torch.cuda.empty_cache()

        if (idx_epoch + 1) % 10 == 0:
            w_net.save(idx_epoch)

    w_net.close_writer()


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
