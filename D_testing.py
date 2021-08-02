import argparse
import functools

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.data import DataLoader

from networks.sub_networks import DeblurringNet
from utils.data_processing import *
import torch.nn as nn
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_kernel_width', type=float, default=0.3)
    parser.add_argument('--min_kernel_width', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1.25e-5, help='initial learning rate')
    parser.add_argument('--num_iteration', type=int, default=1, help='')
    parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=20, help='number of epochs to update learning rate')
    parser.add_argument('--testset_dir', type=str, default='/home/dong/exp_data/GOPRO_Large/test/')

    return parser.parse_args()


def inference(test_loader, cfg):
    net = DeblurringNet(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)).to(
        cfg.device)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    pretrained_dict = torch.load('final_model/DeblurringNet_FT.pth')
    net.load_state_dict(pretrained_dict['deblurring_state_dict'])

    psnr_list = []
    ssim_list = []
    # torch.cuda.empty_cache()
    with torch.no_grad():
        for idx_iter, (img_hr, img_blur) in enumerate(test_loader):
            img_hr = img_hr.to(cfg.device)
            img_blur = img_blur.to(cfg.device)

            deblurred_rgb, _ = net(img_blur)

            deblurred_rgb = torch.clamp(deblurred_rgb, -1, 1)

            deblurred_rgb = (deblurred_rgb + 1) / 2

            img_hr = (img_hr + 1) / 2

            deblurring_output = deblurred_rgb[0, ...].detach().permute(1, 2, 0).cpu().numpy()

            hr_numpy = img_hr[0, ...].detach().permute(1, 2, 0).cpu().numpy()

            psnr = peak_signal_noise_ratio(deblurring_output, hr_numpy)
            psnr_list.append(psnr)
            print(psnr)

            ssim = structural_similarity(deblurring_output, hr_numpy, multichannel=True)
            ssim_list.append(ssim)
            print(ssim)
            torch.cuda.empty_cache()
    print()
    print(np.mean(np.array(psnr_list)))
    print(np.mean(np.array(ssim_list)))


def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
    inference(test_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
