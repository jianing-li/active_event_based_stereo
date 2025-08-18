"""
Function: ActiveEventNet is a lightweight neural network that performs active event-based stereo matching.
Jianing Li, lijianing@pku.edu.cn.
"""

from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__
from utils import *
import cv2
import time
import matplotlib.colors as mcolors


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='ActiveEventNet')
parser.add_argument('--model', default='AENet2D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', default=192, help='maximum disparity')
parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--save_frame', default=0, help='save prediction depth frames')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('./predictions', exist_ok=True)

    total_time = 0
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        left_filenames = sample['left_filename']
        disp_gts = sample['disparity']


        scalar_outputs, disp_ests, count_time = test_sample(sample)
        total_time = total_time + count_time

        # save prediction disparity images.
        for idx, disp_est in enumerate(disp_ests[-1]):
            assert len(disp_est.shape) == 2

            if float(args.save_frame) == 1:
                file_elements = left_filenames[idx].split('/')
                output_file = './predictions/{}/'.format(file_elements[-3])
                os.makedirs(output_file, exist_ok=True)
                filename = output_file + '{}_{}'.format(file_elements[-2], file_elements[-1].replace('.npy', '_nomask.png'))
                ground_truth_filename = output_file + '{}_{}'.format(file_elements[-2], file_elements[-1].replace('.npy', '_truth.png'))

                # save disparity error map.
                device = disp_est.device
                disp_gt_tensor = disp_gts[idx].to(device)
                disp_est_tensor = disp_est.to(device)
                mask_tensor = ((disp_gt_tensor < args.maxdisp) & (disp_gt_tensor > 0)).to(device)
                if disp_gt_tensor.ndim == 2:
                    disp_gt_tensor = disp_gt_tensor.unsqueeze(0)
                    disp_est_tensor = disp_est_tensor.unsqueeze(0)
                    mask_tensor = mask_tensor.unsqueeze(0)
                EPE_value = EPE_metric(disp_est_tensor, disp_gt_tensor, mask_tensor)
                D1_value = D1_metric(disp_est_tensor, disp_gt_tensor, mask_tensor)
                disp_gt = disp_gt_tensor.squeeze(0).cpu().numpy()
                disp_est_np = disp_est_tensor.squeeze(0).cpu().numpy()

                mask = (disp_gt > 0) & (disp_gt < args.maxdisp)
                disp_est_np[~mask] = disp_gt[~mask]
                disp_error = np.abs(disp_gt - disp_est_np)
                max_error = disp_error.max() if disp_error.max() > 0 else 1.0
                error_norm = disp_error.astype(np.float32) / max_error
                gamma = 0.5
                error_gamma = np.power(error_norm, gamma)
                custom_colors = [
                    (1.0, 1.0, 1.0),  # white
                    (0.85, 0.95, 1.0),  # light blue
                    (0.6, 0.85, 1.0),  # sky blue
                    (0.3, 0.7, 1.0),  # blue
                    (1.0, 0.85, 0.5),  # orange
                    (0.9, 0.0, 0.0),  # red
                    (0.5, 0.0, 0.0)  # dark red
                ]
                custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_hot", custom_colors)
                error_colormap = (custom_cmap(error_gamma)[:, :, :3] * 255).astype(np.uint8)
                error_colormap = cv2.cvtColor(error_colormap, cv2.COLOR_RGB2BGR)
                error_colormap[~mask] = [255, 255, 255]
                disp_error_filename = output_file + '{}_{}'.format(file_elements[-2], file_elements[-1].replace('.npy', '_error.png'))
                cv2.imwrite(disp_error_filename, error_colormap)
                # cv2.putText(error_colormap, f"EPE:{EPE_value:.3f}, D1-all:{D1_value:.3f}", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.87, (0, 0, 0), 2, cv2.LINE_AA) # DAVIS, 346*260
                cv2.putText(error_colormap, f"EPE:{EPE_value:.3f}, D1-all:{D1_value:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4, cv2.LINE_AA) # RealSense, 640*480
                disp_error_txt_filename = output_file + '{}_{}'.format(file_elements[-2], file_elements[-1].replace('.npy', '_error_text.png'))
                cv2.imwrite(disp_error_txt_filename, error_colormap)

                # save ground truth
                disp_gt = tensor2numpy(disp_gts[idx])
                disp_gt_map = cv2.applyColorMap(cv2.convertScaleAbs(disp_gt, alpha=15), cv2.COLORMAP_JET) # 10
                cv2.imwrite(ground_truth_filename, disp_gt_map)

                # save disparity maps.
                disp_est = tensor2numpy(disp_est)
                # disparity_map = cv2.applyColorMap(cv2.convertScaleAbs(disp_est, alpha=15), cv2.COLORMAP_JET)  # DAVIS346
                disparity_map = cv2.applyColorMap(cv2.convertScaleAbs(disp_est, alpha=10), cv2.COLORMAP_JET)  # Realsense
                cv2.imwrite(filename, disparity_map)
                print('{} has been done!'.format(filename))

            else:
                print('Test filename is {}'.format(left_filenames[idx]))

        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs

    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)

    avg_time = total_time/(len(TestImgLoader)*args.batch_size)
    print('The average inference time is {}'.format(avg_time))
    print("Done!")


@make_nograd_func
def test_sample(sample):
    model.eval()

    start_time = time.time()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    end_time = time.time()
    count_time = end_time - start_time

    disp_gt = sample['disparity'].cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    # Evaluation performance.
    scalar_outputs = {}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["RMSE"] = [RMSE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(scalar_outputs), disp_ests, count_time


if __name__ == '__main__':
    test(args)
