from __future__ import print_function, division
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import copy


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = scalar_name + "_" + str(idx)
            logger.add_scalar(scalar_name, value, global_step)


def save_images(logger, tag, image_outputs, global_step):
    """
    save images to tensorBoard
    """

    def get_center_frame(x):
        # x: [B, C, T, H, W] or [C, T, H, W] or [B, T, H, W] or [T, H, W] or [C, H, W] or [B, C, H, W]
        if isinstance(x, list):
            return [get_center_frame(xx) for xx in x]
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            if x.ndim == 5:  # [B, C, T, H, W]
                center = x.shape[2] // 2
                x = x[:, :, center, :, :]
            if x.ndim == 4:
                # [B, C, H, W]
                if x.shape[0] > 1:
                    x = x[0]
                # [C, T, H, W] or [B, T, H, W]
                elif x.shape[1] > 1:
                    center = x.shape[1] // 2
                    x = x[:, center, :, :]
            elif x.ndim == 3:
                return x
        return x

    def ensure_valid_image_tensor(img):
        """Ensure the tensor is in the correct format for TensorBoard"""
        try:
            if torch.is_tensor(img):
                # Print debug info for problematic tensors
                if img.numel() < 100:  # Very small tensors
                    print(f"Debug: Small tensor shape {img.shape}, dtype {img.dtype}")

                # Convert to numpy if needed
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype == torch.int64 or img.dtype == torch.int32:
                    img = img.float()

                # Ensure the tensor is in the correct range [0, 1] for float tensors
                if img.dtype == torch.float32 or img.dtype == torch.float64:
                    if img.max() > 1.0:
                        img = img / 255.0

                # Convert to numpy and ensure correct shape
                img = img.detach().cpu().numpy()

                # Handle different tensor shapes
                if img.ndim == 4:  # [B, C, H, W]
                    img = img[0]  # Take first batch
                elif img.ndim == 3:  # [C, H, W] or [H, W, C]
                    if img.shape[0] <= 3:  # [C, H, W]
                        pass  # Already correct format
                    elif img.shape[-1] <= 3:  # [H, W, C]
                        img = img.transpose(2, 0, 1)  # Convert to [C, H, W]
                    else:
                        # Handle single channel case
                        img = img[np.newaxis, :, :]  # Add channel dimension
                elif img.ndim == 2:  # [H, W]
                    img = img[np.newaxis, :, :]  # Add channel dimension

                # Ensure we have 3 channels for RGB
                if img.shape[0] == 1:  # Single channel
                    img = np.repeat(img, 3, axis=0)  # Repeat to make it RGB
                elif img.shape[0] > 3:  # More than 3 channels
                    img = img[:3]  # Take first 3 channels

                # Ensure values are in [0, 1] range
                img = np.clip(img, 0, 1)

                # Additional safety check for very small images
                if img.shape[1] < 8 or img.shape[2] < 8:
                    # Skip very small images that might cause issues
                    print(f"Warning: Skipping very small image with shape {img.shape}")
                    return None

                # Final shape check
                if img.shape[0] != 3:
                    print(f"Warning: Invalid channel count {img.shape[0]}, expected 3")
                    return None

            return img
        except Exception as e:
            print(f"Error processing image tensor: {e}")
            print(f"Tensor shape: {img.shape if hasattr(img, 'shape') else 'unknown'}")
            print(f"Tensor dtype: {img.dtype if hasattr(img, 'dtype') else 'unknown'}")
            return None

    for key, img in image_outputs.items():
        try:
            img = get_center_frame(img)
            if isinstance(img, list):
                for i, im in enumerate(img):
                    im = ensure_valid_image_tensor(im)
                    if im is not None:
                        logger.add_image(f"{tag}/{key}_{i}", im, global_step, dataformats='CHW')
            else:
                img = ensure_valid_image_tensor(img)
                if img is not None:
                    logger.add_image(f"{tag}/{key}", img, global_step, dataformats='CHW')
        except Exception as e:
            print(f"Error saving image {key}: {e}")
            continue


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(':')
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    # parse downscale rate (after :)
    downscale_rate = float(splits[1])
    print("Downscale learning rate at epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("Setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.sum_value = 0.
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.sum_value += x
        self.count += 1

    def mean(self):
        return self.sum_value / self.count


class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)
