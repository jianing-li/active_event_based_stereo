import os
import random
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from utils.event_representation import make_color_histo
import torch

def log_corrupted_file(filename, log_path="corrupted_files.txt"):
    with open(log_path, "a") as f:
        f.write(filename + "\n")

class Active_Event_Stereo_Dataset(Dataset):
    def __init__(self, datapath, list_filename, training, temporal_sequence_length=3):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.temporal_sequence_length = temporal_sequence_length
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_events(self, filename):
        events = np.load(filename, allow_pickle=True)

        event_image = make_color_histo(events, width=640, height=480)
        event_image = Image.fromarray(cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB))

        return event_image

    def load_disp(self, filename):

        depth_value = np.load(filename)

        # RealSense D435i camera parameters.
        fx = 638.0615844726562  # lense focal length
        baseline = 0.04993824660778046  # distance in mm between the two cameras
        units = 1  # depth units

        # convert depth value to disparity data.
        disparity = np.zeros(shape=depth_value.shape).astype(float)
        disparity[depth_value > 0] = (fx * baseline) / (units * depth_value[depth_value > 0])
        # disparity_data = np.array(disparity, dtype=np.float32) / 256.
        disparity_data = np.array(disparity, dtype=np.float32)

        return disparity_data

    def get_temporal_indices(self, index):
        """get temporal index"""
        total_frames = len(self.left_filenames)

        # compute the center of temporal sequence
        half_length = self.temporal_sequence_length // 2
        start_idx = max(0, index - half_length)
        end_idx = min(total_frames, index + half_length + 1)

        # If the time-series window is insufficient, pad it from both ends.
        if end_idx - start_idx < self.temporal_sequence_length:
            if start_idx == 0:
                end_idx = min(total_frames, start_idx + self.temporal_sequence_length)
            else:
                start_idx = max(0, end_idx - self.temporal_sequence_length)

        # generat temporal indices
        temporal_indices = list(range(start_idx, end_idx))
        while len(temporal_indices) < self.temporal_sequence_length:
            temporal_indices.append(temporal_indices[-1])

        if len(temporal_indices) > self.temporal_sequence_length:
            temporal_indices = temporal_indices[:self.temporal_sequence_length]

        return temporal_indices

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        temporal_indices = self.get_temporal_indices(index)

        left_imgs = []
        right_imgs = []
        disparities = []

        for idx in temporal_indices:
            if self.left_filenames[idx][-4:] == '.npy':
                left_img = self.load_events(os.path.join(self.datapath, self.left_filenames[idx]))
                right_img = self.load_events(os.path.join(self.datapath, self.right_filenames[idx]))
            else:
                left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[idx]))
                right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[idx]))

            if self.disp_filenames:  # has disparity ground truth
                disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[idx]))
            else:
                disparity = None

            left_imgs.append(left_img)
            right_imgs.append(right_img)
            if disparity is not None:
                disparities.append(disparity)
        if self.training:
            w, h = left_imgs[0].size
            crop_w, crop_h = 512, 256
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_imgs_cropped = []
            right_imgs_cropped = []
            disparities_cropped = []
            for i in range(len(left_imgs)):
                left_img_cropped = left_imgs[i].crop((x1, y1, x1 + crop_w, y1 + crop_h))
                right_img_cropped = right_imgs[i].crop((x1, y1, x1 + crop_w, y1 + crop_h))
                left_imgs_cropped.append(left_img_cropped)
                right_imgs_cropped.append(right_img_cropped)
                if disparities:
                    disparity_cropped = disparities[i][y1:y1 + crop_h, x1:x1 + crop_w]
                    disparities_cropped.append(disparity_cropped)
            processed = get_transform()
            left_tensors = []
            right_tensors = []
            for i in range(len(left_imgs_cropped)):
                left_tensor = processed(left_imgs_cropped[i])
                right_tensor = processed(right_imgs_cropped[i])
                left_tensors.append(left_tensor)
                right_tensors.append(right_tensor)
            left_temporal = torch.stack(left_tensors, dim=0)
            right_temporal = torch.stack(right_tensors, dim=0)
            left_temporal = left_temporal.permute(1, 0, 2, 3)
            right_temporal = right_temporal.permute(1, 0, 2, 3)
            if disparities_cropped:
                center_idx = len(disparities_cropped) // 2
                disparity_final = torch.from_numpy(disparities_cropped[center_idx]).float()
            else:
                disparity_final = None
            return {"left": left_temporal,
                    "right": right_temporal,
                    "disparity": disparity_final}
        else:
            w, h = left_imgs[0].size
            crop_w, crop_h = 640, 480
            top_pad = crop_h - h
            right_pad = crop_w - w
            left_imgs_cropped = []
            right_imgs_cropped = []
            disparities_cropped = []
            for i in range(len(left_imgs)):
                left_img_cropped = left_imgs[i].crop((w - crop_w, h - crop_h, w, h))
                right_img_cropped = right_imgs[i].crop((w - crop_w, h - crop_h, w, h))
                left_imgs_cropped.append(left_img_cropped)
                right_imgs_cropped.append(right_img_cropped)
                if disparities:
                    disparity_cropped = disparities[i][h - crop_h:h, w - crop_w: w]
                    disparities_cropped.append(disparity_cropped)
            processed = get_transform()
            left_tensors = []
            right_tensors = []
            for i in range(len(left_imgs_cropped)):
                left_tensor = processed(left_imgs_cropped[i]).numpy()
                right_tensor = processed(right_imgs_cropped[i]).numpy()
                left_tensors.append(left_tensor)
                right_tensors.append(right_tensor)
            left_temporal = np.stack(left_tensors, axis=0)
            right_temporal = np.stack(right_tensors, axis=0)
            left_temporal = np.transpose(left_temporal, (1, 0, 2, 3))
            right_temporal = np.transpose(right_temporal, (1, 0, 2, 3))
            if disparities_cropped:
                center_idx = len(disparities_cropped) // 2
                disparity_final = torch.from_numpy(disparities_cropped[center_idx]).float()
            else:
                disparity_final = None
            return {"left": left_temporal,
                    "right": right_temporal,
                    "disparity": disparity_final,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "left_filename": self.left_filenames[index]}



class DAVIS346_Stereo_Dataset(Dataset):
    def __init__(self, datapath, list_filename, training, temporal_sequence_length=3):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.temporal_sequence_length = temporal_sequence_length
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]

        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        try:
            rgb_image = Image.open(filename).convert('RGB')
        except Exception as e:
            log_corrupted_file(filename)
            print(f"[Error] Failed to load events: {filename}, error: {e}")
        return rgb_image

    def load_events(self, filename):
        try:
            events = np.load(filename, allow_pickle=True)
        except Exception as e:
            log_corrupted_file(filename)
            print(f"[Error] Failed to load events: {filename}, error: {e}")
            return None
        event_image = make_color_histo(events, width=346, height=260)
        event_image = Image.fromarray(cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB))
        return event_image

    def load_disp(self, filename):
        try:
            depth_value = np.load(filename)
        except Exception as e:
            log_corrupted_file(filename)
            print(f"[Error] Failed to load depth: {filename}, error: {e}")
            return None
        # Stereo DAVIS346 camera parameters.
        fx = 340.60769771  # lense focal length
        baseline = 59.42762366 # distance in mm between the two cameras
        units = 1  # depth units
        # convert depth value to disparity data.
        disparity = np.zeros(shape=depth_value.shape).astype(float)
        disparity[depth_value > 0.05] = (fx * baseline) / (units * depth_value[depth_value > 0.05])
        disparity_data = np.array(disparity, dtype=np.float32)
        return disparity_data

    def get_temporal_indices(self, index):
        total_frames = len(self.left_filenames)
        half_length = self.temporal_sequence_length // 2
        start_idx = max(0, index - half_length)
        end_idx = min(total_frames, index + half_length + 1)
        if end_idx - start_idx < self.temporal_sequence_length:
            if start_idx == 0:
                end_idx = min(total_frames, start_idx + self.temporal_sequence_length)
            else:
                start_idx = max(0, end_idx - self.temporal_sequence_length)
        temporal_indices = list(range(start_idx, end_idx))
        while len(temporal_indices) < self.temporal_sequence_length:
            temporal_indices.append(temporal_indices[-1])
        if len(temporal_indices) > self.temporal_sequence_length:
            temporal_indices = temporal_indices[:self.temporal_sequence_length]
        return temporal_indices

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        temporal_indices = self.get_temporal_indices(index)
        left_imgs = []
        right_imgs = []
        disparities = []
        for idx in temporal_indices:
            if self.left_filenames[idx][-4:] == '.npy':
                left_img = self.load_events(os.path.join(self.datapath, self.left_filenames[idx]))
                right_img = self.load_events(os.path.join(self.datapath, self.right_filenames[idx]))
            else:
                left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[idx]))
                right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[idx]))
            if self.disp_filenames:
                disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[idx]))
            else:
                disparity = None
            left_imgs.append(left_img)
            right_imgs.append(right_img)
            disparities.append(disparity)
        if any(img is None for img in left_imgs) or any(img is None for img in right_imgs) or (disparities and any(d is None for d in disparities)):
            print(f"[Warning] Skipping index {index} due to corrupted file.")
            return self.__getitem__((index + 1) % len(self))
        if self.training:
            w, h = left_imgs[0].size
            crop_w, crop_h = 320, 256
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_imgs_cropped = []
            right_imgs_cropped = []
            disparities_cropped = []
            for i in range(len(left_imgs)):
                left_img_cropped = left_imgs[i].crop((x1, y1, x1 + crop_w, y1 + crop_h))
                right_img_cropped = right_imgs[i].crop((x1, y1, x1 + crop_w, y1 + crop_h))
                left_imgs_cropped.append(left_img_cropped)
                right_imgs_cropped.append(right_img_cropped)
                if disparities[i] is not None:
                    disparity_cropped = disparities[i][y1:y1 + crop_h, x1:x1 + crop_w]
                    disparities_cropped.append(disparity_cropped)
            processed = get_transform()
            left_tensors = []
            right_tensors = []
            for i in range(len(left_imgs_cropped)):
                left_tensor = processed(left_imgs_cropped[i])
                right_tensor = processed(right_imgs_cropped[i])
                left_tensors.append(left_tensor)
                right_tensors.append(right_tensor)
            left_temporal = torch.stack(left_tensors, dim=0)  # [T, C, H, W]
            right_temporal = torch.stack(right_tensors, dim=0)
            if disparities_cropped:
                center_idx = len(disparities_cropped) // 2
                disparity_final = disparities_cropped[center_idx]
            else:
                disparity_final = None
            return {"left": left_temporal,
                    "right": right_temporal,
                    "disparity": disparity_final}
        else:
            w, h = left_imgs[0].size
            # pad to size 352x272, DAVIS raw image size 346x260.
            top_pad = 272 - h
            right_pad = 352 - w
            assert top_pad > 0 and right_pad > 0
            left_imgs_padded = []
            right_imgs_padded = []
            disparities_padded = []
            processed = get_transform()
            for i in range(len(left_imgs)):
                left_img = processed(left_imgs[i]).numpy()
                right_img = processed(right_imgs[i]).numpy()
                left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                left_imgs_padded.append(left_img)
                right_imgs_padded.append(right_img)
                if disparities[i] is not None:
                    disparity = disparities[i]
                    disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                    disparities_padded.append(disparity)
            left_temporal = np.stack(left_imgs_padded, axis=0)  # [T, C, H, W]
            right_temporal = np.stack(right_imgs_padded, axis=0)
            if disparities_padded:
                center_idx = len(disparities_padded) // 2
                disparity_final = disparities_padded[center_idx]
                return {"left": left_temporal,
                        "right": right_temporal,
                        "disparity": disparity_final,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}
            else:
                return {"left": left_temporal,
                        "right": right_temporal,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}