import os
import random
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from utils.event_representation import make_color_histo

class Active_Event_Stereo_Dataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
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

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        if self.left_filenames[index][-4:] == '.npy':

            left_img = self.load_events(os.path.join(self.datapath, self.left_filenames[index]))
            right_img = self.load_events(os.path.join(self.datapath, self.right_filenames[index]))
        else:
            left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
            right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))


        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}


        else:
            w, h = left_img.size
            crop_w, crop_h = 640, 480
            #crop_w, crop_h = 656, 496 # Realsense D435i
            # crop_w, crop_h = 352, 272  # 346*260
            top_pad = crop_h - h
            right_pad = crop_w - w

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))


            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            if disparity is not None:
                disparity = disparity[h - crop_h:h, w - crop_w: w]
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}

            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}


class DAVIS346_Stereo_Dataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
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

        event_image = make_color_histo(events, width=346, height=260)
        event_image = Image.fromarray(cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB))

        return event_image

    def load_disp(self, filename):

        depth_value = np.load(filename)

        # Stereo DAVIS346 camera parameters.
        fx = 340.60769771  # lense focal length
        baseline = 59.42762366 # distance in mm between the two cameras
        units = 1  # depth units

        # convert depth value to disparity data.
        disparity = np.zeros(shape=depth_value.shape).astype(float)
        #print('Depth value is {}!'.format(depth_value))


        disparity[depth_value > 0.05] = (fx * baseline) / (units * depth_value[depth_value > 0.05])
        disparity_data = np.array(disparity, dtype=np.float32)
        #print('Disparity data is {}!'.format(disparity_data))

        return disparity_data


    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        if self.left_filenames[index][-4:] == '.npy':

            left_img = self.load_events(os.path.join(self.datapath, self.left_filenames[index]))
            right_img = self.load_events(os.path.join(self.datapath, self.right_filenames[index]))
        else:
            left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
            right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))


        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 320, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}


        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 352x272, DAVIS raw image size 346x260.
            top_pad = 272 - h
            right_pad = 352 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}

            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}