"""
Class that handles the AFFACT transformations
"""
import math

import bob.io.image
import bob.ip.base
import matplotlib
import numpy as np
from PIL import Image
import random

from torchvision.transforms.functional import to_tensor


class ValTransformer():
    """
    Apply AFFACT transformations (scale, rotate, shift, blur), gamma and temperature to image for training
    """

    def __init__(self, config):
        """
        Initialization
        :param config: training configuration file
        """
        if not config:
            raise Exception("No Config defined")
        self.config = config



    def __call__(self, sample):
        """
        Transform operations of AFFACT (scale, rotate, shift, blur, gamma) and temperature
        :param sample: dict containing image, landmarks, bounding boxes, index
        :return: torch tensor of transformed image
        """
        matplotlib.use('Agg')

        image, landmarks, bounding_boxes, index = sample['image'], sample['landmarks'], sample['bounding_boxes'], sample['index']

        # Calculate bounding box based on landmarks according to the AFFACT paper
        if self.config.dataset.bounding_box_mode == 0:
            t_eye_left = np.array((landmarks[0], landmarks[1]))
            t_eye_right = np.array((landmarks[2], landmarks[3]))
            t_mouth_left = np.array((landmarks[4], landmarks[5]))
            t_mouth_right = np.array((landmarks[6], landmarks[7]))

            t_eye = (t_eye_left + t_eye_right) / 2
            t_mouth = (t_mouth_left + t_mouth_right) / 2
            d = np.linalg.norm(t_eye - t_mouth)
            w = h = 5.5 * d
            alpha = math.degrees(np.arctan2((t_eye_right[1] - t_eye_left[1]), (t_eye_right[0] - t_eye_left[0])))

            bbx = [t_eye[0] - 0.5 * w,
                   t_eye[1] - 0.45 * h,
                   t_eye[0] + 0.5 * w,
                   t_eye[1] + 0.55 * h,
                   alpha]

        # If no landmarks provided, bounding boxes from dataset or face detector with rotation angle = 0
        else:
            bbx = [
                bounding_boxes[0],
                bounding_boxes[1],
                bounding_boxes[0] + bounding_boxes[2],
                bounding_boxes[1] + bounding_boxes[3],
                0
            ]

        # Define Crop size
        crop_size = [self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y]

        # Calculate Scale factor
        scale = min(crop_size[0] / (bbx[2] - bbx[0]), crop_size[1] / (bbx[3] - bbx[1]))
        scale *= self.config.preprocessing.transformation.val_scale

        # Extract rotation angle from bounding box
        #rotation_angle = bbx[4]
        rotation_angle = 0

        # Calculate crop center
        crop_center = [crop_size[0] / 2., crop_size[1] / 2. ]

        # Define an input mask
        input_mask = np.ones((image.shape[1], image.shape[2]), dtype=bool)

        # Define output mask
        out_mask = np.ones((crop_size[0], crop_size[1]), dtype=bool)

        # Calculate Center of bounding box
        center = (bbx[1] + (bbx[3] - bbx[1]) / 2., bbx[0] + (bbx[2] - bbx[0]) / 2.)

        # Empty numpy ndarray (serves as placeholder for new image)
        placeholder_out = np.ones((3, self.config.preprocessing.transformation.crop_size.x, self.config.preprocessing.transformation.crop_size.y))
        placeholder_out[placeholder_out > 0] = 0

        # define geometric normalization
        geom = bob.ip.base.GeomNorm(rotation_angle, scale, crop_size, crop_center)

        # Channel-wise application of geonorm and extrapolation of mask
        for i in range(0, 3):
            in_slice = image[i]
            out_slice = np.ones((crop_size[0], crop_size[1]))
            out_slice = out_slice.astype(np.float)
            x = geom.process(in_slice, input_mask, out_slice, out_mask, center)
            try:
                bob.ip.base.extrapolate_mask(out_mask, out_slice)
            except:
                pass

            # Fill channel
            placeholder_out[i] = out_slice

        #placeholder_out = np.transpose(placeholder_out, (2, 0, 1))

        # to create a numpy array of shape H x W x C
        placeholder_out = np.transpose(placeholder_out, (1, 2, 0))

        # convert each pixel to uint8
        placeholder_out = placeholder_out.astype(np.uint8)

        # to_tensor normalizes the numpy array (HxWxC) in the range [0. 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        return to_tensor(placeholder_out)