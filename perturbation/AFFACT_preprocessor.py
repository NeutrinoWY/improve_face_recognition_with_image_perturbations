# bounding box and landmarks mode

from sklearn.base import TransformerMixin, BaseEstimator
import bob.io.image
import bob.io.base
import bob.ip.base
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import yaml
import torch
import math
from torchvision.transforms.functional import to_tensor


class PerturbTransformer(TransformerMixin, BaseEstimator):

  def __init__(self, config, scale_param, angle_param, shift_param, perturb=True):
    self.config = config
    self.scale_param = scale_param
    self.angle_param = angle_param
    self.shift_param = shift_param
    self.perturb = perturb
    self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    self.mtcnn = MTCNN(select_largest=True, post_process=False, device=self.device)

  
  def transform(self, X):

    # face detect
    bounding_boxes, confidence, landmarks = self.mtcnn.detect(Image.fromarray(np.transpose(X, (1, 2, 0)), 'RGB'),
                                                     landmarks=True)
    #print(bounding_boxes, confidence, landmarks[0])

    # select the main face: the one with biggest bounding box
    #max_area, index = 0, 0
    #for i, bbx in enumerate(bounding_boxes):
      #area = (bbx[2] - bbx[0]) * (bbx[3]-bbx[1])
      #if area > max_area:
        #max_area = area
        #index = i
    #bbx = bounding_boxes[index]
    if self.config.basic.bounding_box_mode == 0:
      landmarks = landmarks[0]

      t_eye_left = np.array(landmarks[0])
      t_eye_right = np.array(landmarks[1])
      t_mouth_left = np.array(landmarks[3])
      t_mouth_right = np.array(landmarks[4])

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
      #print(bbx)
    else:
      bbx = bounding_boxes[0]
    # left, bottom, right, top
    #print(bbx[0],bbx[1],bbx[2],bbx[3])


    # Define Crop size
    crop_size = [224, 224]

    # Calculate Scale factor,  min(W/w, H/h)
    if self.perturb:
      scale = min(crop_size[0] / (bbx[2] - bbx[0]), crop_size[1] / (bbx[3] - bbx[1]))
    else:
      scale = 1  #for non-perturbation
  
    # the original image is aligned, so the original rotation angle is 0 
    rotation_angle = 0

    # Default shift offset (x,y)
    shift = [0., 0.]

    # perturbation parameters with exact value
    scale *= self.scale_param  
    rotation_angle += self.angle_param
    shift[0] = self.shift_param[0]
    shift[1] = self.shift_param[1]

    # Calculate crop center
    crop_center = [crop_size[0] / 2. + shift[0], crop_size[1] / 2. + shift[1]]

    # Define an input mask
    input_mask = np.ones((X.shape[1], X.shape[2]), dtype=bool)

    # Define output mask
    out_mask = np.ones((crop_size[0], crop_size[1]), dtype=bool)

    # Calculate Center of bounding box
    center = (bbx[1] + (bbx[3] - bbx[1]) / 2., bbx[0] + (bbx[2] - bbx[0]) / 2.)

    # Empty numpy ndarray (serves as placeholder for new image)
    placeholder_out = np.ones((3, crop_size[0], crop_size[1]))
    placeholder_out[placeholder_out > 0] = 0

    # define geometric normalization
    geom = bob.ip.base.GeomNorm(rotation_angle, scale, crop_size, crop_center)

    # Channel-wise application of geonorm and extrapolation of mask
    for i in range(0, 3):
      in_slice = X[i]
      out_slice = np.ones((crop_size[0], crop_size[1]))
      out_slice = out_slice.astype(np.float)
      x = geom.process(in_slice, input_mask, out_slice, out_mask, center)
      try:
        bob.ip.base.extrapolate_mask(out_mask, out_slice)
      except:
        pass
      # Fill channel
      placeholder_out[i] = out_slice

    #print("placeholder_out shape: ", placeholder_out.shape)
    # to create a numpy array of shape H x W x C
    placeholder_out = np.transpose(placeholder_out, (1, 2, 0))

    # convert each pixel to uint8
    placeholder_out = placeholder_out.astype(np.uint8)

    return to_tensor(placeholder_out)


  def fit(self, X, y=None):
    return self
    

