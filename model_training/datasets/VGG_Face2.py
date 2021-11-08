#!/usr/bin/env python
# define VGGFace2 dataset

import numpy as np
#import PIL.Image
import bob.io.image
import bob.io.base
import scipy.io
import torch
from torch.utils import data
import torchvision.transforms
from facenet_pytorch.models.mtcnn import MTCNN
from utils.utils import save_original_and_preprocessed_image
import pandas as pd


class VGG_Faces2_train(data.Dataset):

    # mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, labels, id_label_dict, landmarks, bounding_boxes, config, transform=None, split='train'):
        """
        :param labels: dataframe of image file list, eg 'train_dev_labels', 'test_dev_labels’
        :param id_label_dict: X[class_id] -> label
        :param split: train or val
        :param transformer: transformer
        :param config: config file
        :param upper: max number of image used for debug
        """

        self.split = split
        self.transform = transform
        self.id_label_dict = id_label_dict
        self.labels = labels

        # use landmarks
        if config.dataset.bounding_box_mode == 0:
            self.landmarks = landmarks

        # use bounding boxes
        elif config.dataset.bounding_box_mode == 1:
            self.bounding_boxes = bounding_boxes

        # use face detector
        elif config.dataset.bounding_box_mode == 2:
            self.mtcnn = MTCNN(select_largest=True, device=config.basic.cuda_device_name.split(',')[0])

        else:
            raise Exception("Chose a valid bounding_box_mode (0=landmarks hand-labeled, 1=bbx hand-labeled, 2=bbx detected")

        self.config = config

    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, index):
        # Get image name. In the csv file lables.csv, the column 'NAME_ID' is set to index
        name_id = self.labels.iloc[index].name
        #print("name_id: ", name_id)

        # Get class_id
        class_id = self.labels.iloc[index].CLASS_ID
        #print("class_id: ", class_id)

        # get one hot encoded label
        y = self.id_label_dict[class_id]
        y = torch.tensor(y)
        #print("label: ", y)

        img_path = '{}/{}/{}.jpg'.format(self.config.dataset.image_dir, self.split, name_id)
        #print("image_path: ", img_path)

        image = bob.io.base.load(img_path)
        #print("original image array: ", image)


        # Prepare bounding boxes/landmarks for transformer
        landmarks, bounding_boxes = None, None
        if self.config.dataset.bounding_box_mode == 0:
            landmarks = self.landmarks.iloc[index].tolist()
            landmarks = landmarks[:4] + landmarks[6:]
            #print(landmarks)

        elif self.config.dataset.bounding_box_mode == 1:
            bounding_boxes = self.bounding_boxes.iloc[index].tolist()
            bounding_boxes = bounding_boxes[1:]
            #print(bounding_boxes)

            if self.config.dataset.bounding_box_scale:
                scale = self.config.dataset.bounding_box_scale
                bounding_boxes[0] = bounding_boxes[0] - ((scale - 1) / 2 * bounding_boxes[2])
                bounding_boxes[1] = bounding_boxes[1] - ((scale - 1) / 2 * bounding_boxes[3])
                bounding_boxes[2] = scale * (bounding_boxes[2])
                bounding_boxes[3] = scale * (bounding_boxes[3])

        # Create input structure
        input = {
            'image': image,
            'landmarks': landmarks,
            'bounding_boxes': bounding_boxes,
            'index': index
        }

        #print(input)

        # Apply transform, output is tensor
        X = self.transform(input)

        # Save every X picture to validate preprocessing
        if self.config.preprocessing.save_preprocessed_image.enabled:
            if index % self.config.preprocessing.save_preprocessed_image.frequency == 0:
                save_original_and_preprocessed_image(index, image, X, self.config.basic.save_dir)

        return X, y, index




class VGG_Faces2_val(data.Dataset):
    def __init__(self, pairs, landmarks, bounding_boxes, config, transform=None, split='test'):
        """
        :param labels: dataframe of image file list, eg 'train_dev_labels', 'test_dev_labels’
        :param id_label_dict: X[class_id] -> label
        :param split: train or test, we use the test set for validation
        :param transformer: transformer
        :param config: config file
        :param upper: max number of image used for debug
        """

        self.split = split
        self.transform = transform
        self.pairs = pairs

        # use landmarks
        if config.dataset.bounding_box_mode == 0:
            self.landmarks = landmarks

        # use bounding boxes
        elif config.dataset.bounding_box_mode == 1:
            self.bounding_boxes = bounding_boxes

        # use face detector
        elif config.dataset.bounding_box_mode == 2:
            self.mtcnn = MTCNN(select_largest=True, device=config.basic.cuda_device_name.split(',')[0])

        else:
            raise Exception("Chose a valid bounding_box_mode (0=landmarks hand-labeled, 1=bbx hand-labeled, 2=bbx detected")

        self.config = config


    def __len__(self):
        return self.pairs.shape[0]

    
    def __getitem__(self, index):

        # Get image name, in the file val_pairs.csv, the index is numbers
        name_1 = self.pairs.iloc[index].NAME_1
        name_2 = self.pairs.iloc[index].NAME_2
        #print("name_1, name_2: ", (name_1, name_2))

        # Get LABEL
        label = self.pairs.iloc[index].LABEL
        y = torch.tensor(label)
        #print("label:", y)

        img_path_1 = '{}/{}/{}.jpg'.format(self.config.dataset.image_dir, self.split, name_1)
        img_path_2 = '{}/{}/{}.jpg'.format(self.config.dataset.image_dir, self.split, name_2)

        image_1 = bob.io.base.load(img_path_1)
        image_2 = bob.io.base.load(img_path_2)


        # Prepare bounding boxes/landmarks for transformer
        landmarks_1, landmarks_2, bounding_boxes_1, bounding_boxes_2 = None, None, None, None
        if self.config.dataset.bounding_box_mode == 0:
            landmarks_1 = self.landmarks.loc[name_1, :].tolist()  #P1X - P5Y
            landmarks_1 = landmarks_1[:4] + landmarks_1[6:]
            #print('landmarks_1:', landmarks_1)

            landmarks_2 = self.landmarks.loc[name_2, :].tolist()
            landmarks_2 = landmarks_2[:4] + landmarks_2[6:]
            #print("landmarks_2:", landmarks_2)

        elif self.config.dataset.bounding_box_mode == 1:
            bounding_boxes_1 = self.bounding_boxes.loc[name_1, :].tolist()
            bounding_boxes_1 = bounding_boxes_1[1:]

            bounding_boxes_2 = self.bounding_boxes.loc[name_2, :].tolist()
            bounding_boxes_2 = bounding_boxes_2[1:]

            if self.config.dataset.bounding_box_scale:
                scale = self.config.dataset.bounding_box_scale
                bounding_boxes_1[0] = bounding_boxes_1[0] - ((scale - 1) / 2 * bounding_boxes_1[2])
                bounding_boxes_1[1] = bounding_boxes_1[1] - ((scale - 1) / 2 * bounding_boxes_1[3])
                bounding_boxes_1[2] = scale * (bounding_boxes_1[2])
                bounding_boxes_1[3] = scale * (bounding_boxes_1[3])

                bounding_boxes_2[0] = bounding_boxes_2[0] - ((scale - 1) / 2 * bounding_boxes_2[2])
                bounding_boxes_2[1] = bounding_boxes_2[1] - ((scale - 1) / 2 * bounding_boxes_2[3])
                bounding_boxes_2[2] = scale * (bounding_boxes_2[2])
                bounding_boxes_2[3] = scale * (bounding_boxes_2[3])

        # Create input structure
        input_1 = {
            'image': image_1,
            'landmarks': landmarks_1,
            'bounding_boxes': bounding_boxes_1,
            'index': index
        }

        input_2 = {
            'image': image_2,
            'landmarks': landmarks_2,
            'bounding_boxes': bounding_boxes_2,
            'index': index
        }


        # Apply transform, output is tensor
        X1 = self.transform(input_1)
        X2 = self.transform(input_2)

        return X1, X2, y, index

    
