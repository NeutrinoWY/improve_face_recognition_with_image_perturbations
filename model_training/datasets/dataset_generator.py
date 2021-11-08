"""
functions that create datasets for training, validation and testing
"""
import os
import pandas as pd
import torch
from torchvision.transforms import transforms, Normalize
from datasets.affact_transformer import AffactTransformer
from datasets.val_transformer import ValTransformer
from datasets.VGG_Face2 import VGG_Faces2_train, VGG_Faces2_val


def get_train_val_dataset(config):
    """
    generate dataloader, dataset and all meta information needed for training
    :param config: the configuration file
    :return: result dict containing dataloader, dataset size, attribute baseline accuracy, dataset meta information
    """

    debug_mode = config.basic.debug_mode

    # Gets the training labels, landmarks, bounding boxes
    labels_train = pd.read_csv(config.dataset.train_list_labels)
    landmarks_train = pd.read_csv(config.dataset.loose_landmark_train)
    bbx_train = pd.read_csv(config.dataset.loose_bb_train)
    #print("path of bbx train:", config.dataset.loose_bb_train)

    # Gets the validation labels, landmarks, bounding boxes
    #print("path of val pairs:", config.dataset.val_pairs)
    pairs = pd.read_csv(config.dataset.val_pairs)
    #print(pairs.head())
    landmarks_val = pd.read_csv(config.dataset.loose_landmark_test)
    bbx_val = pd.read_csv(config.dataset.loose_bb_test)

    labels_train.set_index('NAME_ID', inplace=True)
    landmarks_train.set_index('NAME_ID', inplace=True)
    bbx_train.set_index('NAME_ID', inplace=True)
    landmarks_val.set_index('NAME_ID', inplace=True)
    bbx_val.set_index('NAME_ID', inplace=True)


    # debug mode
    if debug_mode:
        labels_train = labels_train.loc["n000002/0001_01":"n000011/0139_01", :]
        #print("train_labels for debug: ", train_labels)
        #print("select one entry from train_labels debug: ", train_labels.loc["n000002/0001_01"].name)

        landmarks_train = landmarks_train.loc["n000002/0001_01":"n000011/0139_01", :]

        bbx_train = bbx_train.loc["n000002/0001_01":"n000011/0139_01", :]


    # id_label_dict

    train_class_ids = set(labels_train.CLASS_ID)

    class_ids = sorted(list(train_class_ids))
    values = [i for i in range(len(class_ids))]

    id_label_dict = dict(zip(class_ids, values))
    #print("id_label_dict: ", id_label_dict)
    print("number of classes: ", len(id_label_dict))


    # Define the transformations that are applied to each image
    #transforms_train = AffactTransformer(config)
    transforms_train = transforms.Compose([AffactTransformer(config),
                                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transforms_val = transforms.Compose([ValTransformer(config),
                                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Generates the data for training
    # order of the parameters: transform, labels, id_label_dict, landmarks, bounding_boxes, split, config
    dataset_train = VGG_Faces2_train(labels_train, id_label_dict, landmarks_train, bbx_train, config, transforms_train, split="train")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **config.preprocessing.dataloader)

    dataset_val = VGG_Faces2_val(pairs, landmarks_val, bbx_val, config, transforms_val, split="test")
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **config.preprocessing.dataloader)

    dataloaders = {
        'train': dataloader_train,
        'val': dataloader_val
    }

    dataset_sizes = {
        'train': len(dataset_train),
        'val': len(dataset_val)
    }


    result_dict = dict()
    result_dict['dataloaders'] = dataloaders
    result_dict['dataset_sizes'] = dataset_sizes

    return result_dict


