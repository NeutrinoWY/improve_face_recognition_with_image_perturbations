"""
Environment Utils File
"""
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from zipfile import ZipFile
import time
import torch.nn.functional as F
import bob.measure



def save_original_and_preprocessed_image(index, original_image, preprocessed_image, save_dir):
    """
    saves the original and preprocessed image to disk
    :param filename: the name of the file
    :param original_image: original image
    :param preprocessed_image: preprocessed torch/numpy image ready for model
    :param save_dir: path to the save directory
    """

    matplotlib.use('Agg')

    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(np.transpose(original_image, (1, 2, 0)), 'RGB')

    if isinstance(preprocessed_image, np.ndarray):
        preprocessed_image = Image.fromarray(np.transpose(preprocessed_image, (1, 2, 0)).astype(np.uint8), 'RGB')

    if isinstance(preprocessed_image, torch.Tensor):
        preprocessed_image = np.transpose(preprocessed_image.numpy(), (1, 2, 0))
        preprocessed_image = (preprocessed_image * 1 + 0) * 255
        preprocessed_image = preprocessed_image.astype(np.uint8)
        preprocessed_image = Image.fromarray(preprocessed_image, 'RGB')

    plt.ion()
    plt.clf()
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(original_image)
    plt.subplot(122)
    plt.imshow(preprocessed_image)
    plt.savefig('{}/{}.jpg'.format(save_dir, index))


def tensor_to_image(tensor):
    """
    helper function to convert a tensor to a PIL image
    :param tensor: numpy ndarray/torch tensor
    :return: PIL image
    """
    output_image = None
    if isinstance(tensor, np.ndarray):
        output_image = Image.fromarray(np.transpose(tensor, (1, 2, 0)).astype(np.uint8), 'RGB')

    elif isinstance(tensor, torch.Tensor):
        output_image = np.transpose(tensor.numpy(), (1, 2, 0))
        output_image = (output_image * 1 + 0) * 255
        output_image = output_image.astype(np.uint8)
        output_image = Image.fromarray(output_image, 'RGB')

    return output_image



def createDirectory(dirPath, verbose = True):
    """
    if directory not exist, create one. 
    """
       
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
        if verbose:
            print("Folder not found!!!   " + dirPath + " created.")


def createCheckpointDir(outputFolderPath, debug_mode = False):
    
    ## Create output folder, if it does not exist
    createDirectory(outputFolderPath, verbose = False)
    
    ## Create folder to save current version, if it does not exist
    if debug_mode:
        outputCurrVerFolderPath = os.path.join(outputFolderPath, 'Checkpoint_ver_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_debug' )
    else:
        outputCurrVerFolderPath = os.path.join(outputFolderPath,  'Checkpoint_ver_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') )
        
        
    createDirectory(outputCurrVerFolderPath, verbose = False)
    print("Output will be saved to:  " + outputCurrVerFolderPath)
    
    return outputCurrVerFolderPath




def init_environment(config):
    """
    Initialize the environment for training on GPUs
    :param config: Configuration DotMap
    :return: The cuda device
    """

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()

    # Check if enough devices on current system
    #if len(config.basic.cuda_device_name.split(',')) > torch.cuda.device_count():
        #raise Exception("Not enough devices")

    # Specify device to use
    device = torch.device(config.basic.cuda_device_name.split(',')[0] if use_cuda else "cpu")

    # Empty cache to start on a cleared GPU
    if use_cuda:
        torch.cuda.empty_cache()

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    random.seed(0)

    return device



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self):
        return self.avg



def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def load_helpers():
    curr_epoch = 1
    best_epoch = 1
    EER = []
    train_accuracy = []
    train_loss = []

    return curr_epoch, best_epoch, EER, train_accuracy, train_loss



def calculate_similarity(X1, X2):

    similarity_score = F.cosine_similarity(X1, X2)
    return similarity_score


def calculate_eer(similarities, labels):
    
    n = len(labels)
    positives, negatives = [],[]
    for i in range(n):
        if labels[i] == 1:
            positives.append(similarities[i])
        else:
            negatives.append(similarities[i])

    # get the EER (the optimal threshold where FNR=FPR)
    #threshold = bob.measure.eer_threshold(negatives, positives)
    EER = bob.measure.eer(negatives, positives)

    return EER