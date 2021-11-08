import os
import csv
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import bob.measure
import argparse
from sklearn.metrics import accuracy_score
import collections
from datetime import datetime
import yaml
from dotmap import DotMap


def read_pair_names(file_path):
    """
    get the training dataset from a pre-generated dataframe. 
    """
    df = pd.read_csv(file_path)
    names_1 = df["names_1"]
    names_2 = df["names_2"]
    labels = df["labels"]
    return names_1, names_2, labels

    

def calculate_similarity(directories, file_path):
    """
    @param directories: a list that contains the directory path of the selected perturbations (range from 1 to 81)
    @param features_path: the directory path that stores the features
    @file_path: the path of the pairs names file.
    """

    # get the dataset
    names_1, names_2, labels = read_pair_names(file_path)
    #print("names_1, names_2, labels: ", (names_1, names_2, labels))

    n = len(labels)
    # generate a list to store the similarities of the pairs in the dataset
    similarities = [0] * n


    for i in range(n):
        #print(names_1[i], names_2[i])

        # initiate space to store the image features under different perturbations
        features_1, features_2 = [],[]

        # load features from corresponding hdf5 files 
        for directory in directories:
            # selected perturbed features of the first image in the pair
            features_1.append(h5py.File(os.path.join(directory, names_1[i] + ".h5"), "r")["features"])
            # selected perturbed features of the second image in the pair
            features_2.append(h5py.File(os.path.join(directory, names_2[i] + ".h5"), "r")["features"])
        
        # combine the features from selected perturbations by averaging
        features_1_mean = np.mean(features_1, axis=0)
        features_2_mean = np.mean(features_2, axis=0)


        # calculate similarity between the pair, and store the result in similarities list
        similarities[i] = Similarity(features_1_mean, features_2_mean)
    
    #print("distances, labels: ", distances, labels)

    return similarities, labels



def calculate_similarity_weighted(directories, weights, file_path):
    # for simpleGA only
    """
    @param directories: a list that contains the directory path of the selected perturbations
    @param weights: a list of weights which are correspondant with the perturbations
    @file_path: the path of the pairs names file.
    """

    # get the dataset
    names_1, names_2, labels = read_pair_names(file_path)
    #print("names_1, names_2, labels: ", (names_1, names_2, labels))

    n = len(labels)
    # generate a list to store the similarities of the pairs in the dataset
    similarities = [0] * n

    # get the the directory path that store the perturbed features with the corresponding perturbations
    # directories = [features_path + "/features_" + str(num) for num in perturbations]

    for i in range(n):
        #print(names_1[i], names_2[i])

        # initiate space to store the image features under different perturbations
        features_1, features_2 = [],[]

        # load features from corresponding hdf5 files 
        for directory in directories:
            # selected perturbed features of the first image in the pair
            features_1.append(h5py.File(os.path.join(directory, names_1[i] + ".h5"), "r")["features"])
            # selected perturbed features of the second image in the pair
            features_2.append(h5py.File(os.path.join(directory, names_2[i] + ".h5"), "r")["features"])
        
        # combine the features from selected perturbations by weighted averaging
        features_1_mean = np.sum([weights[j] * features_1[j] for j in range(len(weights))], axis=0) / sum(weights)
        features_2_mean = np.sum([weights[j] * features_2[j] for j in range(len(weights))], axis=0) / sum(weights)


        # calculate similarity between the pair, and store the result in similarities list
        similarities[i] = Similarity(features_1_mean, features_2_mean)
    
    #print("distances, labels: ", distances, labels)

    return similarities, labels




def calculate_results(similarities, labels, process="train", threshold=None):
    n = len(labels)
    print(n)

    #positives = distances[:int(n/2)]
    #negatives = distances[int(n/2):]

    positives, negatives = [],[]
    for i in range(n):
        if labels[i] == 1:
            positives.append(similarities[i])
        else:
            negatives.append(similarities[i])

    if process == "test":
        # FPR: false positive rate , false match rate
        # FNR: false negativve rate, false non match rate
        FPR, FNR = bob.measure.fprfnr(negatives, positives, threshold)
        predicted_labels = predict_labels(similarities, threshold)
        Acc = accuracy_score(labels, predicted_labels)
        return FPR, FNR, Acc
    

    elif process == "train":
        #range_max = max(similarities).item()
        #range_min = min(similarities).item()
        #print("threshold_max, threshold_min", (range_max, range_min))
        
        #FPRs = []
        #FNRs = []
        #thresholds = []
        #for threshold in np.linspace(range_min, range_max, 1000):
            #FPR, FNR = bob.measure.fprfnr(negatives, positives, threshold)
            #thresholds.append(threshold)
            #FPRs.append(FPR)
            #FNRs.append(FNR)
        
        
        # plot the FPR,FNR curve
        #fig, ax = plt.subplots()
        #ax.plot(thresholds, FPRs, 'r--', label='FMR')
        #ax.plot(thresholds, FNRs, 'g--', label='FNMR')
        #plt.xlabel('Threshold')
        #legend = ax.legend(loc='upper right', shadow=False, fontsize='x-large')

        #plt.savefig(os.path.join(result_path, "perturb_"+str(perturbationNo)+".png"))

        # plot ROC (x=FPR, y=1-FNR)
        #plt.figure()
        #ax = FPRs
        #ay = [1 - FNR for FNR in FNRs]
        #plt.plot(ax, ay, 'g--')
        #plt.xlabel('FPR')
        #plt.ylabel('1-FNR')

        #plt.savefig(os.path.join(result_path, "ROC_perturb_"+str(perturbationNo)+".png"))

        # get the EER (the optimal threshold where FNR=FPR)
        threshold = bob.measure.eer_threshold(negatives, positives)
        EER = bob.measure.eer(negatives, positives)
        
        # get the FPR, FNR, given the optimal threshold
        FPR, FNR = bob.measure.fprfnr(negatives, positives, threshold)
        predicted_labels = predict_labels(similarities, threshold)
        Acc = accuracy_score(labels, predicted_labels)
        print("EER, Acc: ", (EER, Acc))

        return threshold, FPR, FNR, EER, Acc


    else:
        raise ValueError("The argument 'precess' should choose between train and test.")


def predict_labels(similarities, threshold):
    n = len(similarities)
    predicted_labels = [0] * n   
    for i in range(n):
        if similarities[i] > threshold:
            predicted_labels[i] = 1
        else:
            predicted_labels[i] = 0
    return predicted_labels


def Similarity(X1, X2):
    X1 = torch.Tensor(X1).unsqueeze(0)
    X2 = torch.Tensor(X2).unsqueeze(0)

    similarity_score = F.cosine_similarity(X1, X2)
    return similarity_score




