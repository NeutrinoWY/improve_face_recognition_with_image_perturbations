import pandas as pd
from utils.config_utils import get_config
from utils.utils import calculate_results, Similarity
import h5py
import os
import numpy as np


def cross_validation(config, perturbs, weights):
    data = pd.read_csv(config.basic.pairs_path)
    #print(data.head())

    test_accs = []
    for i in range(10):
        test = data.iloc[i * 600 : i * 600 + 600, :]
        train = data.drop(test.index)
        
        if i == 1:
            print(test)
            print(train)

        # get the directories that store the perturbed features
        directories = [config.basic.features_path + "/features_" + str(perturb_No) for perturb_No in perturbs]
        print(directories)

        # results on training set
        similarities, labels = calculate_similarity_weighted(directories, weights, train)
        threshold, FPR, FNR, EER, Acc = calculate_results(similarities, labels, process='train')

        # results on test set

        similarities, labels = calculate_similarity_weighted(directories, weights, test)
        test_FPR, test_FNR, test_Acc = calculate_results(similarities, labels, process='test', threshold=threshold)
        test_accs.append(test_Acc)
    
    print("Test Accs for the 10 folds: ", test_accs)
    return sum(test_accs) / 10



def calculate_similarity_weighted(directories, weights, dataset):
    # for simpleGA only
    """
    @param directories: a list that contains the directory path of the selected perturbations
    @param weights: a list of weights which are correspondant with the perturbations
    @dataset: the dataframe of the dataset
    """

    # get the dataset
    names_1, names_2, labels = dataset['names_1'].values.tolist(), dataset['names_2'].values.tolist(), dataset['labels'].values.tolist()
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
        features_1_mean = np.sum([weights[j] * np.array(features_1[j]) for j in range(len(weights))], axis=0) / sum(weights)
        features_2_mean = np.sum([weights[j] * np.array(features_2[j]) for j in range(len(weights))], axis=0) / sum(weights)


        # calculate similarity between the pair, and store the result in similarities list
        similarities[i] = Similarity(features_1_mean, features_2_mean)
    
    #print("distances, labels: ", distances, labels)

    return similarities, labels


if __name__ == "__main__":
    config = get_config()

    # for model resnet51
    #perturbs = [23, 19, 10]
    #weights = [9, 6, 5]

    # for model resnet51_align
    perturbs = [14]
    weights = [1]
    print(cross_validation(config, perturbs, weights))





    