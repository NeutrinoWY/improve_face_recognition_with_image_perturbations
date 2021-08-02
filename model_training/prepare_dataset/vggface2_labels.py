# remove '.jpg' from the name_id, let the name_id the same with that in landmarks and bounding_boxes
# add class_id to train/test list file, save them into csv file
# generate and save partition file for futher train val dataset split

import os
import pandas as pd
from shutil import copyfile
import numpy as np
import random


# merge list, landmarks, bounding_box
def get_labels(list_file, save_path, upper=None):
    """
    :param list_file: the path of the original image list files, eg train_list.txt
    :param save_path: the path to save the generated file
    :param upper: None or integer, for debug
    """
    name_ids = []
    class_ids = []

    with open(list_file, 'r') as f:
        for i, name_id in enumerate(f):
            name_id = name_id.replace(".jpg", "").strip()
            class_id = name_id.split("/")[0]
            name_ids.append(name_id)
            class_ids.append(class_id)

            # for debug
            if upper and i == upper:
                break


    # save into dataframe
    df = pd.DataFrame(data = {"NAME_ID": name_ids, "CLASS_ID": class_ids})
    print("length:", df.shape[0])

    # whether the file is train or test
    file_mode = list_file.split("/")[-1]
    file_mode = file_mode.split(".")[0]

    df.to_csv(os.path.join(save_path, file_mode + "_labels.csv"), index=False, header=True)

    return os.path.join(save_path, file_mode + "_labels.csv")



def partition(train_file_path, save_path, upper=None):
    """
    :param train_list_labels: the path of train_list_labels.csv
    :param save_path: the path to save the generated partition file
    :param upper: None or integer, for debug mode
    """
    train_list_labels = pd.read_csv(train_file_path)
    length = train_list_labels.shape[0]

    partition = [0] * length
    for i in range(length):
        partition[i] = random.choices([0, 1], [0.2, 0.8])[0]
        if upper and i == upper:
            break
    print("training data percentage: {}".format(sum(partition) / length))

    name_ids = train_list_labels.NAME_ID

    df = pd.DataFrame(data = {"NAME_ID": name_ids, "PARTITION": partition})
    df.to_csv(os.path.join(save_path, "partition.csv"), index=False, header=True)

    return os.path.join(save_path, "partition.csv")


def get_val_pairs(test_labels, save_path, upper=5000, debug_mode=1):
    """
    generate positive and negative pairs from the test_labels.csv for model validation.
    """
    test_labels = pd.read_csv(test_labels)
    #print(test_labels.head())

    # select a subset for debug
    if debug_mode:
        test_labels = test_labels.loc[0:1000, :]

    test_labels.set_index("NAME_ID", inplace=True)
    name_ids = test_labels.index.tolist()
    #print("name_ids:", name_ids)

    class_ids = test_labels.CLASS_ID.tolist()
    n = len(set(class_ids))
    print("num_classes:", n)

    NAME_1, NAME_2, LABEL = [0] * n * 4, [0] * n * 4, [0] * n * 4
    # generate 2n positive pairs
    i = 0 
    for class_id in list(set(class_ids)):
        print(class_id)
        samples = test_labels[test_labels.CLASS_ID.eq(class_id)]
        #print(samples)
        pairs = random.sample(samples.index.tolist(), 4)
        NAME_1[i] = pairs[0]
        NAME_2[i] = pairs[1]
        NAME_1[i + 1] = pairs[2]
        NAME_2[i + 1] = pairs[3]
        LABEL[i] = 1
        LABEL[i+1] = 1

        i += 2
    

    # generate n negative pairs
    count = -1
    for i in range(upper):
        pair = random.sample(name_ids, 2)
        print(pair)
        class_1 = pair[0].split("/")[0]
        class_2 = pair[1].split("/")[0]
        if class_1 != class_2:
            count += 1
            LABEL[2 * n + count] = 0
            NAME_1[2 * n + count] = pair[0]
            NAME_2[2 * n + count] = pair[1]
        if count == 2 * n - 1:
            break
    
    df = pd.DataFrame(data = {"NAME_1": NAME_1, "NAME_2": NAME_2, "LABEL": LABEL})
    df.to_csv(os.path.join(save_path, "val_pairs.csv"), index=False, header=True)

    

def main(upper):
    """
    :param upper: None or interger, for debug
    """
    train_list_file = "/Users/yunwang/VGGFace2/train_list.txt"
    test_list_file = "/Users/yunwang/VGGFace2/test_list.txt"
    save_path = "/Users/yunwang/VGGFace2/datasets"

    train_file_path = get_labels(train_list_file, save_path, upper=upper)
    test_file_path = get_labels(test_list_file, save_path, upper=upper)

    partition("/Users/yunwang/VGGFace2/datasets/train_list_labels.csv", save_path, upper=upper)


if __name__ == "__main__":

    #main(upper=None)

    test_labels = "/Users/yunwang/VGGFace2/datasets/test_list_labels.csv"
    save_path = "/Users/yunwang/VGGFace2/datasets"
    get_val_pairs(test_labels, save_path, upper=20000, debug_mode=0)