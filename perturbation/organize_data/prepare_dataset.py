
import csv
import pandas as pd
import numpy as np



def organzie_pairs_labels(pairs_file_path, save_path):
    """
    organize the names of positive and negative pairs.
    save the pair images' names and their corresponding labels into a csv file.

    :param pairs_file_path: the original txt file that list the pairs.
    :param save_path: the path that used to save the output csv file
    """

    file = open(pairs_file_path, 'r')
    Lines = file.readlines()
 
    count = 0
    names_1, names_2, labels = [], [], []
    for line in Lines:
        content = line.split('\t')

        # for line that is positive pairs
        if len(content) == 3: 
            identity = content[0]
            name_1 = identity + "_" + str(int(content[1]) + 10000)[1:]
            name_2 = identity + "_" + str(int(content[2]) + 10000)[1:]
            names_1.append(name_1)
            names_2.append(name_2)
            labels.append(1)
        
        # for line that is negative pairs
        elif len(content) == 4:
            identity_1 = content[0]
            identity_2 = content[2]
            name_1 = identity_1 + "_" + str(int(content[1]) + 10000)[1:]
            name_2 = identity_2 + "_" + str(int(content[3]) + 10000)[1:]
            names_1.append(name_1)
            names_2.append(name_2)
            labels.append(0)

        else:
            continue
    
    print("number of pairs:", len(labels))

    # save names_1, names_2, labels into a csv file
    names_1 = np.array(names_1).transpose()
    names_2 = np.array(names_2).transpose()
    labels = np.array(labels).transpose()
    df = pd.DataFrame({"names_1":names_1, "names_2":names_2, "labels":labels})
    df.to_csv(save_path,index=False, header=True)
        

def organize_images(images_path, save_path):
    # This is for simpleGA, as simpleGA was proposed very later and coded separately,
    # which doesn't fit the original structures of files,
    # so I organize the images to simplify the coding of simpleGA. 

    """copy images into one folder"""

    for dirpath, sf, files in os.walk(images_path):
        for file in files:
            if ".jpg" in file:
                file_path = os.path.join(dirpath, file)
                print(file_path)
                copyfile(file_path, os.path.join("/Users/yunwang/Ali_LFW/data", file))




if __name__ == "__main__":

    # organize datasets from txt file to csv file
    #pairs_file_path = "./pairs.txt"
    #save_path = "./pairs.csv"
    #organzie_pairs_labels(pairs_file_path, save_path)

    # organize images
    config = get_config()
    organized_images_path = config.basic.organized_images_path
    organize_images(image_path, organized_images_path)








