# generate perturbation space which includes many perturbations
# apply each perturbation on image data and pass the perturbed images into extractor to get features
# store features under each perturbation in one folder

import os
import bob.io.image
import bob.io.base
import numpy as np
from extractor import AFFFE, MyExtractor
from AFFACT_preprocessor import PerturbTransformer
from torchvision.transforms import transforms, Normalize
import h5py
import pandas as pd
from utils.config_utils import get_config


def get_perturbed_extracted_features(preprocessor, extractor, num, config):
    """
    load image arrays, perturb the image, feed the perturbed image into extractor and get the features.
    Store the features of each image in file.

    :param preprocessor: transformer/preprocessor
    :param extractor: model to extract the features
    :num: int, number to note the perturbations/transforms
    :images_path: directory path that stores the image data
    :features_path: directory path that save the features
    """

    # create a directory to store the features, one perturbation one directory
    parent_dir = config.basic.features_path  #"/local/scratch/wangy/features"
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    # directory name define as [rs,ra,rx,ry]
    directory = "features_" + str(num)

    save_path = os.path.join(parent_dir, directory)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # create a txt file to store the perturbation parameters of the preprocessor
    #txt_file = open(save_path+"/ReadMe.txt","w")
    #Lines = ["scale: " + str(preprocessor.scale_param) +"\n", "angle: " + str(preprocessor.angle_param)+"\n", "shift: " + str(preprocessor.shift_param)]
    #txt_file.writelines(Lines)
    #txt_file.close()

    # walk throught the image folder to read each image data
    for dirpath, sf, files in os.walk(config.basic.images_path):
        #print(files)
        for file in files:
            if ".jpg" in file:
                file_path = os.path.join(dirpath, file)
                #print(file_path)
                image_array = bob.io.image.read(file_path)
                #print(image_array.shape)

                # feed into transformers
                preprocessed_image = preprocessor.transform(image_array)
                preprocessed_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(preprocessed_image)
                features = extractor.transform(preprocessed_image)

                name = file.replace(".jpg", "")
                print(name)

                # save features into h5 file
                h5f = h5py.File(save_path + '/'+ name + '.h5', 'w')
                h5f.create_dataset('features', data=features)
                h5f.close()



def generate_perturbation_spaces(config):

    scales = [1.0, 1.1, 1.2, 1.3, 1.4]
    angles = [-10, 0, 10]
    shifts = [-10, 0, 10]
    perturbation_spaces = []
    for scale in scales:
        for angle in angles:
            for shift_x in shifts:
                for shift_y in shifts:
                    perturbation_spaces.append([scale, angle, shift_x, shift_y])
    
    #print(perturbation_spaces)
    return perturbation_spaces



def main():
    # get needed directory paths from config file
    config = get_config()

    # get all the perturbations
    perturbation_spaces = generate_perturbation_spaces(config)

    n = len(perturbation_spaces)

    # initialize extractor
    if config.extractor_name == "AFFFE":
        extractor = AFFFE()
    elif config.extractor_name == "MyExtractor":
        extractor = MyExtractor(config)
    else:
        raise ValueError


    perturb_No, perturbation = [0] * n , []
    for index, item in enumerate(perturbation_spaces):
        print(index, item)
        # initialize one perturbation preprocessor with specific parameters
        preprocessor = PerturbTransformer(config, scale_param=item[0], angle_param=item[1], shift_param=item[2:], perturb=True)
        # get the features under this perturbation
        get_perturbed_extracted_features(preprocessor=preprocessor, extractor=extractor, num=index+1, config=config)

        # save perturbation parameters into a csv file
        perturb_No[index] = index+1
        perturbation.append(item)

    df = pd.DataFrame(data={"Perturb_No.": perturb_No, "Perturbation":perturbation})
    df.to_csv(config.result.perturbations_save_path, index=False, header=True)



if __name__ == "__main__":
    main()

