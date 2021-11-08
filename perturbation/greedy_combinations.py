
from utils.utils import calculate_results, calculate_similarity
from utils.config_utils import get_config
import pandas as pd
import numpy as np
import argparse




def greedy_combination(config):
    """
    :param features_path: the path the stores all folders of the perturbed features
    :file_path: the path of the dev pair list
    :result_save_path: the path to save the result file
    :num_perturbations: the total number of perturbations used for the greedy search of combinations
    """
    # the global MAX_ACC, the max_acc of a loop
    MAX_ACC, max_acc = -1, 0
    # the global threshold
    BEST_THRESHOLD = -1
    # list to store the best combination of a loop
    selected_perturbations = []

    # LISTs to save the best combination of each loop
    MAX_ACC_RECORD, BEST_THRESHOLD_RECORD, SELECTED_PERTURBATION_RECORD= [], [], []
    TEST_FPR_RECORD, TEST_FNR_RECORD, TEST_ACC_RECORD = [], [], []
    

    while max_acc >= MAX_ACC:
        print("selected perturbations: ", selected_perturbations)
        # initialize lists to store the thresholds and accuracies of the 162 perturbations
        num_perturbations = config.experiment.greedy_combination.n_perturbations
        Accs = [0] * num_perturbations
        Thresholds = [0] * num_perturbations
        for i in range(num_perturbations):
            perturbationNo = i + 1
            print("perturbation ", i + 1)

            # initiative directories to store the directories path of the perturbed features
            directories = [] 
            if perturbationNo in selected_perturbations:
                continue
            else:
                # get the the directory path that store the perturbed features with the corresponding perturbation
                directories.append(config.basic.features_path + "/features_" + str(perturbationNo))
            
            # add features that already selected 
            if len(selected_perturbations) > 0: 
                selected_feature_paths = [config.basic.features_path + "/features_" + str(num) for num in selected_perturbations]
                for path in selected_feature_paths:
                    directories.append(path)

            similarities, labels = calculate_similarity(directories, config.basic.train_pairs_path)
            threshold, FPR, FNR, EER, Acc = calculate_results(similarities, labels, process="train")

            # store accuracy and threshold of the perturbation in lists
            Accs[i] = Acc
            Thresholds[i] = threshold
        
        #print("Accs, Thresholds: ", Accs, Thresholds)

        # get the max accuracy and best threshold among all the perturbations
        max_acc = max(Accs)
        index = np.argmax(np.array(Accs))
        best_threshold = Thresholds[index]

        # if the max_acc of this iteration is larger or equal to the acc of the selected_perturbations(MAX_ACC),
        # update the MAX_ACC, add the corresponding perturbation number into selected_perturbations
        if max_acc >= MAX_ACC:
            MAX_ACC = max_acc
            BEST_THRESHOLD = best_threshold
            selected_perturbations.append(index + 1)

            # and store the best accuracy, thereshold, selected perturbation No. of this loop into the record lists
            MAX_ACC_RECORD.append(MAX_ACC)
            BEST_THRESHOLD_RECORD.append(BEST_THRESHOLD)
            SELECTED_PERTURBATION_RECORD.append(str(selected_perturbations))

            # validate the selected combination of perturbations of this loop
            directories = [config.basic.features_path + "/features_" + str(perturb_No) for perturb_No in selected_perturbations]
            similarities, labels = calculate_similarity(directories, config.basic.test_pairs_path)
            Test_FPR, Test_FNR, Test_Acc = calculate_results(similarities, labels, process="test", threshold=BEST_THRESHOLD)

            # add the validation results to record
            TEST_FPR_RECORD.append(Test_FPR)
            TEST_FNR_RECORD.append(Test_FNR)
            TEST_ACC_RECORD.append(Test_Acc)


    # save the result record into csv file
    df = pd.DataFrame(data={"selected_perturbations": SELECTED_PERTURBATION_RECORD, "Threshold": BEST_THRESHOLD_RECORD, 
                            "Acc": MAX_ACC_RECORD, "Test_FPR": TEST_FPR_RECORD, "Test_FNR": TEST_FNR_RECORD,
                            "Test_ACC": TEST_ACC_RECORD})
    df.to_csv(config.result.greedy_save_path, index=False, header=True)
    
    #print("selected_perturbations, MAX_ACC, BEST_THRESHOLD: ", selected_perturbations, MAX_ACC, BEST_THRESHOLD)
    return selected_perturbations, MAX_ACC, BEST_THRESHOLD



if __name__ == "__main__":
    config = get_config()


    selected_perturbations, MAX_ACC, BEST_THRESHOLD = greedy_combination(config)
    print("___________Search Results (find the whole result record in greedy_combination_results.csv)_______")
    print("selected_perturbations: ", selected_perturbations)
    print("The best threshold and accuracy of the selected perturbations: Threshold={:.5f}, Acc={:.5f}".format(BEST_THRESHOLD, MAX_ACC))

