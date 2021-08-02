
import pandas as pd
import argparse
from utils.utils import calculate_results, calculate_similarity
from utils.config_utils import get_config






def bag_of_experiments(config):

    num_perturbations = config.experiment.experiments.n_perturbations

    Thresholds, EERs, FPRs, FNRs, Accs = [0]*num_perturbations, [0]*num_perturbations, [0]*num_perturbations, [0]*num_perturbations, [0]*num_perturbations
    Test_FPRs, Test_FNRs, Test_Accs = [0]*num_perturbations, [0]*num_perturbations, [0]*num_perturbations
    perturbation = [0]*num_perturbations

    for i in range(0, num_perturbations):
        directories = []

        perturb_No = i + 1
        print("perturb_No. : ", perturb_No)
        directory = config.basic.features_path + "/features_" + str(perturb_No)
        directories.append(directory)

        similarities, labels = calculate_similarity(directories, config.basic.train_pairs_path)
        threshold, FPR, FNR, EER, Acc = calculate_results(similarities, labels, process="train") 
        print("threshold: {:.4f}, EER: {:.4f}, Acc: {:.4f}".format(threshold, EER, Acc))

        # save the (train) results
        perturbation[i] = perturb_No
        Thresholds[i] = round(threshold, 5)
        EERs[i] = round(EER, 5)
        FPRs[i] = round(FPR, 5)
        FNRs[i] = round(FNR, 5)
        Accs[i] = round(Acc, 5)

        # use the obtained threshold to test
        similarities, labels = calculate_similarity(directories, config.basic.test_pairs_path)
        FPR, FNR, Acc = calculate_results(similarities, labels, process="test", threshold=threshold)
        print("Test FPR: {:.5f}, Test FNR: {:.5f}, Test Acc: {:.5f}".format(FPR, FNR, Acc))

        # save the test results
        Test_FPRs[i] = round(FPR, 5)
        Test_FNRs[i] = round(FNR, 5)
        Test_Accs[i] = round(Acc, 5)

    
    # save the results of all experiments on the all perturbations into csv file
    d = {'Perturb_No.': perturbation, 'Threshold': Thresholds, 'FPR': FPRs, 'FNR': FNRs, 'EER': EERs, 'Acc': Accs, 'Test_FPR': Test_FPRs, 'Test_FNR': Test_FNRs, 'Test_Acc': Test_Accs}
    df = pd.DataFrame(data=d)
    df.to_csv(config.result.results_save_path,index=False, header=True)


def merge_result_dataframes(config):
    """
    merge the perturbation.csv generated from get_features.py and results.csv generated from experiments.py
    """
    df1 = pd.read_csv(config.result.perturbations_save_path)
    df2 = pd.read_csv(config.result.results_save_path)
    df = pd.merge(df1, df2, on="Perturb_No.")

    # save df in a new csv file
    df.to_csv(config.result.merged_results_path, index=False, header=True)

    



if __name__ == "__main__":
    config = get_config()

    bag_of_experiments(config)

    # merge results and perturbations
    merge_result_dataframes(config)



