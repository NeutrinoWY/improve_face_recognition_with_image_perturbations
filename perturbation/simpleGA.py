from utils.utils import calculate_results, Similarity, read_pair_names, calculate_similarity_weighted
from utils.config_utils import get_config
import pandas as pd
import numpy as np
import argparse
from AFFACT_preprocessor import PerturbTransformer
import random



random.seed(42)


def get_full_population(config):
    """
    results of individual perturbations are already run and stored in csv file.
    read the csv file to get the existing results.
    Add column "Weight" to denote the weight of the perturbation 
    """
    population = pd.read_csv(config.result.merged_results_path)
    # select the columns that will need in simpleGA
    population = population[["Perturb_No.", "Perturbation", "Thresholds", "Accs", "Test Accs"]]
    # population = population[["Perturb_No.", "Perturbation", "Threshold", "EER", "Acc", "Test_Acc"]]
    # add column weight, initialize weight as the number of the max_iteration, so the weight won't be negative after mutation
    weight = [3] * population.shape[0]
    print(population.shape[0])
    population['Weight'] = weight
    return population



def selection(population, config):
    """
    select part of the individuals as parents from the population.
    population here are perturbation spaces.
    :param population: dataframe with columns: weight, [rs, ra, rx, ry], threshold, train accuracy, test accuracy
    :param confg: config
    choromosome here is the weight and perturbation parameters, fitness score is test accuracy.
    """
    # select elites/parents based on the scores of the test set
    if not config.simpleGA.random_selection:
        elite_selection = config.simpleGA.elite_selection
    parents = population.nlargest(elite_selection, 'Accs')
    #population.nlargest(elite_selection, 'Acc')
    return parents




def fitness_score_simple(weights, perturbs, config):
    """
    combine all the selected population to get the fitness score.
    :param weights: list [w, ...] store the weights
    :param perturbs: list [1, 2, ...]  store the perturbation Numbers. 
    :param config: config
    """

    # directories that store the perturbed features
    directories = [config.basic.features_path + "/features_" + str(num) for num in perturbs]

    # calculated metrics of the weighted combination on training set
    similarities, labels = calculate_similarity_weighted(directories, weights, config.basic.train_pairs_path)
    threshold, train_FPR, train_FNR, train_EER, train_Acc = calculate_results(similarities, labels, process="train") 
    
    # test the weighted combination on test dataset
    similarities, labels = calculate_similarity_weighted(directories, weights, config.basic.test_pairs_path)
    test_FPR, test_FNR, test_Acc = calculate_results(similarities, labels, process="test", threshold=threshold)
    
    return threshold, train_Acc, test_Acc, perturbs, weights



# TODO
def fitness_score_greedy(weights, perturbs, config):
    """
    use greedy search to search the best weighted perturbation combination in a generation
    :param weights: list [w, ...] store the weights
    :param perturbs: list [1, 2, ...]  store the perturbation Numbers. 
    :param config: config 
    """

    # the global max acc of the greedy search
    MAX_ACC = 0
    # the max acc of each iteration
    max_acc = 0

    # best threshold of each iteration
    BEST_THRESHOLD = -10000
    # list to save the found perturbation combination of each iteration, initialize
    selected_perturbations = []
    # list to save the weights of perturbation combination of each iteration, initialize
    weights_list = []

    # LISTs to save the best combination of each iteration
    MAX_ACC_RECORD, SELECTED_PERTURBATION_RECORD= [], []
    BEST_THRESHOLD_RECORD, TEST_ACC_RECORD, WEIGHT_RECORD = [], [], []
    

    while max_acc >= MAX_ACC:
        print("selected perturbations: ", selected_perturbations)
        # initialize lists to store the thresholds and accuracies of the perturbations
        Accs = [0] * len(perturbs)
        Thresholds = [0] * len(perturbs)
        for i, perturb in enumerate(perturbs):

            # initialize weights to store the weights of each selected perturbation
            w = []
            # initiative directories to store the directories path of the perturbed features
            directories = []
            if perturb in selected_perturbations:
                continue
            else:
                # get the the directory path that store the perturbed features with the corresponding perturbation
                directories.append(config.basic.features_path + "/features_" + str(perturb))
                w.append(weights[i])
            
            # add features that already selected 
            if len(selected_perturbations) > 0: 
                selected_feature_paths = [config.basic.features_path + "/features_" + str(num) for num in selected_perturbations]
                for path in selected_feature_paths:
                    directories.append(path)
                for num in selected_perturbations:
                    w.append(weights[perturbs.index(num)])

            similarities, labels = calculate_similarity_weighted(directories, w, config.basic.train_pairs_path)
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
        # and store the accuracy, thereshold, selected perturbation No. into the record lists
        if max_acc >= MAX_ACC:
            MAX_ACC = max_acc
            BEST_THRESHOLD = best_threshold
            selected_perturbations.append(perturbs[index])
            weights_list.append(weights[index])

            MAX_ACC_RECORD.append(MAX_ACC)
            BEST_THRESHOLD_RECORD.append(BEST_THRESHOLD)
            SELECTED_PERTURBATION_RECORD.append(str(selected_perturbations))
            WEIGHT_RECORD.append(str(weights_list))

            # test the best combination of this loop
            directories = [config.basic.features_path + "/features_" + str(perturb_No) for perturb_No in selected_perturbations]
            similarities, labels = calculate_similarity_weighted(directories, weights_list, config.basic.test_pairs_path)
            TEST_FPR, TEST_FNR, TEST_ACC = calculate_results(similarities, labels, process="test", threshold=BEST_THRESHOLD)

            TEST_ACC_RECORD.append(TEST_ACC)
    
    # based on the best test accuracy to get the found perturbation numbers and their weights as the final result of the greedy search
    # index in which iteration we get the best test accuracy
    idx = np.argmax(np.array(TEST_ACC_RECORD))
    # get the best test Acc
    best_test_acc = TEST_ACC_RECORD[idx]
    # get the selected perturbation numbers
    best_perturbations = SELECTED_PERTURBATION_RECORD[idx]
    # get the weights of the selected perturbations
    corresponding_weights = WEIGHT_RECORD[idx]
    # get the train Acc of this perturbation combination
    train_acc = MAX_ACC_RECORD[idx]
    # threshold
    threshold = BEST_THRESHOLD_RECORD[idx]

    
    return threshold, train_acc, best_test_acc, best_perturbations, corresponding_weights




def fitness_score(weights, perturbs, config):
    """
    choose the method to calculate fitness_score of a generation.
    """
    # greedy search the weighted combination: features = sum(wi * feature_i) / sum(wi)
    if config.simpleGA.fitness_score == "greedy_search":
        threshold, acc, test_acc, selected_perturbs, correspond_weights = fitness_score_greedy(weights, perturbs, config)

    elif config.simpleGA.fitness_score == "simple":
        # in this method, all input perturbs are selected
        threshold, acc, test_acc, selected_perturbs, correspond_weights = fitness_score_simple(weights, perturbs, config)

    else:
        raise ValueError

    return threshold, acc, test_acc, selected_perturbs, correspond_weights







    
def crossover(parent1, parent2):
    """
    crossover two parents to create two children by exchanging parameters [w, rs, ra, rx, ry]
    """
	# children are copies of parents by default
    child1, child2 = parent1.copy(), parent2.copy()
    # define crossover point
    pt = random.randint(1, len(parent1) - 2)
	# perform crossover
    child1 = parent1[:pt] + parent2[pt:]
    child2 = parent2[:pt] + parent1[pt:]

    return child1, child2



# mutation operator
def mutation(child):
    """
    apply mutation on child [w, rs, ra, rx, ry]
    """
    # initialize mutation mask
    mask = [0] * 5
    for i in range(5):
        value = random.choice([0, 1])
        mask[i] = value

    # define the mutation values for each gene/element
    mutate_w = random.choice([-1, 1])
    mutate_rs = random.choice([-0.1, 0, 0.1])
    mutate_ra = random.choice([-10, 0, 10])
    mutate_rx = random.choice([-10, 0, 10])
    mutate_ry = random.choice([-10, 0, 10])
    mutation = [mutate_w, mutate_rs, mutate_ra, mutate_rx, mutate_ry]

    # apply mutation on weight  
    child[0] += mutate_w

    # ensure the mutated perturbation is still in the original peturbation space
    # if rs is already the smallest value in perturbation space, then not add negative value
    # if rs is already the largest value in perturbation space, then not add positive value
    if child[1] + mutate_rs >= 0.6 and child[1] + mutate_rs <= 1.1:
        child[1] = round(child[1] + mutate_rs, 1)
    
    # mutation on angle param ra
    if child[2] * mutate_ra <= 0:
        child[2] += mutate_ra

    # mutation on shift params rx, ry
    if child[3] * mutate_rx <= 0:
        child[3] += mutate_rx
    
    if child[4] * mutate_ry <= 0:
        child[4] += mutate_ry
    
    return child




# genetic algorithm
# population: dataframe with columns:Perturb_No., Weight, Perturbation, Threshold, Acc, Test_Acc
# selected_parents: part of the dataframe of population
# parents / children: list of lists [[w, rs, ra, rx, ry], [w, rs, ra, rx, ry], ....]
# weights: list [w, ...] store the weights
# perturbs: list [1, 2, ...]  store the perturbation Numbers. 
def genetic_algorithm(config):

    # get the full population from the record
    population = get_full_population(config)

    # select parents from pupulation, based on the train_acc of the single perturbations
    select_parents = selection(population, config)
    # deal with the string, change to list
    select_parents["Perturbation"] = select_parents["Perturbation"].apply(eval)

    # get the chromosomes/parameters of parents, which will be crossover and mutated later
    # to store the chromosome of all individual who is selected as a parent
    parents = []
    # to store the chromosome of an individual [w, rs, ra, rx, ry]
    for i in range(select_parents.shape[0]):
        individual = [0] * 5
        individual[0] = select_parents.Weight.iloc[i]
        individual[1] = select_parents.Perturbation.iloc[i][0]
        individual[2] = select_parents.Perturbation.iloc[i][1]
        individual[3] = select_parents.Perturbation.iloc[i][2]
        individual[4] = select_parents.Perturbation.iloc[i][3]

        parents.append(individual)
    print(parents)

    # initialize list variables to track the values of each generation
    generations = []
    selected_individuals = []
    individual_weights = []
    accuracies = []
    test_accuracies = []

    best_test_accuracy = -1   # global accuracy, the best test acc among all generations
    test_accuracy = 0    # accuracy of a generation
    generation = 0      # initialize the generation
    while test_accuracy >= best_test_accuracy:
        generation += 1

        # initialize weights and perturbs to store the weights and perturbations of the generation
        weights = []
        perturbs = []
        # get the perturb No.s of the generation from the full population
        for i, choromosome in enumerate(parents):
            weights.append(choromosome[0])
            perturbation = choromosome[1:]
            # get the perturb NO. crrespond to the perturbation
            perturb_No = population.loc[population.Perturbation == str(perturbation), ["Perturb_No."]].values.item()
            # find the perturb No. of the perturbation
            perturbs.append(perturb_No)
        print(perturbs)
        
        threshold, accuracy, test_accuracy, selected_perturbs, correspond_weights = fitness_score(weights, perturbs, config)

        # store the generation number, selected perturbations and accuracies of this generation
        generations.append(generation)
        selected_individuals.append(selected_perturbs)
        accuracies.append(accuracy)
        test_accuracies.append(test_accuracy)
        individual_weights.append(correspond_weights)
        

        # update the global best_accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        
        # if the number of parents is less than 1, stop
        if len(parents) <= 1:
            break

        # initialize children list to store the next generation
        children = []
        # select parents to produce the next generation (two parents -> two children)
        for i in range(0, len(parents)-1, 2):
            # select pair of parents
            parent1, parent2 = parents[i], parents[i+1]
            for child in crossover(parent1, parent2):
                # mutate
                child = mutation(child)
                # if weight!=0, add to the next generation, otherwise exclude it
                if child[0] != 0:
                    children.append(child)

        # if there is one parent that has no partner to produce child, 
        # put it back to the next generation with the produced children  
        if len(parents) % 2 != 0:
            children.append(parents[-1])

        # remove duplicated child
        #children = list(set(children))

        # replace parents  // or add to parents?
        parents = children
        print(parents)

        # early stop when max iteration is met
        if generation == config.simpleGA.max_iterations:
            break

    # save the results of each generation into csv file
    df = pd.DataFrame(data={"Generation": generations, "Individuals": selected_individuals, "Weights": individual_weights, "Acc": accuracies, "Test_Acc": test_accuracies})
    df.to_csv(config.result.simpleGA_save_path, index=False, header=True)




if __name__ == "__main__":
    config = get_config()
    #population = get_full_population(config)
    #perturbation = [0.9, 10, 10, 10]
    #print(population.loc[population.Perturbation == str(perturbation), ["Perturb_No."]])
    genetic_algorithm(config)


