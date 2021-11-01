# improve_face_recognition_with_image_perturbations

Python version and libraries:

conda create -n FR python=3.7 

conda activate FR 

conda install -c https://www.idiap.ch/software/bob/conda bob.io.image 

conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base 

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

_____________________________________________________________________
1. Using AFFACT data augmentation technique to train deep face recognition model. 

Train defined model with categorical cross-entropy loss:

\# inside the conda environment, run the following command

python model_training/main.py --model.name resnet51

Train model with arcface loss:

\# inside the conda environment, run the following command

python model_training/main.py --model.name arcface

Using parallel training:   python model_training/main.py --training.parallel 1


2. Experiment image perturbations in the test time. 

Experiment single perturbations in defined perturbation space:

python perturbation/expriments.py

3.Search satisfactory combination of perturbations by greedy algorithm and simple genetic algorithm to boost model performance. 

Using greedy algorithm:   python perturbation/greedy_combinations.py

Using SimpleGA variant approach:   python perturbation/simpleGA.py

