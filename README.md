# improve_face_recognition_with_image_perturbations

Python version and libraries:

conda create -n FR python=3.7 \n

conda activate FR \n

conda install -c https://www.idiap.ch/software/bob/conda bob.io.image \n

conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base \n

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch \n

_____________________________________________________________________
1. Using AFFACT data augmentation technique to train deep face recognition model. 

Train defined model with categorical cross-entropy loss:\n
\# inside the conda environment, run the following command \n
python model_training/main.py --model.name resnet51

Train model with arcface loss: \n
\# inside the conda environment, run the following command \n
python model_training/main.py --model.name arcface

Using parallel training: \n
python model_training/main.py --training.parallel 1


2. Experiment image perturbations in the test time. 

Experiment single perturbations in defined perturbation space: \n
python perturbation/expriments.py

3.Search satisfactory combination of perturbations by greedy algorithm and simple genetic algorithm to boost model performance. 

Using greedy algorithm: \n
python perturbation/greedy_combinations.py

Using SimpleGA variant approach: \n
python perturbation/simpleGA.py

