# improve_face_recognition_with_image_perturbations

Python version and libraries:

conda create -n FR python=3.8 

conda activate FR 

conda install -c https://www.idiap.ch/software/bob/conda bob.io.image 

conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base 

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

This code was run on Linux machine with CUDA GPU

_____________________________________________________________________
Using AFFACT data augmentation technique to train deep face recognition model
_____________________________________________________________________

* Train defined model with categorical cross-entropy loss:

\# inside the conda environment, run the following command

python model_training/main.py --model.name resnet51

* Train model with arcface loss:

\# inside the conda environment, run the following command

python model_training/main.py --model.name arcface

* Using parallel training:   python model_training/main.py --training.parallel 1

_____________________________________________________________________
Experiment image perturbations in the test time
______________________________________________________________________

* Experiment single perturbations in defined perturbation space:

python perturbation/expriments.py


* Search satisfactory combination of perturbations by greedy algorithm and simple genetic algorithm to boost model performance. 

Using greedy algorithm:   python perturbation/greedy_combinations.py

Using SimpleGA variant approach:   python perturbation/simpleGA.py

