# improve_face_recognition_with_image_perturbations

Python version and libraries:

conda create -n FR python=3.7
conda activate FR

conda install -c https://www.idiap.ch/software/bob/conda bob.io.image
conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

_____________________________________________________________________
Using AFFACT data augmentation technique to train deep face recognition model. 

Experiment image perturbations in the test time. 

Search satisfactory combination of perturbations by greedy algorithm and simple genetic algorithm to boost model performance. 

