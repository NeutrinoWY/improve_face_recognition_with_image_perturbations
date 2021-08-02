import torch
from torch import nn
from utils.utils import init_environment
from network.resnet51 import resnet51
from network.arcface import ArcFace
from collections import OrderedDict    

# self.old_chekpoint, self.state_dict = get_checkpoint(self)

def get_checkpoint():
    """
    get the best_checkpoint of the last training, use for resume and continue the training.
    """
    PATH = "{}/{}/{}".format("/local/scratch/wangy/model_results", "Checkpoint_ver_2021-07-09_21-06-58", "best_checkpoint.pth")

    old_checkpoint = torch.load(PATH)

    # if the model was trained by nn.DataParallel, need to remove "module." from the keys
    new_state_dict = OrderedDict()
    for i, key in enumerate(old_checkpoint['state_dict']):
        name = key.replace("module.", "")
        value = old_checkpoint['state_dict'][key]
        new_state_dict[name] = value
        if i == 0:
            print(new_state_dict)
            print(name)

    return old_checkpoint, new_state_dict

    
def create_model(state_dict):

    model = resnet51(pretrained=0, progress=False, feature_size=512,
                            num_classes=8631)

    model.load_state_dict(state_dict)

    return model
    
if __name__ == "__main__":
    checkpoint, state_dict = get_checkpoint()
    model = create_model(state_dict)

    

