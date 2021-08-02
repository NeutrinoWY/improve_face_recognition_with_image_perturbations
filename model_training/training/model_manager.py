"""
Model Manager Class which helps setting up the model for training
"""
import torch
from torch import nn
from utils.utils import init_environment
from network.resnet51 import resnet51
from network.arcface import ArcFace
from collections import OrderedDict



class ModelManager():
    """
    Model Manager Class
    """

    def __init__(self, config):
        """
        Init Model Manager
        :param config: DotMap Configuration
        """
        self.config = config
        self._check_config_parameters()

        self.device = init_environment(config)

        # if resume training model, load checkpoint
        if self.config.training.resume:
            self.old_checkpoint, self.state_dict = self._get_checkpoint()

        # choose to compute parallel or not
        if self.config.training.parallel:
            # Get model for training on multiple GPUs
            self.model = nn.DataParallel(self._create_model().to(self.device), 
                        device_ids=[int(x[-1]) for x in self.config.basic.cuda_device_name.split(',')])
        else:
            self.model = self._create_model().to(self.device)


    def _check_config_parameters(self):
        if not isinstance(self.config.training.optimizer.learning_rate, float):
            raise ValueError
        elif not isinstance(self.config.training.optimizer.weight_decay, float):
            raise ValueError
        elif not isinstance(self.config.preprocessing.dataloader.batch_size, int):
            raise ValueError
        elif not isinstance(self.config.training.epochs, int):
            raise ValueError
        elif not isinstance(self.config.training.early_stop, int):
            raise ValueError


    def _get_checkpoint(self):
        """
        get the best_checkpoint of the last training, use for resume and continue the training.
        """
        PATH = "{}/{}/{}".format(self.config.basic.result_dir, self.config.basic.checkpoint, "checkpoint.pth" )
        old_checkpoint = torch.load(PATH)

        # if the model was trained by nn.DataParallel, need to remove "module." from the keys
        new_state_dict = OrderedDict()
        for i, key in enumerate(old_checkpoint['state_dict']):
            name = key.replace("module.", "")
            new_state_dict[name] = old_checkpoint['state_dict'][key]

        return old_checkpoint, new_state_dict
 



    def _create_model(self):

        if self.config.model.name == "resnet51":
            model = resnet51(pretrained=self.config.model.pretrained, progress=False, feature_size=self.config.model.feature_size,
                            num_classes=self.config.model.num_classes)

        elif self.config.model.name == "arcface":
            model = ArcFace(pretrained=self.config.model.pretrained, feature_size=self.config.model.feature_size,
                        num_classes=self.config.model.num_classes)
        else:
            raise ValueError
        
        if self.config.training.resume:
            # get the best model from the checkpoint of a previous training
            model.load_state_dict(self.state_dict)

        return model


