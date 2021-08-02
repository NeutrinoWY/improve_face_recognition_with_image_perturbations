from sklearn.base import TransformerMixin, BaseEstimator
import numpy
import torch
import imp
import numpy as np
from model.resnet51 import resnet51


class AFFFE(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.config = config
        self.MainModel = imp.load_source('MainModel', "/Users/yunwang/AFFACT+AFFFE/perturbation_1/model/AFFFE_model.py")
        self.network = torch.load("/Users/yunwang/AFFACT+AFFFE/perturbation_1/model/AFFFE.pth")

    def transform(self, X):
        # setup network
        self.network.eval()
        # OPTIONAL: set to cuda environment if available
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.network.to(device)

        # turn it into torch data structure
        X = X.astype(np.float32) / 255.
        tensor = torch.Tensor(X).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            # extract feature vector
            features = self.network(tensor)
            # transform it into 1D numpy array
            features = features.cpu().numpy().flatten()

        return features

    def fit(self, X, y=None):
        return self



class MyExtractor(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.network = resnet51(pretrained=0, progress=False, feature_size=512,
                            num_classes=8631)
        self.checkpoint = torch.load("/Users/yunwang/AFFACT+AFFFE/perturbation_1/model/best_checkpoint.pth")
        self.network.load_state_dict(self.checkpoint['state_dict'])

    def transform(self, X):
        # setup network
        #self.network.eval()
        # OPTIONAL: set to cuda environment if available
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.network.to(device)

        # turn it into torch data structure
        X = X.astype(np.float32) / 255.
        tensor = torch.Tensor(X).unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            # extract feature vector
            features, _ = self.network(tensor)
            # transform it into 1D numpy array
            features = features.cpu().numpy().flatten()

        return features

    def fit(self, X, y=None):
        return self