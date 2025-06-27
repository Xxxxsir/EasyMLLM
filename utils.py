import os
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from models import model_dict 


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name:str, num_classes:int, pre_trained:bool = False, pretrained_path:str = None, **kwargs):

    if model_name not in model_dict:
        raise ValueError(f"model is not supported from {list(model_dict.keys())}")

    model = model_dict[model_name](num_classes=num_classes, pretrained=pre_trained, pretrained_path=pretrained_path,**kwargs)

    return model