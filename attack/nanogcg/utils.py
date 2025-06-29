import functools
import gc
import inspect
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

