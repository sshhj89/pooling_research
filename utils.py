import torch
from typing import List
import torch.nn as nn

def save_checkpoint(state,filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)