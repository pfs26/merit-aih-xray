#Module to handle imbalance in the dataset
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
 
import utils


# ─────────────────────────────────────────────
# 1. CLASS WEIGHTS
# ─────────────────────────────────────────────

def get_class_weights(train_set):
    """
    Function that computes inverse-frequency class weights for a given training dataset.

    """
    targets=train_set.targets

    class_counts = torch.tensor(
        [targets.count(i) for i in range(len(train_set.classes))],
        dtype=torch.float
    )
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    return class_weights.to(utils.device)
  



    