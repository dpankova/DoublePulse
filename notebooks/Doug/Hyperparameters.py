import numpy as np
import random

def GetHyperparameters(index):
    #
    # For each hyperparameter, either cycle through a list of values,
    # or choose randomly from a range of values.
    #
    # For now, though, set epochs equal to a constant
    #
    Epochs = 1
    #
    # BatchSize comes from a list
    #
    BatchSizes = [32, 64, 96, 128, 192, 256]
    #
    BatchSizeIndex = index % len(BatchSizes)
    #
    BatchSize = BatchSizes[BatchSizeIndex]
    #
    # Continuous variables
    #
    lr_min = 0.001
    lr_max = 0.050
    lr = random.uniform(lr_min,lr_max)
    lr = float("{0:.3f}".format(lr))
    #
    return lr, Epochs, BatchSize
