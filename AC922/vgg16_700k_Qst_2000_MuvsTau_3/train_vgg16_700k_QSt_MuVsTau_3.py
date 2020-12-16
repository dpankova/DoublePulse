# Trains a three string "fan in" model
#
# Aaron Fienberg
# April 2020

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

import time
import sys
import tensorflow as tf
import tensorflow.keras as keras

from glob import glob


sys.path.append('../')
from cnn_common.build_and_train import build_and_train_model
from cnn_common.taucnn_models import three_chan_adapter, CNN_vgg16

model_name = 'vgg16_700k_QSt2000_dataset_norm_MuVsTau_3'

training_conf = {
    'n_training' : 350000,
    'n_validation': 35000,
    'initial_rate': 0.01,
    'momentum': 0.9,
    'n_epochs': 100,
    'patience': 10,
    'norm_mode': 'whole_dataset',
    'batch_size': 256,
    'validation_files_per_flavor': [1, 3],
    'e_file_list': glob('/gpfs/summit/home/dup193/TauCNN/images/charge_numu/Images_NuMu*'),
    'tau_file_list': glob('/gpfs/summit/home/abf5460/tauCNN/threeStringsQSt2000/NuTau*'),
    'model_factory': CNN_vgg16,
    'optimizer': 'sgd',	
    'input_adapter': three_chan_adapter
}

initial_rate = training_conf['initial_rate']
def scheduler(epoch):
    if epoch < 5:
        return initial_rate
    else:
        # taken/adpated from https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
        return initial_rate * tf.math.exp(0.05*(5-epoch))

training_conf['scheduler'] = scheduler
    
def main():        
    model = build_and_train_model(**training_conf)
    
    model.save(f'{model_name}.h5')

if __name__ == '__main__':
    sys.exit(main())
