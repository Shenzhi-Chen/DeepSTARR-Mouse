
######
# Shenzhi Chen (2024)
######

#########
### Load arguments
#########

import sys, getopt

def main(argv):
   model_ID_output = ''
   try:
      opts, args = getopt.getopt(argv,"hi:w:v:x:a:o:y:z:l:")
   except getopt.GetoptError:
      print('ledidi_enhancer_design.py -i <reference> -w <number_generate> -v <enhancer_model_path> -x <access_model_path> -a <output> -o <if start from randomly sequence(1/0)> -y <deseried access> -z <deseried act> -l <lamda>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('ledidi_enhancer_design.py -i <reference> -w <number_generate> -v <enhancer_model_path> -x <access_model_path> -a <output> -o <if start from randomly sequence(1/0)> -y <deseried access> -z <deseried act> -l <lamda>')
         sys.exit()
      elif opt in ("-i"):
         reference = arg
      elif opt in ("-w"):
         number_generate = arg
      elif opt in ("-v"):
         enhancer_model_path = arg
      elif opt in ("-x"):
         access_model_path = arg
      elif opt in ("-a"):
         output = arg
      elif opt in ("-o"):
         ran_gen = arg
      elif opt in ("-y"):
         access = arg
      elif opt in ("-z"):
         act = arg
      elif opt in ("-l"):
         lamda = arg     
            
   if reference=='': sys.exit("reference not found")
   if number_generate=='': sys.exit("number_generate not found")
   if enhancer_model_path=='': sys.exit("enhancer_model_path not found")
   if access_model_path=='': sys.exit("access_model_path not found")  
   if output=='': sys.exit("output not found")
   if ran_gen=='': sys.exit("ran_gen not found")
   if access=='': sys.exit("deseried access not found")
   if act=='': sys.exit("deseried act not found")
   if lamda=='': sys.exit("lamda not found")

   print('reference ', reference)
   print('number_generate ', number_generate)
   print('enhancer_model_path ', enhancer_model_path)
   print('access_model_path ', access_model_path) 
   print('output ', output)
   print('ran_gen ', ran_gen)
   print('access ', access)
   print('act ', act)
   print('lamda ', lamda)
    
   return reference, number_generate, enhancer_model_path, access_model_path, output, ran_gen, access, act, lamda

if __name__ == "__main__":
   reference, number_generate, enhancer_model_path, access_model_path, output, ran_gen, access, act, lamda = main(sys.argv[1:])

#########
### Load libraries
#########

Y_access = float(access)
Y_act = float(act)
lamda = float(lamda)
number_generate = int(number_generate)

print('variable reference ', reference)
print('lamda ', lamda)
print('desired access ', Y_access)
print('desired activity logit ', Y_act)
print('variable number_generate ', number_generate)
print('variable enhancer model ', enhancer_model_path)
print('variable access model ', access_model_path)
print('variable output ', output)
print('variable ran_gen ', ran_gen)
print('desired access ', access)
print('desired act ', act)

import numpy as np
import numpy
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import datetime
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, InputLayer, Input, GlobalAvgPool1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import plot_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List
from tensorflow.keras.models import model_from_json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

import sys
sys.path.append('bin/')
sys.path.append('bin/Neural_Network_DNA_Demo/')
from helper import IOHelper, SequenceHelper # from https://github.com/bernardo-de-almeida/Neural_Network_DNA_Demo.git
sys.path.append('bin/Taskiran_et_al_Supplemental_Code/General/')
import utils
import random

def write_sequences_to_fasta_original_format(X_hat, raw_seqs, y_orign, y_hat_orign, input_loss, output_file):
    """
    Write designed sequences to a FASTA file with detailed headers (original format)
    For X_hat and raw_seqs with batch_size=X_hat.shape[0] and seq_len=X_hat.shape[2]
    """
    # Move tensors to CPU and convert to numpy if needed
    X_hat_np = X_hat.detach().cpu().numpy() if hasattr(X_hat, 'cpu') else X_hat
    raw_seqs_np = raw_seqs.detach().cpu().numpy() if hasattr(raw_seqs, 'cpu') else raw_seqs

    idx_to_nuc = {0: "A", 1: "C", 2: "G", 3: "T"}
    batch_size = X_hat_np.shape[0]
    seq_len = X_hat_np.shape[2]

    with open(output_file, "w") as fasta_file:
        for i in range(batch_size):
            header = f">ledidi designed enhancer {i}"
            header += f"[mutation steps:{input_loss[i]}]"
            header += f"[score:{y_orign[i][0]}|{y_orign[i][1]}>>{y_hat_orign[i][0]}|{y_hat_orign[i][1]}]["

            # Process mutations
            for j in range(seq_len):
                # Find which nucleotide is 1 in raw and designed
                raw_nuc_vec = raw_seqs_np[i, :, j]
                new_nuc_vec = X_hat_np[i, :, j]

                # Use .get() to handle any out-of-range indices
                raw_nuc_idx = np.argmax(raw_nuc_vec)
                new_nuc_idx = np.argmax(new_nuc_vec)

                if raw_nuc_idx != new_nuc_idx:
                    raw_nuc = idx_to_nuc.get(raw_nuc_idx, "N")
                    new_nuc = idx_to_nuc.get(new_nuc_idx, "N")
                    header += f"position_{j+1}:{raw_nuc}>{new_nuc}｜"

            header += "]\n"
            fasta_file.write(header)

            # Write sequence
            sequence = ""
            for j in range(seq_len):
                nuc_vec = X_hat_np[i, :, j]
                nuc_idx = np.argmax(nuc_vec)
                sequence += idx_to_nuc.get(nuc_idx, "N")  # Fixed: was 'idx' instead of 'idx_to_nuc.get(nuc_idx, "N")'

            fasta_file.write(sequence + "\n")
            
def load_model(model_path):
    import deeplift
    from tensorflow.keras.models import model_from_json
    keras_model_weights = model_path + '.h5'
    keras_model_json = model_path + '.json'
    keras_model = model_from_json(open(keras_model_json).read())
    keras_model.load_weights(keras_model_weights)
    #keras_model.summary()
    return keras_model, keras_model_weights, keras_model_json

def transfer_weights(model2, tfweights):
    """
    transfer Keras weights to PyTorch model
    """
    state_dict = model2.state_dict()

    mapping = {
        # Conv1 + BN1
        'conv1.0.weight': (0, lambda x: x.transpose(2,1,0)),  # (7,4,256) -> (256,4,7)
        'conv1.0.bias': (1, None),
        'conv1.1.weight': (2, None),          # BN gamma
        'conv1.1.bias': (3, None),            # BN beta
        'conv1.1.running_mean': (4, None),    # BN moving_mean
        'conv1.1.running_var': (5, None),     # BN moving_variance

        # Conv blocks
        'conv_blocks.0.0.weight': (6, lambda x: x.transpose(2,1,0)),
        'conv_blocks.0.0.bias': (7, None),
        'conv_blocks.0.1.weight': (8, None),
        'conv_blocks.0.1.bias': (9, None),
        'conv_blocks.0.1.running_mean': (10, None),
        'conv_blocks.0.1.running_var': (11, None),

        'conv_blocks.1.0.weight': (12, lambda x: x.transpose(2,1,0)),
        'conv_blocks.1.0.bias': (13, None),
        'conv_blocks.1.1.weight': (14, None),
        'conv_blocks.1.1.bias': (15, None),
        'conv_blocks.1.1.running_mean': (16, None),
        'conv_blocks.1.1.running_var': (17, None),

        'conv_blocks.2.0.weight': (18, lambda x: x.transpose(2,1,0)),
        'conv_blocks.2.0.bias': (19, None),
        'conv_blocks.2.1.weight': (20, None),
        'conv_blocks.2.1.bias': (21, None),
        'conv_blocks.2.1.running_mean': (22, None),
        'conv_blocks.2.1.running_var': (23, None),

        # Dense blocks
        'dense_blocks.0.0.weight': (24, lambda x: x.transpose(1,0)),
        'dense_blocks.0.0.bias': (25, None),
        'dense_blocks.0.1.weight': (26, None),
        'dense_blocks.0.1.bias': (27, None),
        'dense_blocks.0.1.running_mean': (28, None),
        'dense_blocks.0.1.running_var': (29, None),

        'dense_blocks.1.0.weight': (30, lambda x: x.transpose(1,0)),
        'dense_blocks.1.0.bias': (31, None),
        'dense_blocks.1.1.weight': (32, None),
        'dense_blocks.1.1.bias': (33, None),
        'dense_blocks.1.1.running_mean': (34, None),
        'dense_blocks.1.1.running_var': (35, None),

        # Output layer
        'output_weight': (36, lambda x: x.transpose(1,0)),  # (256,1) -> (1,256)
        'output_bias': (37, None),
    }

    # transfer weights
    for key, (idx, transform) in mapping.items():
        if key in state_dict:
            weight = tfweights[idx]
            if transform is not None:
                weight = transform(weight)
            try:
                state_dict[key] = torch.from_numpy(weight).to(torch.float32)
            except Exception as e:
                print(f"Error transferring {key}: {e}")
                print(f"TF weight shape: {weight.shape}")
                print(f"PyTorch expected shape: {state_dict[key].shape}")
                raise

    # load weights
    model2.load_state_dict(state_dict)
    return model2

params_DeepSTARR2_access = {
    'batch_size': 128,
    'epochs': 100,
    'early_stop': 10,
    'kernel_size1': 7,
    'kernel_size2': 3,
    'kernel_size3': 3,
    'kernel_size4': 3,
    'lr': 0.005,
    'num_filters': 256,
    'num_filters2': 120,
    'num_filters3': 60,
    'num_filters4': 60,
    'n_conv_layer': 4,
    'n_add_layer': 2,
    'dropout_prob': 0.5,  
    'dense_neurons1': 64,
    'dense_neurons2': 256,
    'pad': 'same',
    'act': 'relu'
}

class DeepSTARR2_access(nn.Module):
    def __init__(self, params=params_DeepSTARR2_access):
        super(DeepSTARR2_access, self).__init__()
        self.params = params


        self.bn_params = {'momentum': 0.99, 'eps': 1e-3}

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=params['num_filters'],
                kernel_size=params['kernel_size1'],
                padding=(params['kernel_size1'] - 1) // 2  # same padding
            ),
            nn.BatchNorm1d(params['num_filters'], **self.bn_params),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3)
        )

        # Additional convolutional blocks
        self.conv_blocks = nn.ModuleList()
        input_channels = params['num_filters']
        for i in range(1, params['n_conv_layer']):
            output_channels = params[f'num_filters{i+1}']
            kernel_size = params[f'kernel_size{i+1}']
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2  # same padding
                ),
                nn.BatchNorm1d(output_channels, **self.bn_params),
                nn.ReLU(),
                nn.MaxPool1d(3, stride=3)
            ))
            input_channels = output_channels

        # Dense layers
        self.flatten_size = self._get_flatten_size()

        self.dense_blocks = nn.ModuleList()
        input_size = self.flatten_size
        for i in range(params['n_add_layer']):
            output_size = params[f'dense_neurons{i+1}']
            self.dense_blocks.append(nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size, **self.bn_params),
                nn.ReLU(),
                nn.Dropout(params['dropout_prob'])
            ))
            input_size = output_size

        # Output layer (linear, no sigmoid)
        self.output_weight = nn.Parameter(torch.Tensor(1, input_size))
        self.output_bias = nn.Parameter(torch.Tensor(1))

    def _get_flatten_size(self):
        x = torch.randn(1, 4, 1001)
        x = self.conv1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x.numel()

    def forward(self, x):
        # (batch, 1001, 4) -> (batch, 4, 1001)
        if x.shape[1] == 1001 and x.shape[2] == 4:
            x = x.transpose(1, 2)

        x = self.conv1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = x.transpose(1, 2)
        x = torch.flatten(x, 1)

        for dense_block in self.dense_blocks:
            x = dense_block(x)

        x = torch.nn.functional.linear(x, self.output_weight, self.output_bias)
        return x
    
def transfer_weights_access(model, tfweights):
    """
    Transfer Keras weights to PyTorch accessibility model
    (Following the same approach as the successful enhancer model)
    """
    state_dict = model.state_dict()

    # Complete weight mapping table (identical structure to enhancer model)
    mapping = {
        # Conv1 + BN1
        'conv1.0.weight': (0, lambda x: x.transpose(2,1,0)),  # (7,4,256) -> (256,4,7)
        'conv1.0.bias': (1, None),
        'conv1.1.weight': (2, None),          # BN gamma
        'conv1.1.bias': (3, None),            # BN beta
        'conv1.1.running_mean': (4, None),    # BN moving_mean
        'conv1.1.running_var': (5, None),     # BN moving_variance

        # Conv blocks
        'conv_blocks.0.0.weight': (6, lambda x: x.transpose(2,1,0)),
        'conv_blocks.0.0.bias': (7, None),
        'conv_blocks.0.1.weight': (8, None),
        'conv_blocks.0.1.bias': (9, None),
        'conv_blocks.0.1.running_mean': (10, None),
        'conv_blocks.0.1.running_var': (11, None),

        'conv_blocks.1.0.weight': (12, lambda x: x.transpose(2,1,0)),
        'conv_blocks.1.0.bias': (13, None),
        'conv_blocks.1.1.weight': (14, None),
        'conv_blocks.1.1.bias': (15, None),
        'conv_blocks.1.1.running_mean': (16, None),
        'conv_blocks.1.1.running_var': (17, None),

        'conv_blocks.2.0.weight': (18, lambda x: x.transpose(2,1,0)),
        'conv_blocks.2.0.bias': (19, None),
        'conv_blocks.2.1.weight': (20, None),
        'conv_blocks.2.1.bias': (21, None),
        'conv_blocks.2.1.running_mean': (22, None),
        'conv_blocks.2.1.running_var': (23, None),

        # Dense blocks
        'dense_blocks.0.0.weight': (24, lambda x: x.transpose(1,0)),
        'dense_blocks.0.0.bias': (25, None),
        'dense_blocks.0.1.weight': (26, None),
        'dense_blocks.0.1.bias': (27, None),
        'dense_blocks.0.1.running_mean': (28, None),
        'dense_blocks.0.1.running_var': (29, None),

        'dense_blocks.1.0.weight': (30, lambda x: x.transpose(1,0)),
        'dense_blocks.1.0.bias': (31, None),
        'dense_blocks.1.1.weight': (32, None),
        'dense_blocks.1.1.bias': (33, None),
        'dense_blocks.1.1.running_mean': (34, None),
        'dense_blocks.1.1.running_var': (35, None),

        # Output layer
        'output_weight': (36, lambda x: x.transpose(1,0)),  # (256,1) -> (1,256)
        'output_bias': (37, None),
    }

    # Transfer weights - using the exact same approach as the enhancer model
    for key, (idx, transform) in mapping.items():
        if key in state_dict:
            weight = tfweights[idx]
            if transform is not None:
                weight = transform(weight)
            try:
                state_dict[key] = torch.from_numpy(weight).to(torch.float32)
            except Exception as e:
                print(f"Error transferring {key}: {e}")
                print(f"TF weight shape: {weight.shape}")
                print(f"PyTorch expected shape: {state_dict[key].shape}")
                raise

    # Load weights
    model.load_state_dict(state_dict)
    return model

params_DeepSTARR2_enhancer = {
    'batch_size': 128,
    'epochs': 100,
    'early_stop': 10,
    'kernel_size1': 7,
    'kernel_size2': 3,
    'kernel_size3': 3,
    'kernel_size4': 3,
    'lr': 0.005,
    'num_filters': 256,
    'num_filters2': 120,
    'num_filters3': 60,
    'num_filters4': 60,
    'n_conv_layer': 4,
    'n_add_layer': 2,
    'dropout_prob': 0.5,  
    'dense_neurons1': 64,
    'dense_neurons2': 256,
    'pad': 'same',
    'act': 'relu'
}

class DeepSTARR2_enhancer(nn.Module):
    def __init__(self, params=params_DeepSTARR2_enhancer):
        super(DeepSTARR2_enhancer, self).__init__()
        self.params = params

        self.bn_params = {'momentum': 0.99, 'eps': 1e-3}

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=params['num_filters'],
                kernel_size=params['kernel_size1'],
                padding=(params['kernel_size1'] - 1) // 2  # same padding
            ),
            nn.BatchNorm1d(params['num_filters'], **self.bn_params),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )

        # Additional convolutional blocks
        self.conv_blocks = nn.ModuleList()
        input_channels = params['num_filters']
        for i in range(1, params['n_conv_layer']):
            output_channels = params[f'num_filters{i+1}']
            kernel_size = params[f'kernel_size{i+1}']
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2  # same padding
                ),
                nn.BatchNorm1d(output_channels, **self.bn_params),
                nn.ReLU(),
                nn.MaxPool1d(3)
            ))
            input_channels = output_channels

        # Dense layers
        self.flatten_size = self._get_flatten_size()

        # Dense blocks with consistent BN parameters
        self.dense_blocks = nn.ModuleList()
        input_size = self.flatten_size
        for i in range(params['n_add_layer']):
            output_size = params[f'dense_neurons{i+1}']
            self.dense_blocks.append(nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size, **self.bn_params),
                nn.ReLU(),
                nn.Dropout(params['dropout_prob'])
            ))
            input_size = output_size

        # Output layer 
        self.output_weight = nn.Parameter(torch.Tensor(1, input_size))
        self.output_bias = nn.Parameter(torch.Tensor(1))
        self.sigmoid = nn.Sigmoid()

    def _get_flatten_size(self):
        x = torch.randn(1, 4, 1001)
        x = self.conv1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x.numel()

    def forward(self, x):
        if x.shape[1] == 1001 and x.shape[2] == 4:
            x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Flatten 
        x = x.transpose(1, 2)  
        x = torch.flatten(x, 1)  

        # Dense layers
        for dense_block in self.dense_blocks:
            x = dense_block(x)

        # Output layer
        x = torch.nn.functional.linear(x, self.output_weight, self.output_bias)
        x = self.sigmoid(x)

        return x
   
class DeepSTARR2_enhancer_logits(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        # Copy all layers from the original model except sigmoid
        self.params = original_model.params
        self.bn_params = original_model.bn_params

        # Copy convolutional layers
        self.conv1 = original_model.conv1
        self.conv_blocks = original_model.conv_blocks

        # Copy dense layers
        self.flatten_size = original_model.flatten_size
        self.dense_blocks = original_model.dense_blocks

        # Copy output parameters (weights and bias)
        self.output_weight = original_model.output_weight
        self.output_bias = original_model.output_bias

        # No sigmoid layer

    def forward(self, x):
        # Ensure input shape is correct (batch, 1001, 4) -> (batch, 4, 1001)
        if x.shape[1] == 1001 and x.shape[2] == 4:
            x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Flatten (same as original)
        x = x.transpose(1, 2)
        x = torch.flatten(x, 1)

        # Dense layers
        for dense_block in self.dense_blocks:
            x = dense_block(x)

        # Output layer (no sigmoid)
        x = torch.nn.functional.linear(x, self.output_weight, self.output_bias)
        return x
    
#########
#convert tf2 to torch
#########

MODEL_enhancer = enhancer_model_path+".json"
parameter_enhancer = enhancer_model_path+".h5"
with open(MODEL_enhancer, 'r') as f:
    json_config_enhancer = f.read()
    
print('Loading model...')

modelt_enhancer = model_from_json(json_config_enhancer)
modelt_enhancer.compile()
modelt_enhancer.load_weights(parameter_enhancer)


enhancer_model = DeepSTARR2_enhancer()

tfweights_enhancer = modelt_enhancer.get_weights()
enhancer_model = transfer_weights(enhancer_model, tfweights_enhancer)
enhancer_model.eval()
enhancer_model = enhancer_model

MODEL_access = access_model_path+".json"
parameter_access = access_model_path+".h5"
with open(MODEL_access, 'r') as f:
    json_config_access = f.read()

print('Loading accessibility model...')

modelt_access = model_from_json(json_config_access)
modelt_access.compile()
modelt_access.load_weights(parameter_access)

# Create PyTorch accessibility model
access_model = DeepSTARR2_access()

# Transfer weights
tfweights_access = modelt_access.get_weights()
access_model = transfer_weights_access(access_model, tfweights_access)
access_model.eval()

# remove the last layer from enhancer prediction model
enhancer_model_without_last_layer = DeepSTARR2_enhancer_logits(enhancer_model)
enhancer_model_without_last_layer.eval()




#########
### Generate sequences
#########
seq_len = 1001
if ran_gen == '1': #start from random sequence
    print('Generating random sequence...')
    insilico_evolution_dict = {}
    insilico_evolution_dict["regions"] = utils.random_sequence(seq_len, number_generate)
    gen_name = 'random'
    print('Completely random sequence generated.')
else: #start from random sequence adjust by reference sequence dinucleotide distribution
    insilico_evolution_dict = {}
    insilico_evolution_dict["regions"] = utils.random_sequence_dinucleotide_adjusted(seq_len, number_generate, reference)
    gen_name = 'adjusted'
    print('Adjusted sequences generated.')
    
    
X = torch.tensor(insilico_evolution_dict["regions"], dtype=torch.float32).permute(0, 2, 1).cuda()
print(X.shape)

from ledidi.wrappers import DesignWrapper  
access_model = access_model.cuda()
enhancer_model_without_last_layer = enhancer_model_without_last_layer.cuda()
enhancer_model = enhancer_model.cuda()
designer_orign = DesignWrapper([access_model, enhancer_model])
designer = DesignWrapper([access_model, enhancer_model_without_last_layer])
                         
Y = designer(X)

torch.manual_seed(0)
y_bar = torch.tensor([[Y_access,Y_act]], dtype=torch.float32, device='cuda')
from ledidi import ledidi
import numpy

from ledidi import ledidi

# Initialize storage lists
X_hat_list = []
y_hat_list = []
input_loss_list = []
output_loss_list = []

for i in range(number_generate):
    print(f"Processing batch {i+1}")
    
    # Process one batch
    X_hat_batch = ledidi(designer, X[[i]], y_bar, batch_size=1, l=lamda)
    y_hat_batch = designer(X_hat_batch).cpu().detach().numpy()
    
    # Calculate losses for this batch
    input_loss_batch = torch.abs(X_hat_batch - X).sum(axis=(1, 2)).cpu().detach().numpy() // 2
    output_loss_batch = numpy.square(y_bar.cpu().detach().numpy() - y_hat_batch)
    
    # Store results
    X_hat_list.append(X_hat_batch)
    y_hat_list.append(y_hat_batch)
    input_loss_list.append(input_loss_batch)
    output_loss_list.append(output_loss_batch)

# Concatenate all results
X_hat = torch.cat(X_hat_list, dim=0)
y_hat = numpy.concatenate(y_hat_list, axis=0)
input_loss = numpy.concatenate(input_loss_list, axis=0)
output_loss = numpy.concatenate(output_loss_list, axis=0)

print(f"Final shapes: X_hat: {X_hat.shape}, y_hat: {y_hat.shape}")

y_hat_orign = designer_orign(X_hat).cpu().detach().numpy()
y_orign = designer_orign(X).cpu().detach().numpy()

fasta_file = output + gen_name + str(Y_access) + "_" + str(Y_act) + "_" + str(lamda) + "_designed_sequence_evolution.fasta"

write_sequences_to_fasta_original_format(
    X_hat=X_hat,
    raw_seqs=X,
    y_orign=y_orign,
    y_hat_orign=y_hat_orign,
    input_loss=input_loss,
    output_file=fasta_file
)