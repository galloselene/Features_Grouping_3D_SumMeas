
import numpy as np
from architectures import Net, LeNet
from solver import Solver
import pdb

from my_functions import create_k_splits, load_data, prepare_masker


SummaryMeasure ='reho'
# MLP Parameters
hidden_layer_sizes = [256]
activation = 'relu'

# Feature Grouping Parameters
n_clusters = 820 # k in the manuscript
n_sample_rena = 50 # r in the manuscript
n_phi = 5 # b in the manuscript
dropout = 0.0 # let's apply dropout in the second layer

# Training Parameters
learning_rate = 0.05
batch_size = 64
seed = 10003
early_stopping = True
n_epochs = 1000
display = 2#100

# ----- prepare data
k=1 # work on 1st k fold for testing purpose

train_split, test_split = create_k_splits()
X_train, y_train = load_data('train', train_split[k], test_split[k], SummaryMeasure)
X_test, y_test = load_data('test', train_split[k], test_split[k], SummaryMeasure)

# prepare mask and masker 
mask=np.load("brain_mask", allow_pickle= True)
masker = prepare_masker(X_train)
# -------
# flatten data for mlp
X_train = X_train.reshape(len(X_train),-1)
X_test = X_test.reshape(len(X_test),-1)

X_train=X_train[:,mask.reshape(-1)]
X_test = X_test[:,mask.reshape(-1)]
# Construct MLP with Feature Grouping
n_features = X_train.shape[1]
n_output = len(np.unique(y_train))
net_mlp_FG = Net(n_features, n_output, hidden_layer_sizes=hidden_layer_sizes,
                 activation=activation, n_cluster=n_clusters, dropout=dropout)


solver = Solver(net_mlp_FG, learning_rate=learning_rate, n_epochs=n_epochs, 
                batch_size=batch_size, seed=seed, n_phi=n_phi,
                n_sample_rena=n_sample_rena, masker=masker, lambda_l2=0,
                lambda_l1=0, early_stopping=early_stopping, display=display)

X_train=X_train.astype('int64')
X_test=X_test.astype('int64')

solver.train_numpy(X_train, y_train, X_test, y_test)