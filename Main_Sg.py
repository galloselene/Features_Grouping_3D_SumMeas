
import numpy as np
from architectures import Net, LeNet
from solver import Solver
import pdb
import os
from my_functions import * #create_k_splits, load_ABIDE_data, prepare_masker



# ----- prepare data

def train_test_crossval(SummaryMeasure, k,
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation, n_cluster=n_clusters, dropout=dropout,
                    
                    regularizer = regularizer__, learning_rate=learning_rate, n_epochs=n_epochs, 
                    batch_size=batch_size, seed=seed, n_phi=n_phi,
                    n_sample_group=n_sample_group, lambda_l2=0,
                    lambda_l1=0, early_stopping=early_stopping, display=display
                    ):
    
    if Dataset == 'ABIDE':
        X_train, X_test, y_train, y_test, masker = prepare_ABIDE_dataset(SummaryMeasure, k)

    # Construct MLP with Feature Grouping
    n_features = X_train.shape[1]
    n_output = len(np.unique(y_train))
    net_mlp_FG = Net(n_features, n_output, hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation, n_cluster=n_clusters, dropout=dropout)


    solver = Solver(net_mlp_FG, regularizer = regularizer__, learning_rate=learning_rate, n_epochs=n_epochs, 
                    batch_size=batch_size, seed=seed, n_phi=n_phi,
                    n_sample_group=n_sample_group, masker=masker, lambda_l2=0,
                    lambda_l1=0, early_stopping=early_stopping, display=display)

    X_train=X_train.astype('int64')
    X_test=X_test.astype('int64')

    accuracy_train, accuracy_test = solver.train_numpy(X_train, y_train, X_test, y_test)

    return accuracy_train, accuracy_test

#------------------
Dataset = 'ABIDE'
SummaryMeasure ='reho'
# ['T1', 'alff', 'autocorr', 'degree_centrality_binarize', 'degree_centrality_weighted', 'eigenvector_central..._binarize', 'eigenvector_central..._weighted', 'entropy', 'falff', 'lfcd_binarize', 'lfcd_weighted', 'reho', 'vmhc']


# Feature Grouping Parameters
regularizer = ['ReNa']
n_clusters = 5708  # k in the manuscript set at 10% of the features, here is 20%
n_sample_group = 50 # r in the manuscript
n_phi = 1 # 100 (b) in the manuscript
dropout = 0.0 # let's apply dropout in the second layer

# MLP Parameters
hidden_layer_sizes = [256] #[2048]#
activation = 'relu'

# Training Parameters
learning_rate = 0.05
batch_size = 64
seed = 10003
early_stopping = True
n_epochs = 100
display = 5#100

# save result
bd = "/data/sgallo/Features_Grouping_3D_SumMeas/Results/" 
results_path = os.path.join(bd, regularizer[0], Dataset)
csvname= results_path + '/' + SummaryMeasure + '_5foldCV.csv'
d= {'Summary measure': [], 'Kfold': [], 
                'Accuracy training': [], 'Accuracy test': []}   
df_results = pd.DataFrame(data=d)

for k in range(5):
    print('--------------- Executing fold #{}'.format(k+1))
    accuracy_train, accuracy_test = train_test_crossval(SummaryMeasure, k,
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation, n_cluster=n_clusters, dropout=dropout,
                    
                    regularizer = regularizer__, learning_rate=learning_rate, n_epochs=n_epochs, 
                    batch_size=batch_size, seed=seed, n_phi=n_phi,
                    n_sample_group=n_sample_group, lambda_l2=0,
                    lambda_l1=0, early_stopping=early_stopping, display=display)

    # save results in dataframe
    r= {'Summary measure': [SummaryMeasure], 'Kfold': [k+1], 
    'Accuracy training': [accuracy_train], 'Accuracy test': [accuracy_test]}  
    df_r = pd.DataFrame(data=r)
    df_results=pd.concat([df_results, df_r])        
    
    # save the table on disk evey cv
    export_csv = df_results.to_csv (csvname, index = None, header=True) 