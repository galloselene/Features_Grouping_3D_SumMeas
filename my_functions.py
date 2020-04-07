import pandas as pd
import h5py as h5
import numpy as np
from sklearn.model_selection import KFold

import torch

import nibabel
from nilearn.input_data import NiftiMasker


datafile = '/data_local/deeplearning/ABIDE_SummaryMeasures/fmri_summary_abideI_II.hdf5'


def select_sbj_id():
    # select data that passed QC
# https://www.sciencedirect.com/science/article/pii/S1053811919304963#bib28

# select data from paper list
    filename=r'/data/sgallo/abide/Model_130120/AbideI_sbjID_Khola_v2.csv'
    select_ID_I=pd.read_csv(filename, header=None)

    filename=r'/data/sgallo/abide/Model_130120/Abide_I_II_sbjID_Khola_v2.csv'
    select_ID_I_II=pd.read_csv(filename, header=None)

    select_ID=pd.concat([select_ID_I, select_ID_I_II]) 
    select_ID=select_ID.to_numpy()

    hfile = h5.File(datafile)
    cid=hfile['summaries'].attrs['SUB_ID']

    select_ID_i=[]
    missing_ID=[]
    for s in select_ID:
        itemindex=np.where(cid == s)
        if np.size(itemindex)==0:
            missing_ID.append(s)
        else: 
            select_ID_i.append(itemindex[0][0])

    return select_ID_i

def create_k_splits(n_splits=5):
    kf = KFold(n_splits=n_splits,random_state=42, shuffle=True)

    n_subjects = 1162  #len(data) 
    X = np.zeros((n_subjects, 1))

    train_split = []
    test_split = []

    for i_split, (train_list, test_list) in enumerate(kf.split(X)):

        train_split.append(train_list)
        test_split.append(test_list)

    return train_split, test_split


#class load_data(Dataset):
def load_data(  split,
                train_split, test_split,
                summaryMeasure
                 ):
        
    hfile = h5.File(datafile)
    select_ID_i = select_sbj_id()
    
    data = hfile['summaries/'+summaryMeasure].value[select_ID_i] 
    label= hfile['summaries'].attrs['DX_GROUP'][select_ID_i]
    
    dataset_size = len(label)
    
    #RANDOMIZATION OF THE DATA WITH SEED
    np.random.seed(23)
    idx = np.arange(len(data))
    np.random.shuffle(idx)

    data= data[idx]
    label= label[idx]
    hfile.close()
    
    trData=data[train_split]
    meanTrData = np.mean(trData, axis=0)
    stdTrData = np.std(trData, axis=0)
    
    # SPLIT DATA               
    if split=='train':
        data = (trData - meanTrData)/(stdTrData + 1.0)
        label = label[train_split]
        print('TRAINING n ads {}/{}'.format((label==1).sum(),len(label)))
    elif split =='test':
        testData = data[test_split]
        data = (testData - meanTrData)/(stdTrData + 1.0)
        label = label[test_split]
    else:
            raise ValueError('Error! the split name is not recognized')
                         
    return data, label      

def prepare_masker(data):
    n_x, n_y, n_z = data.shape[1], data.shape[2], data.shape[3] 
    p = np.prod([n_x, n_y, n_z, 1])
    mask = np.zeros(p).astype(np.bool)
    mask[:data.shape[-1]] = 1
    mask = mask.reshape([n_x, n_y, n_z, 1])
    mask = np.ones((n_x, n_y, n_z))
    affine = np.eye(4)
    mask=np.load("brain_mask", allow_pickle= True)
    mask_img = nibabel.Nifti1Image(mask.astype(np.float), affine)
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()
    return masker