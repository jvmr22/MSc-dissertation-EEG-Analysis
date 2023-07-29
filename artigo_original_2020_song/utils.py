import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
simplefilter("ignore", category=ConvergenceWarning)

all_freqs_dict = {
    "delta":0,
    "theta":1,
    "alpha":2,
    "beta":3,
    "gamma":4}

def normalize_A(A,lmax=2):
    # STEP 3 - Regularizing ADJ matrix using Relu
    A=F.relu(A)
    N=A.shape[0]
    # A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A*(torch.ones(N,N)-torch.eye(N,N))
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    # STEP 4 - Calculate Laplacian Matrix
    # L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D) # Laplacian?
    L = torch.eye(N,N)-torch.matmul(torch.matmul(D, A), D) # Laplacian?
    # STEP 5 - Calculate normalized Laplacian
    # Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    Lnorm=(2*L/lmax)-torch.eye(N,N)
    return Lnorm

# STEP 6 - Calculate Chebyshev Polynomial
def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(L.shape[-1]).cuda())
            support.append(torch.eye(L.shape[-1]))
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support

def generating_data(data_dict, clip_label, feature_name, frequencies_selection=['all']):
    # first 9 movies as training, the last 6 movies as testing
    train_data = data_dict[feature_name+'1']
    print("\t### TRAIN ###")
    _, num, _ = train_data.shape 
    train_label = np.zeros(num,) + clip_label[0]
    train_data = np.swapaxes(train_data, 0, 1)
    if 'all' in frequencies_selection:
        # train_data = np.reshape(train_data, (num, -1))
        pass
    else:
        # apenas uma frequencia por vez
        # TODO processamento paralelo de cada frequencia
        frequencie_idx = all_freqs_dict[frequencies_selection[0]]
        print(f"## select only {frequencies_selection} freq - idx: {frequencie_idx} ##")
        train_data = train_data[:,:,frequencie_idx]
    print(f"\t\t{feature_name + '1'} - video 1 - {train_data.shape} observacoes")
    train_residual_index = [2,3,4,5,6,7,8,9]
    for ind,i in enumerate(train_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape 
        used_label = np.zeros(num,) + clip_label[ind+1]
        used_data = np.swapaxes(used_data, 0, 1)
        if 'all' in frequencies_selection:
            # used_data = np.reshape(used_data, (num, -1))
            pass
        else:
            # apenas uma frequencia por vez
            # TODO processamento paralelo de cada frequencia
            frequencie_idx = all_freqs_dict[frequencies_selection[0]]
            used_data = used_data[:,:,frequencie_idx]
            
        print(f"\t\t{feature_name + str(i)} - video {str(i)} - {used_data.shape} observacoes")
        train_data = np.vstack((train_data, used_data))
        train_label = np.hstack((train_label, used_label))
    
    test_data = data_dict[feature_name+'10']
    print("\t### TEST ###")
    _, num, _ = test_data.shape 
    test_label = np.zeros(num,) + clip_label[9]
    test_data = np.swapaxes(test_data, 0, 1)
    if 'all' in frequencies_selection:
        # test_data = np.reshape(test_data, (num, -1))
        pass
    else:
        # apenas uma frequencia por vez
        # TODO processamento paralelo de cada frequencia
        frequencie_idx = all_freqs_dict[frequencies_selection[0]]
        test_data = test_data[:,:,frequencie_idx]
    print(f"\t\t{feature_name + '10'} - video 10 - {test_data.shape} observacoes")
    test_residual_index = [11,12,13,14,15]
    for ind,i in enumerate(test_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape 
        used_label = np.zeros(num,) + clip_label[ind+10]
        used_data = np.swapaxes(used_data, 0, 1)
        if 'all' in frequencies_selection:
            # used_data = np.reshape(used_data, (num, -1))
            pass
        else:
            # apenas uma frequencia por vez
            # TODO processamento paralelo de cada frequencia
            frequencie_idx = all_freqs_dict[frequencies_selection[0]]
            used_data = used_data[:,:,frequencie_idx]
        # used_data = np.reshape(used_data, (num, -1))
        print(f"\t\t{feature_name + str(i)} - video {str(i)} - {used_data.shape} observacoes")
        test_data = np.vstack((test_data, used_data))
        test_label = np.hstack((test_label, used_label))
    return train_data, test_data, train_label, test_label