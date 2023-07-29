import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,Linear
from utils import normalize_A, generate_cheby_adj
# A   - adjancecy matrix
# fc    - full connection
# L     - normalized Laplacian

class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            # jvmrf gera tantas layers quanto K especificado.
            self.gc.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        # L = Normalized Laplacian Matrix, 
        # x = Train/Test data
        # STEP 6 - Calculate Chebyshev Polynomial Items,
        # Qual a dimensao de ADJ? 
        adj = generate_cheby_adj(L, self.K)
        # STEP 7 - SUM(THETA_k*T_k[Normalized Laplacian Matrix])
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        # STEP 8 - 1x1 conv results and regularization using Relu
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        # STEP 1 - initialize other model parameters
        # ------- Dynamical Graph convolution layer (K=2, therefore 2 layers)
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        # ------- ? ref: BatchNorm1d is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch.
        # makes the DNN training faster and more stable
        self.BN1 = nn.BatchNorm1d(in_channels) # jvmrf ACHO QUE E A CONVOLUTION LAYER de 1x1
        # ------- Full connection layer
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        # STEP 1 - initialize A = Adjacency Matrix
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes)) # me parece ser NxN - adj matrix - do comentario do git.
        nn.init.uniform_(self.A,0.01,0.5)
        # inicializada com valores nulos? quais os valores de A?

    # train.py - line 85 - output = model(train_data_tensor)  
    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        # STEP 3,4 & 5 - Calculate normalized Laplacian from  A (Adjacency Matrix)
        L = normalize_A(self.A)
        # jvmrf - o que e essa combinacao de BN1 + Laplaciana normalizada?
        # STEP 6,7 & 8 - Calculate Chebyshed Polynomial, SUM and Conv. Results.
        result = self.layer1(x, L) # Chebynet.forward()
        result = result.reshape(x.shape[0], -1)
        # STEP 9 - Calculating result of FULL CONNECTION LAYER
        result = self.fc(result) # Linear.foward()
        return result
