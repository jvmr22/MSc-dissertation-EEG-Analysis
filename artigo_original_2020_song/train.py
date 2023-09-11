import glob
import os
from datetime import datetime

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from model import DGCNN
from sklearn import preprocessing
from utils import generating_data
from glob import glob

lr_dgcnn = 0.001
weight_decay_dgcnn = 5e-4 #tem no codigo do GMSS tmbm
K=2
epochs=20 # DEFFERARD TMBM USA 20
device = 'cpu'

label_list = np.array([1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]) + 1 # -1 for 
fn = 'de_LDS'


full_path = "/home/joaovmrf/Documents/mestrado/SEED/SEED_EEG/SEED_EEG/ExtractedFeatures"
all_files = [each_file for each_file in glob(full_path+'/*.mat') if 'label.mat' not in each_file]
all_files.sort()
# all_files = glob.glob(full_path+"*")


for cur_file in all_files: 
    #item = "15_20130709.mat"
    #item = "13_20140527.mat"
    item = os.path.basename(cur_file)
    cur_user = item.split("_")[0]
    user_session = datetime.strptime(item.split("_")[1][:-4],"%Y%m%d")
    print(f"- reading file: {item}\n--user:{cur_user}, sessions: {user_session}")

    # cur_file = os.path.join(full_path, item)
    all_data = sio.loadmat(cur_file)


    train_data, test_data, train_label, test_label = generating_data(all_data, label_list, fn)

    #train_data = preprocessing.scale(train_data)
    #test_data = preprocessing.scale(test_data)

    train_data_tensor = torch.from_numpy(train_data).to(torch.float)#.to(device)
    test_data_tensor = torch.from_numpy(test_data).to(torch.float)#.to(device)
    train_label_tensor = torch.from_numpy(train_label).to(torch.long)#.to(device)
    test_label_tensor = torch.from_numpy(test_label).to(torch.long)#.to(device)


    # define model
    model = DGCNN(
        in_channels=5, #5 frequencias
        num_electrodes=62, #62 eletrodos
        k_adj=K, #numero de layers - copiando do GMSS - na verdade no GMSS K=ordem de chebyshev
        # out_channels=32, #copiando do GMSS, mas ainda nao entendi
        out_channels=32,
        num_classes=3
    )

    best_test_res = {
        'acc':0,
        'predict_label':None,
        'trur_label':None
    }

    optimizer = optim.Adam(model.parameters(),
                        lr=lr_dgcnn, weight_decay=weight_decay_dgcnn)
    myloss = nn.CrossEntropyLoss()

    for cur_epoch in range(epochs):
        # ordem 1
        # model.train() # como apontar  referencia aos dados de treino? R: E na definicao do modelo
        # optimizer.zero_grad() #etapa comum em todos, zerar o gradiente.

        # output = model(train_data_tensor)
        # #TODO accuracy.
        # #print(output)

        # # import torch.nn.functional as F
        # # loss_train = F.nll_loss(output, train_label)

        # myloss = nn.CrossEntropyLoss()
        # loss = myloss(output, train_label_tensor)
        # loss.backward()
        # optimizer.step()

        # ordem 2
        model.train() # como apontar  referencia aos dados de treino? R: E na definicao do modelo
        output = model(train_data_tensor)
        # STEP 10 - Calculate the loss function
        loss = myloss(output, train_label_tensor)
        
        optimizer.zero_grad() #etapa comum em todos, zerar o gradiente.

        #TODO accuracy.
        #print(output)

        # import torch.nn.functional as F
        # loss_train = F.nll_loss(output, train_label)
        # STEP 11 - Updating the Adj matrix
        loss.backward()
        optimizer.step()
        # Estou sentindo falta do coidgo de atualizacao dessa matriz

        train_acc = torch.sum(torch.argmax(output, 1) == train_label_tensor) / train_data_tensor.shape[0]

        print('Epoch : {} -- TrainLoss : {} -- TrainAcc : {}\n'.format(cur_epoch, loss.cpu().data, train_acc.cpu().data))

    #model.eval()
    test_output = model(test_data_tensor)
    test_loss = myloss(test_output, test_label_tensor)

    test_acc = torch.sum(torch.argmax(test_output, 1) == test_label_tensor) / test_data_tensor.shape[0]
    # test_output = model(train_data_tensor)
    # test_loss = myloss(test_output, train_label_tensor)

    # test_acc = torch.sum(torch.argmax(test_output, 1) == train_label_tensor) / train_data_tensor.shape[0]


    print('Epoch : {} -- TestLoss : {} -- TestAcc : {}\n'.format(cur_epoch, test_loss.cpu().data, test_acc.cpu().data))

    # save the best results    
    # if best_test_res['acc'] < test_acc.cpu().data:
    #     best_test_res['acc'] = test_acc.cpu().data 
    #     best_test_res['loss'] = test_loss.cpu().data 
    #     best_test_res['predict_label'] = torch.argmax(test_output, 0).cpu().numpy()
    #     best_test_res['true_label'] = test_label_tensor.cpu().numpy()
    #     print('update res')

    print(best_test_res)


