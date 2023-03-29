import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import pickle
import pandas as pd
simplefilter("ignore", category=ConvergenceWarning)

def concat_process(eeg_data, eye_data):
    train_data_eye = np.asarray(eye_data['cell']['train_data_eye'].tolist()).squeeze()
    test_data_eye = np.asarray( eye_data['cell']['test_data_eye'].tolist()).squeeze()
    
    train_data_eeg = pickle.loads( eeg_data['train_data'] )
    test_data_eeg = pickle.loads( eeg_data['test_data'] )
    train_label = eeg_data['train_label']
    test_label = eeg_data['test_label']

    train_data_all_bands = []
    test_data_all_bands = []

    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        train_tmp = train_data_eeg[bands]
        test_tmp = test_data_eeg[bands]
        if bands == 'delta':
            train_data_all_bands = train_tmp
            test_data_all_bands = test_tmp
        else:
            train_data_all_bands = np.hstack((train_data_all_bands, train_tmp))
            test_data_all_bands = np.hstack((test_data_all_bands, test_tmp))

    print('train_eeg_all_bands : {}'.format(train_data_all_bands.shape))
    print('test_data_all_bands : {}'.format(test_data_all_bands.shape))
    print('train_eye_data : {}'.format(train_data_eye.shape))
    print('test_eye_data : {}'.format(test_data_eye.shape))
    print('train_label : {}'.format(train_label.shape))
    print('test_label : {}'.format(test_label.shape))

    all_train = np.hstack((train_data_all_bands, train_data_eye))
    all_test = np.hstack((test_data_all_bands, test_data_eye))

    return all_train, all_test, train_label, test_label



def generating_data(data_dict, clip_label, feature_name):
    # first 9 movies as training, the last 6 movies as testing
    train_data = data_dict[feature_name+'1']
    print("\t### TRAIN ###")
    print(f"\t\t{feature_name + '1'} - video 1 - {train_data.shape} observacoes")
    _, num, _ = train_data.shape 
    train_label = np.zeros(num,) + clip_label[0]
    train_data = np.swapaxes(train_data, 0, 1)
    train_data = np.reshape(train_data, (num, -1))
    train_residual_index = [2,3,4,5,6,7,8,9]
    for ind,i in enumerate(train_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape 
        used_label = np.zeros(num,) + clip_label[ind+1]
        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        print(f"\t\t{feature_name + str(i)} - video {str(i)} - {used_data.shape} observacoes")
        train_data = np.vstack((train_data, used_data))
        train_label = np.hstack((train_label, used_label))
    
    test_data = data_dict[feature_name+'10']
    print("\t### TEST ###")
    print(f"\t\t{feature_name + '10'} - video 10 - {test_data.shape} observacoes")
    _, num, _ = test_data.shape 
    test_label = np.zeros(num,) + clip_label[9]
    test_data = np.swapaxes(test_data, 0, 1)
    test_data = np.reshape(test_data, (num, -1))
    test_residual_index = [11,12,13,14,15]
    for ind,i in enumerate(test_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape 
        used_label = np.zeros(num,) + clip_label[ind+10]
        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        print(f"\t\t{feature_name + str(i)} - video {str(i)} - {used_data.shape} observacoes")
        test_data = np.vstack((test_data, used_data))
        test_label = np.hstack((test_label, used_label))
    return train_data, test_data, train_label, test_label

def generating_data_as_dataframe(data_dict, clip_label, feature_name):
    # first 9 movies as training, the last 6 movies as testing
    train_data = data_dict[feature_name+'1']
    print("\t### TRAIN ###")
    print(f"\t\t{feature_name + '1'} - video 1 - {train_data.shape} observacoes")
    _, num, _ = train_data.shape 
    train_label = np.zeros(num,) + clip_label[0]

    train_film_ref = np.zeros(num,) + 1

    train_data = np.swapaxes(train_data, 0, 1)
    train_data = np.reshape(train_data, (num, -1))
    train_residual_index = [2,3,4,5,6,7,8,9]
    for ind,i in enumerate(train_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape 
        used_label = np.zeros(num,) + clip_label[ind+1]

        used_film_ref = np.zeros(num,) + i

        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        print(f"\t\t{feature_name + str(i)} - video {str(i)} - {used_data.shape} observacoes")
        train_data = np.vstack((train_data, used_data))
        train_label = np.hstack((train_label, used_label))

        train_film_ref = np.hstack((train_film_ref, used_film_ref))
     
    test_data = data_dict[feature_name+'10']
    print("\t### TEST ###")
    print(f"\t\t{feature_name + '10'} - video 10 - {test_data.shape} observacoes")
    _, num, _ = test_data.shape 
    test_label = np.zeros(num,) + clip_label[9]

    test_film_ref = np.zeros(num,) + 10

    test_data = np.swapaxes(test_data, 0, 1)
    test_data = np.reshape(test_data, (num, -1))
    test_residual_index = [11,12,13,14,15]
    for ind,i in enumerate(test_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape 
        used_label = np.zeros(num,) + clip_label[ind+10]

        used_film_ref = np.zeros(num,) + i

        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        print(f"\t\t{feature_name + str(i)} - video {str(i)} - {used_data.shape} observacoes")
        test_data = np.vstack((test_data, used_data))
        test_label = np.hstack((test_label, used_label))

        test_film_ref = np.hstack((test_film_ref, used_film_ref))

    all_data = np.vstack((train_data,test_data))
    all_data_df = pd.DataFrame(all_data)

    all_films_ref = np.hstack((train_film_ref, test_film_ref))
    all_films_ref_df = pd.DataFrame(all_films_ref)
    return all_data_df, all_films_ref_df

def logistic_classification(train_data, test_data, train_label, test_label):
    best_res = {}
    best_res['n'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label 
    best_res['proba'] = 0
    clf = LogisticRegression()
    clf.fit(train_data, train_label)
    p_labels = clf.predict(test_data)
    score = clf.score(test_data, test_label)
    proba = clf.predict_proba(test_data)
    best_res['acc'] = score
    best_res['p_label'] = p_labels
    best_res['proba'] = proba 
    return best_res 

def knn_classification(train_data, test_data, train_label, test_label):
    best_res = {}
    best_res['n'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label 
    best_res['proba'] = 0
    for n in range(3, 10):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        proba = clf.predict_proba(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score
            best_res['n'] = n
            best_res['p_label'] = p_labels
            best_res['proba'] = proba 
    return best_res  

def svm_classification(train_data, test_data, train_label, test_label):
    best_res = {}
    best_res['c'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label 
    for c in range(-10, 10):
        print(f"\t\t evaluating LinearSVC kernel with c={c}")
        clf = svm.LinearSVC(C=2**c)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        decision_value = clf.decision_function(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score
            best_res['c'] = 2**c
            best_res['p_label'] = p_labels
            best_res['decision_val'] = decision_value 
    for c in np.arange(0.1, 20, 0.5):
        print(f"\t\t evaluating LinearSVC kernel with c={c}")
        clf = svm.LinearSVC(C=c)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        decision_value = clf.decision_function(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score 
            best_res['c'] = c 
            best_res['p_label'] = p_labels 
            best_res['decision_val'] = decision_value
    return best_res  
