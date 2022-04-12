import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import sys

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

path = '/home/hangni/HeCo-main/data/freebase/'

#MAM
mam = np.load(path + 'mam.npz')
MAM = scipy.sparse.coo_matrix((mam['data'], (mam['row'], mam['col'])), shape=(mam['shape'][0], mam['shape'][1]))
MAM = (MAM.A).astype(float)
#MDM
mdm = np.load(path + 'mdm.npz')
MDM = scipy.sparse.coo_matrix((mdm['data'], (mdm['row'], mdm['col'])), shape=(mdm['shape'][0], mdm['shape'][1]))
MDM = (MDM.A).astype(float)

#MWM
mwm = np.load(path + 'mwm.npz')
MWM = scipy.sparse.coo_matrix((mwm['data'], (mwm['row'], mwm['col'])), shape=(mwm['shape'][0], mwm['shape'][1]))
MWM = (MWM.A).astype(float)

#label
labels = np.load(path + 'labels.npy')
label = encode_onehot(labels)

#feature
feature = scipy.sparse.eye(3492)
feature = feature.A

#idx_20
test_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/test_20.npy')
train_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/train_20.npy')
val_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/val_20.npy')

#idx_40
test_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/test_40.npy')
train_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/train_40.npy')
val_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/val_40.npy')

#idx_60
test_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/test_60.npy')
train_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/train_60.npy')
val_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/freebase/val_60.npy')

#idx_eval
test_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/freebase/eval_test_40.npy')
train_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/freebase/eval_train_40.npy')
val_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/freebase/eval_val_40.npy')

# # print('MAM:{}, MDM:{}, MKM:{}, feature:{}, label:{}'.format(MAM, MDM, MKM, feature, label))

# #saving
file_data = {'MAM':MAM, 'MDM':MDM, 'MWM':MWM, 'feature':feature, 'label':label,
            'test_idx_20':test_idx_20, 'train_idx_20':train_idx_20, 'val_idx_20':val_idx_20, 
            'test_idx_40':test_idx_40, 'train_idx_40':train_idx_40, 'val_idx_40':val_idx_40, 
            'test_idx_60':test_idx_60, 'train_idx_60':train_idx_60, 'val_idx_60':val_idx_60, 
            'test_idx_eval':test_idx_eval, 'train_idx_eval':train_idx_eval, 'val_idx_eval':val_idx_eval
            }

pkl.dump(file_data, open('freebase.pkl',"wb"), protocol=4)