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

path = '/home/hangni/HeCo-main/data/dblp/'

#APA
apa = np.load(path + 'apa.npz')
APA = scipy.sparse.coo_matrix((apa['data'], (apa['row'], apa['col'])), shape=(apa['shape'][0], apa['shape'][1]))
APA = (APA.A).astype(float)

#APCPA
apcpa = np.load(path + 'apcpa.npz')
APCPA = scipy.sparse.coo_matrix((apcpa['data'], (apcpa['row'], apcpa['col'])), shape=(apcpa['shape'][0], apcpa['shape'][1]))
APCPA = (APCPA.A).astype(float)

#APTPA
aptpa = np.load(path + 'aptpa.npz')
APTPA = scipy.sparse.coo_matrix((aptpa['data'], (aptpa['row'], aptpa['col'])), shape=(aptpa['shape'][0], aptpa['shape'][1]))
APTPA = (APTPA.A).astype(float)

#feature
feat = np.load(path + 'a_feat.npz')
feature = scipy.sparse.csr_matrix((feat['data'],feat['indices'], feat['indptr']), shape=(feat['shape'][0], feat['shape'][1]))
feature = (feature.A).astype(float)

#label
labels = np.load(path + 'labels.npy')
label = encode_onehot(labels)
print(label)

#idx_20
test_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/test_20.npy')
train_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/train_20.npy')
val_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/val_20.npy')

#idx_40
test_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/test_40.npy')
train_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/train_40.npy')
val_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/val_40.npy')

#idx_60
test_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/test_60.npy')
train_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/train_60.npy')
val_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/dblp/val_60.npy')

#idx_eval
test_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/dblp/eval_test_40.npy')
train_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/dblp/eval_train_40.npy')
val_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/dblp/eval_val_40.npy')

# print('MAM:{}, MDM:{}, MKM:{}, feature:{}, label:{}'.format(MAM, MDM, MKM, feature, label))

#saving
file_data = {'APA':APA, 'APCPA':APCPA, 'APTPA':APTPA, 'feature':feature, 'label':label,
            'test_idx_20':test_idx_20, 'train_idx_20':train_idx_20, 'val_idx_20':val_idx_20, 
            'test_idx_40':test_idx_40, 'train_idx_40':train_idx_40, 'val_idx_40':val_idx_40, 
            'test_idx_60':test_idx_60, 'train_idx_60':train_idx_60, 'val_idx_60':val_idx_60, 
            'test_idx_eval':test_idx_eval, 'train_idx_eval':train_idx_eval, 'val_idx_eval':val_idx_eval
            }
pkl.dump(file_data, open('dblp_new.pkl',"wb"), protocol=4)