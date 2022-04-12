import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

#MAM
mam = np.load('/home/hangni/HeCo-main/data/imdb/mam.npz')
row = mam['row']
col = mam['col']
data = mam['data']

MAM = scipy.sparse.coo_matrix((data, (row, col)), shape=(mam['shape'][0], mam['shape'][0]))
MAM = MAM.A
MAM = MAM.astype(float)

#MDM
mdm = np.load('/home/hangni/HeCo-main/data/imdb/mdm.npz')
row = mdm['row']
col = mdm['col']
data = mdm['data']

MDM = scipy.sparse.coo_matrix((data, (row, col)), shape=(mdm['shape'][0], mdm['shape'][0]))
MDM = MDM.A
MDM = MDM.astype(float)

#MKM
mkm = np.load('/home/hangni/HeCo-main/data/imdb/mkm.npz')
row = mkm['row']
col = mkm['col']
data = mkm['data']

MKM = scipy.sparse.coo_matrix((data, (row, col)), shape=(mkm['shape'][0], mkm['shape'][0]))
MKM = MKM.A
MKM = MKM.astype(float)

#feature
feat = np.load('/home/hangni/WangYC/imdb_process/m_feat.npz')
feature = scipy.sparse.coo_matrix((feat['data'], (feat['row'], feat['col'])), shape=(feat['shape'][0], feat['shape'][1]))
feature = feature.A

#label
labels = np.genfromtxt('/home/hangni/WangYC/imdb_process/m_label.txt')
label = encode_onehot(labels[:, 1])

#idx_20
test_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/test_20.npy')
train_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/train_20.npy')
val_idx_20 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/val_20.npy')

#idx_40
test_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/test_40.npy')
train_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/train_40.npy')
val_idx_40 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/val_40.npy')

#idx_60
test_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/test_60.npy')
train_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/train_60.npy')
val_idx_60 = np.load('/home/hangni/HeCo-main/data/my_data/imdb/val_60.npy')

#idx_eval
test_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/imdb/eval_test_40.npy')
train_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/imdb/eval_train_40.npy')
val_idx_eval = np.load('/home/hangni/HeCo-main/data/my_data/imdb/eval_val_40.npy')

print('MAM:{}, MDM:{}, MKM:{}, feature:{}, label:{}'.format(MAM, MDM, MKM, feature, label))

#saving
file_data = {'MAM':MAM, 'MDM':MDM, 'MKM':MKM, 'feature':feature, 'label':label,
            'test_idx_20':test_idx_20, 'train_idx_20':train_idx_20, 'val_idx_20':val_idx_20, 
            'test_idx_40':test_idx_40, 'train_idx_40':train_idx_40, 'val_idx_40':val_idx_40, 
            'test_idx_60':test_idx_60, 'train_idx_60':train_idx_60, 'val_idx_60':val_idx_60, 
            'test_idx_eval':test_idx_eval, 'train_idx_eval':train_idx_eval, 'val_idx_eval':val_idx_eval
            }
pkl.dump(file_data, open('imdb_new.pkl',"wb"), protocol=4)