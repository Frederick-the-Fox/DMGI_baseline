import pickle
import pprint
import numpy as np

file=open("/home/hangni/WangYC/DMGI/data/imdb_new.pkl","rb")
file_data=pickle.load(file)
pprint.pprint(file_data)
MAM = file_data['MAM']
MDM = file_data['MDM']
feature = file_data['feature']
label = file_data['label']
print('MAM: {}, type:{}. '.format(MAM, MAM.shape))
print('MDM: {}, type:{}. '.format(MDM, MDM.shape))
print('feature: {}, type:{}. '.format(feature, feature.shape))
print('label: {}, type:{}. '.format(label, label.shape))

file.close()

mam = np.load('/home/hangni/HeCo-main/data/imdb/mam.npz')
print('mam: {}, type: {}'.format(mam.files, type(mam.files)))
print('row:{}, col:{}, format:{}, shape:{}, data:{}'.format(mam['row'], mam['col'], mam['format'], mam['shape'], mam['data']))