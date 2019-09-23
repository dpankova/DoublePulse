import pickle
import numpy as np
from sklearn.utils import shuffle

def ReadInData(DataDir):

    # _d means DoublePulse, _s means SinglePulse
    info_d = []
    info_s = []
    data_d = []
    data_s = []
    #
    # On Luzern, DataDir = /home/dfc13/DoublePulse/data/
    #

    for i in range(0,10):
        name_dd = DataDir+"Tau05to15PeV_00{0}_data.npy".format(i)
        name_di = DataDir+"Tau05to15PeV_00{0}_info.pkl".format(i)
        name_sd = DataDir+"Electron05to15PeV_00{0}_data.npy".format(i)
        name_si = DataDir+"Electron05to15PeV_00{0}_info.pkl".format(i)
        info_d_temp = pickle.load(open(name_di, "rb"))
        info_s_temp = pickle.load(open(name_si, "rb"))
        data_d_temp = np.load(name_dd ,allow_pickle=True,encoding='bytes')
        data_s_temp = np.load(name_sd ,allow_pickle=True,encoding='bytes')
        info_d = info_d + info_d_temp
        info_s = info_s + info_s_temp
        data_d.append(data_d_temp)    
        data_s.append(data_s_temp)
    data_d = np.vstack(data_d)
    data_s = np.vstack(data_s)
    info_d = np.array(info_d)
    info_s = np.array(info_s)
    
    return data_d, data_s, info_d, info_s


def SeparateShuffleData(data_d,label_d,data_s,label_s):

    data = np.concatenate((data_d, data_s), axis = 0) 
    label = np.concatenate((label_d, label_s), axis = 0) 
    data, label = shuffle(data, label, random_state =12)

    train_data = data[:24000]
    train_label = label[:24000]
    train_data = train_data.reshape((len(train_data),300,60,1))
    train_data = train_data.astype('float32')/10**-8
    mean = np.mean(train_data)
    std = np.std(train_data)
    #print(mean,std)
    train_data = train_data - mean
    train_data = train_data/std

    valid_data = data[24000:28000]
    valid_label = label[24000:28000]
    valid_data = valid_data.reshape((len(valid_data),300,60,1))
    valid_data = valid_data.astype('float32')/10**-8
    valid_data = valid_data - mean
    valid_data = valid_data/std

    test_data = data[28000:]
    test_label = label[28000:]
    test_data = test_data.reshape((len(test_data),300,60,1))
    test_data = test_data.astype('float32')/10**-8
    test_data = test_data - mean
    test_data = test_data/std

    return train_data, train_label, valid_data, valid_label, test_data, test_label
