"""
data processing
"""
import os
import numpy as np
import pandas as pd
import tqdm

from utils.math_utils import z_score

class Dataset(object):
    """
    Dataset
    """

    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):  # type: train, val or test
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    """Generate data in the form of standard sequence unit."""
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(
                data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def data_gen_mydata(input_file, input_prev, n, n_his, n_pred, interval, dataset, task, n_config):
    """data processing
    """
    #param
    filename = dataset + '_' + str(n_pred) + '_' + task + '.csv'
    
    n_val, n_test = n_config
    nums_per_hour = int(60/interval)
    if os.path.isfile(filename):
        print("read file "+filename)
        df = pd.read_csv(filename)
       #df = df.drop(columns=['date'])
   
    # data
    else:
        x = pd.read_csv(input_file)
        x_prev = pd.read_csv(input_prev)
        x = x.drop(columns=['date'])
        x_prev = x_prev.drop(columns=['date'])
        
        df = pd.DataFrame(columns=x.columns)
        #the time step we want to predict = i + n_his 
        for i in tqdm.tqdm(range(0, len(x) - n_pred - n_his + 1)):
            #df = [func(j) for j in xrange(i + n_his - 24*n_his*nums_per_hour, i + n_his - 24*nums_per_hour + 1, 24*nums_per_hour)]
            for k in range(n_pred):
                for j in range(i+n_his-24*n_his*nums_per_hour, i+n_his-24*nums_per_hour+1, 24*nums_per_hour):
                    if j + k < 0:
                       arr = x_prev[j-1+k:j+k].to_numpy()
                       print((',').join([str(item) for item in arr[0]]))
                       #df = df.append(x_prev[j-1+k:j+k])
                    else:
                       arr = x[j+k:j+k+1].to_numpy()
                       print((',').join([str(item) for item in arr[0]]))
                       # df = df.append(x[j+k:j+k+1])
            arr = x[i:i+n_his+n_pred].to_numpy()
            for array in arr:
                print((',').join([str(item) for item in array]))
           #df = df.append(x[i:i + n_his + n_pred])
        """
        data = np.array()
        for i in tqdm.tqdm(range(0, len(x) - n_pred - n_his +1)):
            
            for k in range(n_pred):
                for j in range(i+n_his-24*n_his*nums_per_hour, i+n_his-24*nums_per_hour+1, 24*nums_per_hour):
                    if j + k < 0:
                        data = np.concatenate((data, x_prev[j-1+k:j+k]), axis=1)
                    else:
                        data = np.concatenate((data, x[j+k:j+k+1]), axis=1)
                    ]
        """
        exit(0)
    print(df.iloc[:,:7].head(70))
    data = df.values.reshape(-1, (n_pred+1)*n_his + n_pred,  n, 1)
    #total num of data n * (n_his + n_pred)
    n_train = data.shape[0] - n_val - n_test
    x_stats = {'mean': np.mean(data), 'std': np.std(data)}
        
    x_train = data[:n_train]
    x_val = data[n_train:n_train + n_val]
    x_test = data[n_train + n_val:n_train + n_val + n_test]
    
    if task == 'train':
        x_data = {'train': x_train, 'val': x_val, 'test': x_test}
        dataset = Dataset(x_data, x_stats)
        return dataset
    elif task == 'infer':
        print("generating infer dataset")
        x_infer = {'test':data}
        dataset = Dataset(x_infer, x_stats)
        return dataset
def data_gen(file_path, data_config, n_route, n_frame=21, day_slot=288):
    """Source file load and dataset generation."""
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route,
                       day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    """Data iterator in batch.
    Args:
        inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
        batch_size: int, size of batch.
        dynamic_batch: bool, whether changes the batch size in the last batch 
            if its length is less than the default.
        shuffle: bool, whether shuffle the batches.
    """
    len_inputs = len(inputs)
    
    if shuffle:
        np.random.seed(2021)
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    
    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        yield inputs[slide]
