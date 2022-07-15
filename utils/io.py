import os
import yaml
import stat
import pickle
import numpy as np
import pandas as pd
from os import listdir
import tensorflow as tf
from ast import literal_eval
from os.path import isfile, join
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz


def load_pandas(path, name, sep, df_name, value_name, shape=None):
    df = pd.read_csv(path + name, sep=sep)
    rows = df[df_name[0]]
    cols = df[df_name[1]]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))


def load_pandas_without_names(path, name, sep, df_name, value_name, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    rows = df[df_name[0]]
    cols = df[df_name[1]]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))


def save_numpy(matrix, path, model):
    save_npz('{0}{1}'.format(path, model), matrix)


def save_array(array, path, model):
    np.save('{0}{1}'.format(path, model), array)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def find_best_hyperparameters(folder_path, meatric, scene='r'):
    csv_files = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.endswith('tuning_'+scene+'.csv') and not f.startswith('final')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(meatric+'_Score', axis=1)

    return df


def find_single_best_hyperparameters(folder_path, meatric):
    df = pd.read_csv(folder_path)
    df[meatric + '_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
    best_settings = df.loc[df[meatric + '_Score'].idxmax()].to_dict()

    return best_settings


def save_dataframe_latex(df, path, model):
    with open('{0}{1}_parameter_tuning.tex'.format(path, model), 'w') as handle:
        handle.write(df.to_latex(index=False))


def load_csv(path, name, shape=(1010000, 2262292)):
    data = np.genfromtxt(path + name, delimiter=',')
    matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)
    save_npz(path + "rating.npz", matrix)
    return matrix


def save_pickle(path, name, data):
    with open('{0}/{1}.pickle'.format(path, name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path, name):
    with open('{0}/{1}.pickle'.format(path, name), 'rb') as handle:
        data = pickle.load(handle)

    return data


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def get_file_names(folder_path, extension='.yml'):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(extension)]


def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)


class NpyDataset(object):
    def __init__(self, data, y_dim, x_dim, mode='train'):
        self.data = np.load(data)
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.mode = mode
        self.subsample_data = None

    def __call__(self, subsample=False):
        if subsample:
            y_arr = self.subsample_data[:, :self.y_dim]
            x_arr = self.subsample_data[:, self.y_dim:self.x_dim + self.y_dim]
            t_arr = self.subsample_data[:, self.x_dim + self.y_dim:self.x_dim * 2 + self.y_dim]
        else:
            y_arr = self.data[:, :self.y_dim]
            x_arr = self.data[:, self.y_dim:self.x_dim + self.y_dim]
            t_arr = self.data[:, self.x_dim + self.y_dim:self.x_dim * 2 + self.y_dim]

        if self.mode == 'test':
            x_arr = x_arr * t_arr

        return tf.data.Dataset.from_tensor_slices({'y': y_arr, 'x': x_arr})

    def subsample(self, size):
        subsample_idx = np.random.randint(0, len(self.data) - 1, size)
        self.subsample_data = self.data[subsample_idx]


class SubNpyDataset(object):
    def __init__(self, data, y_dim, x_dim):
        self.data = data
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.subsample_data = None

    def __call__(self, subsample=False):
        if subsample:
            x_arr = self.subsample_data[:, :self.x_dim]
            y_arr = self.subsample_data[:, self.x_dim:self.x_dim + self.y_dim]
        else:
            y_arr = self.data[:, :self.x_dim]
            x_arr = self.data[:, self.x_dim:self.x_dim + self.y_dim]

        return tf.data.Dataset.from_tensor_slices({'y': y_arr, 'x': x_arr})

    def subsample(self, size):
        subsample_idx = np.random.randint(0, len(self.data) - 1, size)
        self.subsample_data = self.data[subsample_idx]




class NpyDataset_public(object):
    def __init__(self, data, y_dim, x_dim, mode='train'):
        self.data = np.load(data)
        #self.data = np.asarray(self.data, order='F')
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.mode = mode
        self.subsample_data = None

    def __call__(self, subsample=False):
        if subsample:
            y_arr = self.subsample_data[:, -1:]
            x_arr = self.subsample_data[:, :-1]
        else:
            y_arr = self.data[:, -1:]
            x_arr = self.data[:, :-1]

        y_arr_ = np.hstack((1 - y_arr, y_arr))

        return tf.data.Dataset.from_tensor_slices({'y': y_arr_, 'x': x_arr})

    def subsample(self, size):
        subsample_idx = np.random.randint(0, len(self.data) - 1, size)
        self.subsample_data = self.data[subsample_idx]
