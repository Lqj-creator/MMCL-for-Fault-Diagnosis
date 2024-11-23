import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_CWRU_finetune(dataset, num_perclass):
    train_file = os.path.join('datasets/CWRU_overlap_0.8/train/')
    test_file = os.path.join('datasets/CWRU_overlap_0.8/test/')
    train = np.load(train_file + 'train_dataset.npy')
    train_labels = np.load(train_file + 'train_label.npy')

    pre_num = train.shape[0]

    indices = np.arange(10)
    length = int(train.shape[0])
    length_per = int(length/10)
    length_new = int(length*num_perclass/10)
    data_new = [train[each*length_per:(each*length_per+length_new),:] for each in indices]
    label_new = [train_labels[each*length_per:(each*length_per+length_new)] for each in indices]
    train = np.concatenate(data_new)
    train_labels = np.concatenate(label_new)
    
    finetune_num = train.shape[0]

    test = np.load(test_file + 'test_dataset.npy')
    test_labels = np.load(test_file + 'test_label.npy')
    test_num = test.shape[0]
    return train, train_labels, test, test_labels, pre_num, finetune_num, test_num


def load_CWRU(dataset):
    train_file = os.path.join('datasets/CWRU_overlap_0.8/train/')
    test_file = os.path.join('datasets/CWRU_overlap_0.8/test/')
    train = np.load(train_file + 'train_dataset.npy')
    train_labels = np.load(train_file + 'train_label.npy')
    test = np.load(test_file + 'test_dataset.npy')
    test_labels = np.load(test_file + 'test_label.npy')
    return train, train_labels, test, test_labels

def load_Diagnosis(dataset):
    train_file = os.path.join('datasets/Diagnosis/train/')
    test_file = os.path.join('datasets/Diagnosis/test/')
    train = np.load(train_file + 'train_dataset.npy')
    train_labels = np.load(train_file + 'train_label.npy')
    test = np.load(test_file + 'test_dataset.npy')
    test_labels = np.load(test_file + 'test_label.npy')
    
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels





