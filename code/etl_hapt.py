import urllib
import zipfile
import pandas as pd
import numpy as np
import glob
import re
import sys
import os
from os.path import basename

def get_Xy(data_dir, download_url):
    """
    Get raw input data (X) and labels (y) and merge these into a single data object. If data is not yet locally available download it first.
    """
    if not os.path.isdir(data_dir):
        print('Specified data set directory does not exist. Data set will be downloaded.')
        download_extract_data(download_url, data_dir)

    X = get_X(data_dir)
    y = get_y(data_dir)

    X['activity_id'] = np.nan

    for index, sample in y.iterrows():
        exp_mask = X['experiment_id'] == sample['experiment_id']
        start_mask = X['time'] >= sample['sample_start']
        end_mask = X['time'] <= sample['sample_end']

        X.loc[exp_mask & start_mask & end_mask, 'activity_id'] = \
            sample['activity_id']
    Xy = X
    print('Xy shape: ', Xy.shape)
    return Xy


def get_X_gyro(data_dir):
    """
    Get gyroscope data from all experiment files and concatenate them to a single data object.
    """

    gyro_datafiles = glob.glob(os.path.join(data_dir, 'RawData', 'gyro*'))
    X_gyro = pd.DataFrame()
    for datafile in gyro_datafiles:
        sensor_names = ['gyro-X', 'gyro-Y', 'gyro-Z']
        X = pd.read_csv(datafile,
                         delim_whitespace=True,
                         names=sensor_names)

        X.reset_index(inplace=True)
        X.columns = ['time'] + sensor_names
        digits_in_filename = re.findall(r'\d+',basename(datafile))
        X['experiment_id'] = int(digits_in_filename[0])
        X['participant_id'] = int(digits_in_filename[1])
        X_gyro = X_gyro.append(X, ignore_index=True)
    return X_gyro

def get_X_acc(data_dir):
    """
    Get accelerometer data from all experiment files and concatenate them to a single data object.
    """
    acc_datafiles = glob.glob(os.path.join(data_dir, 'RawData', 'acc*'))
    X_acc = pd.DataFrame()
    for datafile in acc_datafiles:
        sensor_names = ['acc-X', 'acc-Y', 'acc-Z']
        X = pd.read_csv(datafile,
                         delim_whitespace=True,
                         names=sensor_names)
        X.reset_index(inplace=True)
        X.columns = ['time'] + sensor_names
        digits_in_filename = re.findall(r'\d+',basename(datafile))
        X['experiment_id'] = int(digits_in_filename[0])
        X['participant_id'] = int(digits_in_filename[1])
        X_acc = X_acc.append(X, ignore_index=True)
    return X_acc

def get_X(data_dir):
    """
    Call functions for getting gyroscope and accelerometer data and merge those 2 data object. Sort them by experiment_id and time.
    """
    X_gyro = get_X_gyro(data_dir)
    X_acc = get_X_acc(data_dir)
    X = pd.merge(X_gyro, X_acc, on=['time', 'experiment_id', 'participant_id'])
    X.sort_values(['experiment_id', 'time'], inplace=True)

    return X

def get_y(data_dir):
    """
    Get labels referencing time windows of raw data.
    """
    data_path = os.path.join(data_dir, 'RawData/labels.txt')
    y = pd.read_csv(data_path,
                            delim_whitespace=True,
                            header=None,
                            names=['experiment_id',
                                   'participant_id',
                                   'activity_id',
                                   'sample_start',
                                   'sample_end'])

    return y


def get_y_translations(data_dir, activity_id_subset=None):
    '''
    data_dir: directory path where extracted HAPT data set is located
    activity_id_subset: list of acitivity ids. Return y translations only for this subset (if provided)
    '''

    y_translations = pd.read_csv(os.path.join(data_dir, 'activity_labels.txt'),
                         delim_whitespace=True,
                         header=None,
                         names=['activity_id', 'activity_name'])
    if activity_id_subset is None:
        return y_translations['activity_name'].values
    else:
        return y_translations.loc[
                    y_translations['activity_id'].isin(activity_id_subset), 'activity_name'
                ].values

def get_Xy_data_sets(data_dir, download_url, ids_train=None,
                     ids_test=None, ids_validation=None):
    """
    data_dir: directory path where extracted HAPT data set is located
    download_url: url of file to download (if needed)
    ids_train: ids of participants that should be in the train data set
    ids_test: ids of participants that should be in the test data set
    ids_validation: ids of participants that should be in the validation data set

    Get train and test and validation data set from HAPT data set. Data split based on participant ids. Only those data sets for which ids are specified are returned. Datasets are returned as pandas dataframes
    """
    data_sets = tuple()

    Xy = get_Xy(data_dir, download_url)

    if ids_train:
        Xy_train = Xy[Xy['participant_id'].isin(ids_train)]
        data_sets = data_sets + (Xy_train,)
    if ids_test:
        Xy_test = Xy[Xy['participant_id'].isin(ids_test)]
        data_sets = data_sets + (Xy_test,)
    if ids_validation:
        Xy_validation = Xy[Xy['participant_id'].isin(ids_validation)]
        data_sets = data_sets + (Xy_validation,)


    return data_sets

def get_Xy_validation(data_dir, download_url, participant_ids_data_split):
    """
    Get validation data set from HAPT data set. Data split based on participant ids.
    """

    Xy = get_Xy(data_dir, download_url)

    Xy_validation = Xy[Xy['participant_id'].isin(
                participant_ids_data_split['validation_participant_ids'])]

    print('Xy_validation shape: ', Xy_validation.shape)

    return Xy_validation


def download_file(download_url):
    """
    download_url: url of file to download
    data_set_dir: directory where zip will be saved (will be created on the fly)
    """
    urllib.request.urlretrieve(download_url, 'data.zip')
    print('Downloaded file from %s' % download_url)

def extract_file(data_set_dir):
    """
    data_set_dir: directory where zip will be saved (will be created on the fly)
    """
    zip = zipfile.ZipFile('data.zip', 'r')
    zip.extractall(data_set_dir)
    zip.close()
    print('Extracted downloaded zip to %s' % data_set_dir)
    os.remove('data.zip')
    print('Removed zip')

def download_extract_data(download_url, data_set_dir):
    """
    download_url: url of file to download
    data_set_dir: directory where data set in zip file will be extracted to. directory will be created on the fly.
    """
    download_file(download_url)
    extract_file(data_set_dir)


def split_X_y(Xy, index_column_containing_labels):
    """
    Xy: 3d numpy array (squences, observations, columns/features)
    index_column_containing_labels: index of y/label column. Set to -1 if labels are in last column

    Returns X (3d array) and y (2d array). For y it is assumed that values are the same, so only first value within each sequence is kept. Thereby eliminating the second dimension.
    """
    X = Xy[:,:,:index_column_containing_labels]
    y = Xy[:,0,index_column_containing_labels]
    return X, y


def select_labels(X, y, activities):
    """
    X: 3d array (sequences, observations, columns)
    y: 2d array (sequence label, columns)
    activites: list of activity ids to keep in the data (all others will be removed)

    Returns X, y containing only sequences with selected activities.
    """
    activities = np.array(activities)
    activity_mask = np.isin(y, activities)
    X = X[activity_mask]
    y = y[activity_mask]
    return X, y


def read_raw_data(experiment_nr, type):
    '''
    * experiment number in format '01' to '61'
    * type is either 'acc' or 'gyro'
    '''
    datafile = glob.glob('data/HAPT Data Set/RawData/%s_exp%s*' % (type, experiment_nr))[0]
    df = pd.read_csv(datafile, sep=' ', names=list('XYZ'))
    return df

    
def reverse_label_binarize(y_binary, labels):
    # get indices of columns that have the highest value
    y_idx = y_binary.argmax(axis=1)
    # get original label for each column using these indices
    y_pred = np.array(list(map(lambda x: labels[x], y_idx)))
    return y_pred
