from scipy import signal
import scipy
import numpy as np
import pandas as pd

def medfilt_3d_array(X, kernel_size=3):
    '''
    Applies scipy's median filter along the 2nd axis of a 3d numpy array. This
    axis usually contains the sequences, that can then be fed into a RNN. Median
    filter is applied separately to each dimension within a sequence.

    Parameters
    ----------
    X: a 3d array, if containing sequences these are expected on the 2nd axis
    kernel_size: kernel_size parameter of scipy.signal.medfilt
    '''

    X = np.apply_along_axis(scipy.signal.medfilt, 1, X, kernel_size)
    return X

def separate_acc_to_body_gravity(X, idx_acc_columns, sample_rate, cutoff_frequency=0.3):
    '''
    X: 3d array - batch of multivariate sequences (sequences, observations, columns/features)
    idx_acc_columns: indices of columns that contain the acceleration signal
    sample_rate: sample rate of data
    cutoff_frequency: cutoff frequency to separate gravitational components. Defaults to 0.3.

    Takes a batch of multivariate sequences of shape (sequence_batch,
    sequence_length,columns) and
    separates the acceleration signal into a body and gravity signal. The
    original acceleration signal is removed in the output. Body and gravity
    acceleration signals are added in this order to the non-accerlation signal:
    body-1, body-2, body-3, gravity-1, gravity-2, gravity-3
    '''

    # go through each sequence
    number_of_sequences = X.shape[0]
    sequence_length = X.shape[1]
    number_of_columns = X.shape[2]

    X_out = np.zeros([number_of_sequences, sequence_length, number_of_columns + len(idx_acc_columns)])
    columns_idx = np.arange(number_of_columns)
    idx_non_acc_columns = columns_idx[~np.isin(columns_idx, idx_acc_columns)]

    for sequence_id in range(number_of_sequences):

        X_out[sequence_id,:,:len(idx_non_acc_columns)] = X[sequence_id, :, idx_non_acc_columns].T
        X_out[sequence_id,:,len(idx_non_acc_columns):] = acc_to_body_grav(X[sequence_id, :, idx_acc_columns].T, sample_rate, cutoff_frequency)

    print('Shape BEFORE separating acc and body: ', X.shape)
    print('Shape AFTER separating acc and body: ', X_out.shape)
    return X_out

def acc_to_body_grav(acceleration_seq, sample_rate, cutoff_frequency=0.3):
    '''
    acceleration_seq: 2d array containing acceleration signal in the columns (likely 3 columns XYZ)
    sample_rate: sample rate of data
    cutoff_frequency: cutoff frequency to separate gravitational components. Defaults to 0.3.

    Takes a 2d array of accelerations signals and separates the signal in each column into the body and gravity component using a butterworth lowpass filter.

    Returns a 2d array with twice the number of columns as the input. Original acceleration signal is dropped. Body components are in the first columns, gravity components in the last column.
    '''

    b, a = butter_lowpass(cutoff_frequency=cutoff_frequency,sample_rate=sample_rate)
    sequence_length = acceleration_seq.shape[0]
    number_of_columns = acceleration_seq.shape[1]
    acc_body_gravity_seq = np.zeros([sequence_length, number_of_columns * 2])

    gravity_seq = np.apply_along_axis(apply_filtfilt, 0, acceleration_seq, b, a)
    body_seq = acceleration_seq - gravity_seq
    acc_body_gravity_seq[:,:number_of_columns] = body_seq
    acc_body_gravity_seq[:,number_of_columns:]= gravity_seq
    return acc_body_gravity_seq

def apply_filtfilt(arr, b, a):
    '''
    A wrapper around the signal.filtfilt function that expects the array
    as the 1st argument, and the filter coefficients as 2nd and 3rd
    arguments, so an iterable can be passed to this function.
    '''
    return signal.filtfilt(b, a, arr)

def butter_lowpass(cutoff_frequency, sample_rate, order=3):
    '''
    Generate filter coefficients for butterworth lowpass filter
    '''
    nyquist_frequency = 0.5 * sample_rate
    corner_frequency =  cutoff_frequency / nyquist_frequency
    b, a = signal.butter(order, corner_frequency, btype='lowpass')
    return b, a
