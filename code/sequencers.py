import pandas as pd
import numpy as np


def sequencer(X, sequence_length, stepsize):
    """
    X: 1d array
    sequence_length: the length of each sequence generated
    stepsize: how many steps/samples to move forward to begin a new sequence. If stepsize is equal to sequence length there is no overlap. If stepsize is smaller than sequence_length the sequences will overlap by (sequence_length - stepsize).

    Function reshapes a 1d array into sequences and returns a 2d array (sequences on the first dimension and observations within the sequences on the second dimension).
    """

    total_sequences = int(np.ceil((len(X)-(sequence_length-1)) / stepsize))
    X_out = np.empty([total_sequences, sequence_length])
    for sequence_count in range(X_out.shape[0]):
        sequence_start = sequence_count * stepsize
        sequence_end = sequence_start + sequence_length
        X_out[sequence_count] = X[sequence_start:sequence_end]
    return X_out

def multichannel_sequencer(X, sequence_length, stepsize):
    """
    X: 2d array (first dimenion observations, second dimension features/channels)
    sequence_length: the length of each sequence generated
    stepsize: how many steps/samples to move forward to begin a new sequence. If stepsize is equal to sequence length there is no overlap. If stepsize is smaller than sequence_length the sequences will overlap by (sequence_length - stepsize).

    Function iterates over the columns and passes them to the sequencer function which returns sequenced column. This function integrates then again into a single 3d array.
    """

    column_count = X.shape[1]
    total_sequences = int(np.ceil((len(X)-(sequence_length-1)) / stepsize))
    X_out = np.empty([total_sequences, sequence_length, column_count])
    for column in range(column_count):
        X_channel_out = sequencer(X[:,column], sequence_length, stepsize)
        X_out[:,:,column] = X_channel_out
    return X_out


def sequence_dataframe(df, sequence_length, stepsize, drop_columns=[], group_column=None):
    """
    df: 2d pandas dataframe (first dimenion observations, second dimension features/channels)
    sequence_length: the length of each sequence generated
    stepsize: how many steps/samples to move forward to begin a new sequence. If stepsize is equal to sequence length there is no overlap. If stepsize is smaller than sequence_length the sequences will overlap by (sequence_length - stepsize).

    Function takes a 2d dataframe, groups it according to a group column, removes unwanted columns (including the group columns), and then passes the 2d numpy array within each dataframe-group to the multichannel_sequencer function, which returns a sequenced version of the array. All return group-arrays are then concatenated again to a single squenced 3d numpy array which is returned.
    """
    if group_column:
        drop_columns = drop_columns + [group_column]

    keep_columns = df.columns[~df.columns.isin(drop_columns)]
    print('kept columns: ', keep_columns.values)


    if group_column:
        X_out = np.empty([0,sequence_length,len(keep_columns)])
        groups = df[group_column].unique()
        for group in groups:
            X_group = multichannel_sequencer(df.loc[df[group_column]==group, keep_columns].values, sequence_length, stepsize)
            X_out = np.concatenate((X_out, X_group), axis=0)
    else:
        X_out = multichannel_sequencer(df[keep_columns].values, sequence_length, stepsize)

    print('Shape sequenced dataframe (now numpy array): ', X_out.shape)

    return X_out


def remove_sequences_containing_nan_values(X):
    """
    X: 3d numpy array (1st dimension: sequences; 2nd dimension: observations within sequences; 3rd dimension: columns/channels)

    Return 3d array with those sequences removed that contain any NAN values on either input (X) or output (y) vectors.
    """

    mask = np.zeros(X.shape[0], dtype= bool)
    for sequence in X:
        mask[0] = not np.isnan(sequence).any()
        mask = np.roll(mask, shift=-1)
    return X[mask]


def remove_sequences_without_unique_labels(X, column_containing_labels):
    """
    X: 3d numpy array (1st dimension: sequences; 2nd dimension: observations within sequences; 3rd dimension: columns/channels including single output vector with labels)
    column_containing_labels: index of column that contains the labels

    Return 3d array with those sequences removed that contain more than 1 unique label in the output (y) vector.
    """

    mask = np.zeros(X.shape[0], dtype= bool)
    for sequence in X:
        mask[0] = len(np.unique(sequence[:,column_containing_labels])) <= 1
        mask = np.roll(mask, shift=-1)
    return X[mask]
