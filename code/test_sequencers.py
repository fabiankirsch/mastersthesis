# %load_ext autoreload
# %autoreload 2
import sys
import numpy as np
import pandas as pd
from sequencers import sequencer, multichannel_sequencer, sequence_dataframe, remove_sequences_containing_nan_values, remove_sequences_without_unique_labels

def test_sequencer():
    X = np.array([1,2,3,4,5,6,7,8,9])
    X_seq = np.array(
        [[1.,2.,3.,4.],
         [3.,4.,5.,6.],
         [5.,6.,7.,8.]])
    X_out = sequencer(X, sequence_length=4, stepsize=2)
    assert((X_seq==X_out).all())

    X = np.array([1,2,3,4,5,6,7,8,9])
    X_seq = np.array([[1.,2.,3.,4.,5.,6.], [4.,5.,6.,7.,8.,9.]])
    X_out = sequencer(X, sequence_length=6, stepsize=3)
    assert((X_seq==X_out).all())

    X = np.array([1,2,3,4,5,6,7,8,9])
    X_seq = np.array(
        [[1.,2.,3.,4.,5.,6.],
         [2.,3.,4.,5.,6.,7.],
         [3.,4.,5.,6.,7.,8.],
         [4.,5.,6.,7.,8.,9.]])
    X_out = sequencer(X, sequence_length=6, stepsize=1)
    assert((X_seq==X_out).all())

def test_multichannel_sequencer_sequencer():
    X = np.array(
        [[1.,20.,0.3,4.123],
         [2.,40.,0.5,5.123],
         [3.,60.,0.6,6.123],
         [4.,70.,0.7,7.123],
         [5.,80.,0.8,8.123],
         [6.,90.,0.9,9.123],
         ])
    X_seq = np.array(
      [[[ 1.   , 20.   ,  0.3  ,  4.123],
        [ 2.   , 40.   ,  0.5  ,  5.123]],
       [[ 2.   , 40.   ,  0.5  ,  5.123],
        [ 3.   , 60.   ,  0.6  ,  6.123]],
       [[ 3.   , 60.   ,  0.6  ,  6.123],
        [ 4.   , 70.   ,  0.7  ,  7.123]],
       [[ 4.   , 70.   ,  0.7  ,  7.123],
        [ 5.   , 80.   ,  0.8  ,  8.123]],
       [[ 5.   , 80.   ,  0.8  ,  8.123],
        [ 6.   , 90.   ,  0.9  ,  9.123]]])
    X_out = multichannel_sequencer(X, sequence_length=2, stepsize=1)
    assert((X_seq==X_out).all())

    X = np.array(
        [[1.,20.,0.3,4.123],
         [2.,40.,0.5,5.123],
         [3.,60.,0.6,6.123],
         [4.,70.,0.7,7.123],
         [5.,80.,0.8,8.123],
         [6.,90.,0.9,9.123],
         ])
    X_seq = np.array(
      [[[ 1.   , 20.   ,  0.3  ,  4.123],
        [ 2.   , 40.   ,  0.5  ,  5.123],
        [ 3.   , 60.   ,  0.6  ,  6.123],
        [ 4.   , 70.   ,  0.7  ,  7.123]],
       [[ 3.   , 60.   ,  0.6  ,  6.123],
        [ 4.   , 70.   ,  0.7  ,  7.123],
        [ 5.   , 80.   ,  0.8  ,  8.123],
        [ 6.   , 90.   ,  0.9  ,  9.123]]])
    X_out = multichannel_sequencer(X, sequence_length=4, stepsize=2)
    assert((X_seq==X_out).all())

def test_sequence_dataframe():
    X = np.array(
        [[1.,20.,0.3,4.123, 'group A'],
         [2.,40.,0.5,5.123, 'group A'],
         [3.,60.,0.6,6.123, 'group A'],
         [4.,70.,0.7,7.123, 'group B'],
         [5.,80.,0.8,8.123, 'group B'],
         [6.,90.,0.9,9.123, 'group A'],
         ])
    df = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'group'])
    X_seq = np.array(
      [[[ 1.   , 20.   ,  0.3  ,  4.123],
        [ 2.   , 40.   ,  0.5  ,  5.123]],
       [[ 3.   , 60.   ,  0.6  ,  6.123],
        [ 6.   , 90.   ,  0.9  ,  9.123]],
       [[ 4.   , 70.   ,  0.7  ,  7.123],
        [ 5.   , 80.   ,  0.8  ,  8.123]]])
    X_out = sequence_dataframe(df, sequence_length=2, stepsize=2, group_column='group')
    assert((X_seq==X_out).all())

    X = np.array(
        [[1.,20.,0.3,4.123, 'group A'],
         [2.,40.,0.5,5.123, 'group A'],
         [3.,60.,0.6,6.123, 'group A'],
         [4.,70.,0.7,7.123, 'group B'],
         [5.,80.,0.8,8.123, 'group B'],
         [6.,90.,0.9,9.123, 'group A'],
         ])
    df = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'group'])
    X_seq = np.array(
      [[[ 1.   ,  4.123],
        [ 2.   ,  5.123]],
       [[ 3.   ,  6.123],
        [ 6.   ,  9.123]],
       [[ 4.   ,  7.123],
        [ 5.   ,  8.123]]])
    X_out = sequence_dataframe(df, sequence_length=2, stepsize=2, drop_columns=['B', 'C'], group_column='group')
    assert((X_seq==X_out).all())

    X = np.array(
        [[1.,20.,0.3,4.123, 'group A'],
         [2.,40.,0.5,5.123, 'group A'],
         [3.,60.,0.6,6.123, 'group A'],
         [103.,1060.,100.6,106.123, 'group A'],
         [4.,70.,0.7,7.123, 'group B'],
         [5.,80.,0.8,8.123, 'group B'],
         [6.,90.,0.9,9.123, 'group B'],
         [6.,90.,0.9,9.123, 'group B'],
         ])
    df = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'group'])
    X_seq = np.array(
      [[[1.00000e+00, 2.00000e+01, 3.00000e-01, 4.12300e+00],
        [2.00000e+00, 4.00000e+01, 5.00000e-01, 5.12300e+00],
        [3.00000e+00, 6.00000e+01, 6.00000e-01, 6.12300e+00]],

       [[2.00000e+00, 4.00000e+01, 5.00000e-01, 5.12300e+00],
        [3.00000e+00, 6.00000e+01, 6.00000e-01, 6.12300e+00],
        [1.03000e+02, 1.06000e+03, 1.00600e+02, 1.06123e+02]],

       [[4.00000e+00, 7.00000e+01, 7.00000e-01, 7.12300e+00],
        [5.00000e+00, 8.00000e+01, 8.00000e-01, 8.12300e+00],
        [6.00000e+00, 9.00000e+01, 9.00000e-01, 9.12300e+00]],

       [[5.00000e+00, 8.00000e+01, 8.00000e-01, 8.12300e+00],
        [6.00000e+00, 9.00000e+01, 9.00000e-01, 9.12300e+00],
        [6.00000e+00, 9.00000e+01, 9.00000e-01, 9.12300e+00]]])
    X_out = sequence_dataframe(df, sequence_length=3, stepsize=1, group_column='group')
    assert((X_seq==X_out).all())

    X = np.array(
        [[1.,20.,0.3,4.123, 'group A'],
         [2.,40.,0.5,5.123, 'group A'],
         [3.,60.,0.6,6.123, 'group A'],
         [103.,1060.,100.6,106.123, 'group A'],
         [4.,70.,0.7,7.123, 'group B'],
         [5.,80.,0.8,8.123, 'group B'],
         [6.,90.,0.9,9.123, 'group B'],
         [6.,90.,0.9,9.123, 'group C'],
         ])
    df = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'group'])
    X_seq = np.array(
      [[[1.00000e+00, 2.00000e+01, 3.00000e-01, 4.12300e+00],
        [2.00000e+00, 4.00000e+01, 5.00000e-01, 5.12300e+00]],

       [[3.00000e+00, 6.00000e+01, 6.00000e-01, 6.12300e+00],
        [1.03000e+02, 1.06000e+03, 1.00600e+02, 1.06123e+02]],

       [[4.00000e+00, 7.00000e+01, 7.00000e-01, 7.12300e+00],
        [5.00000e+00, 8.00000e+01, 8.00000e-01, 8.12300e+00]]])
    X_out = sequence_dataframe(df, sequence_length=2, stepsize=2, group_column='group')
    assert((X_seq==X_out).all())

def test_remove_sequences_containing_nan_values():
    X = np.array(
        [[[ 1. , 20. ,  0.3,  np.nan],
         [ 2. , 40. ,  0.5,  np.nan],
         [ 3. , 60. ,  0.6,  5. ]],

       [[ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ]]])

    X_cleaned_expected = np.array(
       [[[ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ]]])

    X_cleaned = remove_sequences_containing_nan_values(X)
    assert((X_cleaned_expected==X_cleaned).all())

    X = np.array(
       [[[ 1. , 20. ,  0.3,  np.nan],
        [ 2. , 40. ,  0.5,  5. ],
        [ 3. , 60. ,  0.6,  5. ]],

       [[ 2. , 40. ,  0.5,  5. ],
        [ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ]],

       [[ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ]],

       [[ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ],
        [ 6. , 90. ,  0.9,  np.nan]]])

    X_cleaned_expected = np.array(
       [[[ 2. , 40. ,  0.5,  5. ],
        [ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ]],

       [[ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ]]])

    X_cleaned = remove_sequences_containing_nan_values(X)
    assert((X_cleaned_expected==X_cleaned).all())


def test_remove_sequences_without_unique_labels():

    X = np.array(
       [[[ 1. , 20. ,  0.3,  5.],
        [ 2. , 40. ,  0.5,  5. ],
        [ 3. , 60. ,  0.6,  5. ]],

       [[ 2. , 40. ,  0.5,  2. ],
        [ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ]],

       [[ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ]],

       [[ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ],
        [ 6. , 90. ,  0.9,  np.nan]]])

    X_cleaned_expected = np.array(
      [[[ 1. , 20. ,  0.3,  5. ],
        [ 2. , 40. ,  0.5,  5. ],
        [ 3. , 60. ,  0.6,  5. ]],

       [[ 3. , 60. ,  0.6,  5. ],
        [ 4. , 70. ,  0.7,  5. ],
        [ 5. , 80. ,  0.8,  5. ]]])

    X_cleaned = remove_sequences_without_unique_labels(X, column_containing_labels=-1)
    assert((X_cleaned_expected==X_cleaned).all())
