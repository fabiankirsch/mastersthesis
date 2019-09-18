
import sys
sys.path.append('code')
from etl_hapt import get_Xy_data_sets, split_X_y, select_labels
from sequencers import sequence_dataframe, remove_sequences_containing_nan_values, remove_sequences_without_unique_labels
from sklearn.preprocessing import label_binarize
from filters import separate_acc_to_body_gravity, medfilt_3d_array


def run_default_etl_pipeline(Xy, etl_config):
    """
    Xy: dataframe containing X and y (input and output) features
    etl_config: dictionary containing various parameters for the different
    layers of the the ETL (pipeline)

    The function wraps the default layers of the ETL pipeline. It returns separate X, y and binarized y numpy arrays.
    """

    Xy = sequence_dataframe(Xy,
                            etl_config['sequence_length'],
                            etl_config['sequence_stepsize'],
                            group_column=etl_config['group_column'],
                            drop_columns=etl_config['drop_columns'])

    Xy = remove_sequences_containing_nan_values(Xy)

    index_last_column = -1
    Xy = remove_sequences_without_unique_labels(
                          Xy, column_containing_labels=index_last_column)

    X, y = split_X_y(Xy, index_last_column)

    # keep only sequences that match the selected labels
    X, y = select_labels(X, y, etl_config['selected_labels'])

    y_binary = label_binarize(y, etl_config['selected_labels'])

    return X, y, y_binary


def run_default_preprocessing_pipeline(X, preprocessing_config):

    X = separate_acc_to_body_gravity(X, preprocessing_config['acc_columns_idx'], preprocessing_config['sample_rate'])

    X = medfilt_3d_array(X, preprocessing_config['median_filter_kernel'])

    return X
