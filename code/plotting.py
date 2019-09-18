import matplotlib
from cycler import cycler
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import sys
sys.path.append('code')
from etl_hapt import get_y, read_raw_data

def plot_raw_hapt_data_with_labels_and_sequences(df, experiment_id, start, end, etl_config, path_savefig=None):
  """
df: dataframe containing raw data (non-sequnenced) from hapt data
experiment_id: data from which experiment to use
start: sample to start plotting
end: sample to end plotting
elt_config: elt config used in the pipeline
    Function creates a plot of the raw data with different background colors for actitvities and transitions. Sequences are displayed in multiple rows so they don't visually overlap, although they share data. Sequences that are being dropped are displayed inwhite.
  """

  cmap = matplotlib.cm.plasma
  line_color = matplotlib.colors.rgb2hex(cmap(1)[:3])
  activity_color = cmap(0.8)
  transition_color = cmap(0.99)
  stepsize = etl_config['sequence_stepsize'] / etl_config['sample_rate']
  seq_length = etl_config['sequence_length'] / etl_config['sample_rate']
  start = start / etl_config['sample_rate']
  end = end / etl_config['sample_rate']

  subset = df[df['experiment_id']==experiment_id].copy()
  subset.index = subset['time'] / etl_config['sample_rate']
  subset= subset.loc[start:end]

  labels = get_y(etl_config['data_set_dir'])
  labels_subset = labels[labels['experiment_id']==experiment_id][['activity_id', 'sample_start', 'sample_end']]
  labels_subset['sample_start'] = labels_subset['sample_start'] / etl_config['sample_rate']
  labels_subset['sample_end'] = labels_subset['sample_end'] / etl_config['sample_rate']


  labels_subset = labels_subset[(labels_subset['sample_start'] >= start) & (labels_subset['sample_end'] <= end )]
  fig, ax = plt.subplots()

  ax.plot(subset[etl_config['channel_names_prior_preprocess']], color=line_color, alpha=0.5)
  for index, row in labels_subset.iterrows():
    if row['activity_id'] < 7:
      # color for activities
      facecolor = activity_color
    else:
      facecolor = transition_color

    ax.axvspan(row['sample_start'], row['sample_end'], facecolor=facecolor, alpha=1, label=row['activity_id'])

  seq_plot_row = 0
  for sequence_start in np.arange(subset.index.min(), subset.index.max(), stepsize):

    # check if it sequence is contained within activity
    if len(labels_subset[(labels_subset['sample_start'] <= sequence_start) & (labels_subset['sample_end'] >= (sequence_start + seq_length))]) == 1:
      # entire sequence is contained within frequence
      facecolor = line_color
    else:
      # sequence either contains multiple activities or an unlabeled part
      facecolor = 'white'

    edgecolor=  line_color
    ax.axvspan(sequence_start,sequence_start+seq_length, 0.92 - 0.05 * seq_plot_row, 0.95 - 0.05 * seq_plot_row, facecolor=facecolor, edgecolor=edgecolor)
    seq_plot_row += 1
    if seq_plot_row == 6:
      seq_plot_row = 0

  custom_lines = [Line2D([0], [0], color=line_color, lw=4)]
  handles = [mpatches.Patch(facecolor=line_color, edgecolor=edgecolor),
             mpatches.Patch(facecolor='white', edgecolor=edgecolor),
             mpatches.Patch(facecolor=activity_color),
             mpatches.Patch(facecolor=transition_color),
              Line2D([0], [0], color=line_color, lw=2, alpha=0.5)
              ]
  labels = ['Kept sequence', 'Dropped sequence','Activity', 'Transition',  'Raw sensory data']
  ax.legend(handles, labels, bbox_to_anchor=(1.25,0.8))
  ax.set_xlabel('Time in seconds')
  ax.set_yticks([])
  ylim = ax.get_ylim()
  ax.set_ylim(ylim[0], ylim[1]* 1.5)
  if path_savefig:
    plt.savefig(path_savefig, bbox_inches='tight')
  else:
    plt.show()

def plot_acc_loss_model_history(history, path_savefig=None):
    """
    history: keras model history
    path_savefig: (relative) path where plot should be saved. If not provided plot will be shown.

    Plots model history will accuracy and loss for train and test data.
    """

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    train_color = default_colors[0]
    test_color = default_colors[1]
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(history['acc'], color=train_color)
    ax2.plot(history['val_acc'], color=train_color, ls='--')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs trained')
    ax2.set_ylim(0,1)
    ax2.legend(['Train', 'Test'], bbox_to_anchor=(-0.2,1), title='Accuracy')
    # ax2.legend(['Train', 'Test'], loc='upper left', title='Accuracy')
    ax.plot(history['loss'], color=test_color)
    ax.plot(history['val_loss'], color=test_color, ls='--')
    ax.set_ylabel('Log loss')
    ax.set_yscale('log')
    ax.legend(['Train', 'Test'], bbox_to_anchor=(-0.2, 0.5), title='Loss')
    # ax.legend(['Train', 'Test'], loc='lower left', title='Loss')



def plot_all_columns(seq, column_names=None, path_savefig=None, sampling_frequency=None):
    """
    seq: multivariate array
    columns_names: list of column names that are used for in the plot legend
    path_savefig: (relative) path where plot should be saved. If not provided plot will be shown.
    """
    set_new_plt_color_cycle('plasma', 3, min=0, max=0.8)

    if not column_names:
        column_names = np.arange(seq.shape[1])

    if sampling_frequency:
        x = np.arange(0, len(seq))/sampling_frequency
        xlabel = 'Time in seconds'
    else:
        x = np.arange(0,len(seq))
        xlabel = 'Samples in sequence'

    seq = seq.T
    linestyle = ['-','-', '-', '--', '--', '--', ':',':',':']
    for idx, column_name in enumerate(column_names):
        plt.plot(x,seq[idx], ls=linestyle[idx])

    plt.legend(column_names, loc='upper left')
    plt.xlabel(xlabel)

    if path_savefig:
        plt.savefig(path_savefig, bbox_inches='tight', pad_inches=0)
    else:
        pass
        # plt.show()

def plot_multi_class_roc(y, y_pred, column_labels, path_savefig=None):
    """
    y: multivariate output vector with binary/one-hot-encoding
    y_pred: same as y, but predicted by a model
    column_labels: list of labels used for legend in plot
    path_savefig: (relative) path where plot should be saved. If not provided plot will be shown.

    Creates a one-vs-all roc curve for each category.
    """

    all_fpr = {}
    all_tpr = {}
    all_auc = {}
    for columns in range(y_pred.shape[1]):
        all_fpr[columns], all_tpr[columns], _ = roc_curve(y[:, columns], y_pred[:, columns])
        all_auc[columns] = auc(all_fpr[columns], all_tpr[columns])

    for columns in range(len(all_fpr)):
        plt.plot(all_fpr[columns], all_tpr[columns], label='(AUC=%.3f) %s' % (all_auc[columns], column_labels[columns]))
    plt.plot([0,1], [0,1], color='black')

    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall/Sensitivity)')
    # if path_savefig:
    #     plt.savefig(path_savefig, bbox_inches='tight')
    # else:
    #     plt.show()

def set_new_plt_color_cycle(cmap_name, number_of_colors=10, min=0, max=0.8):
    plt.rcParams['axes.prop_cycle']= matplotlib.cycler(color=get_hex_color_palette_from_cmap(cmap_name, number_of_colors, min, max))


def set_mixed_color_linestyle_cycler(cmap_name, number_of_colors=3,
                                     cmap_min=0, cmap_max=0.8,
                                     linestyles=None):
    """
    cmap: name of cmap. See "https://matplotlib.org/users/colormaps.html"
    number_of_colors: how many colors should be used in the cycler
    cmap_min: min scalar value for color drawn from cmap (range [0,1])
    cmap_max: max scalar value for color drawn from cmap (range [0,1])
    linestyles: list of linestyles that should be used by the cycler.
                Default:['-', '--', ':',  '-.']

    The default cycler is set to contain a mix of the colors drawn from the colormap and the linestyles provided.
    """

    color_palette = get_hex_color_palette_from_cmap(cmap_name,
                                                    number_of_colors,
                                                    cmap_min, cmap_max)
    if not linestyles:
        linestyles = ['-', '--', ':',  '-.']

    default_cycler = (cycler(color=color_palette) *
                      cycler(linestyle=linestyles))
    plt.rcParams['axes.prop_cycle'] = default_cycler

def get_hex_color_palette_from_cmap(cmap, number_of_colors=10, min=0, max=1):
    """
    Returns a list of hex colours extraced from the specified colormap (cmap)
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    color_palette = np.linspace(min,max, number_of_colors)
    color_palette_rgba = cmap(color_palette)
    return list(map(convert_rgba_to_hex, color_palette_rgba))


def convert_rgba_to_hex(rgba):
    return matplotlib.colors.rgb2hex(rgba[:3])


def scale_to_seconds(x, sample_start, sampling_frequency):
    return (x - sample_start) / sampling_frequency

def plot_raw_experiment_data(experiment_nr, sample_start=None, sample_end=None, sampling_frequency=50):
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    sample_end_seconds = scale_to_seconds(sample_end, sample_start,
                                            sampling_frequency)
    sample_start_seconds = scale_to_seconds(sample_start, sample_start,
                                            sampling_frequency)
    df = read_raw_data(experiment_nr, 'acc')
    df.index = scale_to_seconds(df.index, sample_start, sampling_frequency)
    df[sample_start_seconds:sample_end_seconds].plot(ax=ax1, alpha=0.6, title='Accelerometer')
    xticks_locations = ax1.get_xticks()
    ax1.set_ylabel('Linear acceleration (g-units)')
    ax1.set_xlabel('Time (seconds)')
    df = read_raw_data(experiment_nr, 'gyro')
    df.index = scale_to_seconds(df.index, sample_start, sampling_frequency)
    df[sample_start_seconds:sample_end_seconds].plot(ax=ax2, alpha=0.6, title='Gyroscope')
    ax2.set_ylabel('Angular velocity (rad/second)')
    ax2.set_xlabel('Time in seconds')

def plot_confusion_matrix(y, y_pred, classes, title, cmap):
    """
    y: array of true classifications
    y_pred: array of predicted classifications
    classes: array of class labels

    Plots confusion matrix.

    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py (Viewed 2019-05-03)
    """

    cm = confusion_matrix(y, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, vmin=0,vmax=1, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, label='Proportion classified',
                       boundaries=np.linspace(0,1,1000, endpoint=True),
                       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_aspect('auto')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
        rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=20, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'#'.2f' if normalize else 'd'
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black")
    fig.tight_layout()

def plot_acc_loss_model_history_for_variant(history, variant_codename):
    plt.rcParams['figure.figsize'] = [12, 3]
    plt.rcParams.update({'font.size': 17})
    set_new_plt_color_cycle('plasma', 2,)
    plot_acc_loss_model_history(history)
    plt.savefig('figures/%s_learning_loss_accuracy.png' % variant_codename, bbox_inches='tight', pad_inches=0)

def plot_confusion_matrix_for_variant(model, X_test, y_test, etl_config, variant_codename):

    from etl_hapt import get_y_translations, reverse_label_binarize
    y_test_pred_binary = model.predict(X_test)
    labels = get_y_translations(etl_config['data_set_dir'], etl_config['selected_labels'])
    labels = list(map(lambda label: label.lower(), labels))
    y_test_pred = reverse_label_binarize(y_test_pred_binary, etl_config['selected_labels'])
    plt.rcParams.update({'font.size': 17})
    plt.rcParams['figure.figsize'] = [17, 5]
    plot_confusion_matrix(y_test, y_test_pred, classes=labels, title='', cmap='plasma')
    plt.savefig('figures/%s_confusion_matrix.png' % variant_codename, bbox_inches='tight', pad_inches=0)
