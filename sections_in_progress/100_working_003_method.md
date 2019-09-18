::: {.cell .markdown}


\cleardoublepage

# Methods and tools

## Pipeline requirements

The aim of this thesis is to develop a machine learning pipeline for human activity and state recognition that can be applied with only minor changes to different uses cases and data sets. The pipeline should accept raw sensory data as an input and provide the correct labels for the mental or physical state or activity the human is currently performing. The training of the model can happen offline, but online inference, i.e. classifying activities and states should be possible in real-time and happen within less than a second.

## Data
### Data set
There are at least two major obstacles to developing a well working machine learning pipeline, which are often confounded. One obstacle is to find a well working implementation for the actual pipeline. The other obstacle is that the data used for developing and testing the pipeline needs to contain the patterns that one actually aims to find. If the data is too noisy or the patterns that one expects actually don’t exist, there is no way for the algorithm to learn the patterns and classify the behavior correctly. The focus of this thesis lies on developing a well working pipeline implementation. Therefore, a public machine learning data set was chosen, which has been successfully used for machine learning. This way the second obstacle of bad data is avoided. The chosen data set was published by @reyes-ortiz_transition-aware_2016. It can be retrieved from the UCI machine learning repository ^[Data set at UCI machine learning repository: http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions (2019-06-06)]. The data set contains human activity recognition data from 30 participants. The participants performed different physical activities like standing, walking or sitting. For data collection a smartphone (Samsung Galaxy S2) attached to the waist of the participants was used. The data set contains raw sensory data collected at 50Hz from the accelerometer and gyroscope of the smartphone as well as the labels of the current activity ^[Activities: walking, walking_upstairs, walking_downstairs, sitting, standing, laying] or the current transition between two activities ^[Transitions between activities: stand_to_sit, sit_to_stand, sit_to_lie, lie_to_sit, stand_to_lie, lie_to_stand]. The labels refer to specific time windows of the sensor data and do not cover the entire data set. That means, there are data points which are not labeled. See @fig:raw_data_walking for an example of how the data looks like. Besides the raw sensory data the data set also contains pre-processed data, but which is not used in this thesis.

::: {.cell .code tags=['hide']}
```python
import sys
import matplotlib.pyplot as plt
sys.path.append('code')
from plotting import plot_raw_experiment_data, set_new_plt_color_cycle

plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = [15, 4]
set_new_plt_color_cycle('plasma', 3, min=0,max=0.8)

plot_raw_experiment_data(experiment_nr=1, sample_start=7496, sample_end=8078,
                         sampling_frequency=50)
plt.savefig('figures/raw_data.png', bbox_inches='tight', pad_inches=0)

```
::: {.cell .markdown}
![Raw sensory data of gyroscope and accelerometer while walking collected at 50Hz.](figures/raw_data.png){#fig:raw_data_walking}

### Data split

To ensure an accurate performance measurement of the tested models the data is split into a train (60%), test (20%) and validation (20%) set. The data split is based on participant ids, which were randomly assigned once to one of the three groups before the testing of implementations started. The validation set is only used to test the performance of the final model.

## Language and packages

The pipeline is implemented in Python 3 using standard data science packages included in the anaconda distribution for python 3.71 on Linux ^[https://docs.anaconda.com/anaconda/packages/py3.7_linux-64/ (2019-06-06)]. Additionally, the tensorflow-gpu ^[https://www.tensorflow.org/ (2019-06-06)] package is used, a deep learning library that utilizes both CPU and GPU for building machine learning models. Tensorflow provides only a low level API, therefore, the keras package ^[https://keras.io/ (2019-06-06)] is used on top, which provides a high-level API for implementing neural networks.

## Machine setup used for building pipeline

The models were trained on a local desktop computer with a Nvidia GPU. To use the GPU for training tensorflow models Nvidia’s parallel computing platform CUDA ^[https://developer.nvidia.com/cuda-zone (2019-06-06)] and Nvidia’s GPU-accelerated library of primitives for deep neural networks cuDNN ^[https://developer.nvidia.com/cudnn (2019-06-06)] need to be installed. It is important to install matching versions of GPU drivers, CUDA and cuDNN. The installation process can be quite cumbersome and the documentation ^[https://docs.nvidia.com/cuda/index.html (2019-06-06)] should be closely followed. A complete list of hardware specifications, drivers and essential software packages used in this thesiss are listed in [@tbl:machine_specs].

Hardware/Software | Model/Version
--- | ---
GPU | Nvidia GeForce 970 GTX 4GB VRAM
CPU| AMD FX 8350 8-Cores
Memory| 8 GB DDR3 at 1666 MHz
Operating System| Kubuntu 18.04 64-bit, kernel 4.15.0
GPU OpenGL driver| Nvidia 415.27
CUDA version| 10.0
Cudnn version| 7.4.2

: Hardware and Software specifications of the machine used for training the models. {#tbl:machine_specs}


## Reproducability and code structure

This entire thesis is fully reproducible including data acquisition, data pre-processing, modeling and plots. The thesis was written in markdown ^[https://www.markdownguide.org/ (2019-06-06)] including python code blocks. Reusable code is kept in custom python modules and imported within the python code blocks. The markdown files where first converted to jupyter notebooks ^[https://jupyter.org/ (2019-06-06)], then executed and then converted back to markdown files. The final PDF file was then created using pandoc ^[https://pandoc.org/ (2019-06-06)].

See @sec:reproducable_repo in the appendix for a reference where you can find the digital repository of this reproducible thesis. See @sec:colab in the appendix for a guide on how to run parts of this thesis with minimal setup online using Google Colaboratory. See @sec:local_setup in the appendix for how to setup a local machine and reproduce the entire thesis. For some of the crucial custom ETL and pre-processing modules unit tests were written. See @sec:unittests for how to run them.
