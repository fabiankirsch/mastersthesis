# This file only exists to check if the tensor flow has access to a GPU for processing

import tensorflow as tf
tf.test.gpu_device_name()
tf.__version__
