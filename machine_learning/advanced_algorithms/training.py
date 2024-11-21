import numpy as np
import matplotlib.pyplot as plt
plt.style.use(r"C:\Users\Saw\Desktop\machine learning\machine_learning_specialization\machine_learning\advanced_algorithms\deeplearning.mplstyle")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
%matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)