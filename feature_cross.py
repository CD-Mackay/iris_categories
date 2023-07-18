import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

train_df = pd.read_csv('california_housing_train.csv')
test_df = pd.read_csv('california_housing_test.csv')

scale_factor = 1000.0

train_df['median_house_value'] /= scale_factor
test_df['median_house_value'] /= scale_factor

train_df = train_df.reindex(np.random.permutation(train_df.index))