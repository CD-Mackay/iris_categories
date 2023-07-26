import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the examples
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

inputs = {
    'latitude':
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='latitude'),
    'longitude':
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='longitude'),
    'population': 
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='population')
}

## Create normalization layer for median income
median_income = tf.keras.layers.Normalization(
    name='normalization_median_income',
    axis=None
)

median_income.adapt(train_df['median_income'])
