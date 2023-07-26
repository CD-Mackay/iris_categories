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

## Same for population data
population = tf.keras.layers.Normalization(
    name='normalization_population',
    axis=None)
population.adapt(train_df['population'])
population = population(inputs.get('population'))

## Create list of boundaries for bucketing latitude
latitude_boundaries = np.linspace(-3, 3, 20+1)

## Create normalization layer for latitude
latitude = tf.keras.layers.Normalization(
    name='normalization_latitude',
    axis=None)
latitude.adapt(train_df['latitude'])
latitude = latitude(inputs.get('latitude'))

## Create discretization layer to separate latitude data into buckets
latitude = tf.keras.layers.Discretization(
    bin_boundaries=latitude_boundaries,
    name='discretization_latitude'
)(latitude)

# Create a list of numbers representing the bucket boundaries for longitude.
longitude_boundaries = np.linspace(-3, 3, 20+1)

# Create a Normalization layer to normalize the longitude data.
longitude = tf.keras.layers.Normalization(
    name='normalization_longitude',
    axis=None)
longitude.adapt(train_df['longitude'])
longitude = longitude(inputs.get('longitude'))

# Create a Discretization layer to separate the longitude data into buckets.
longitude = tf.keras.layers.Discretization(
    bin_boundaries=longitude_boundaries,
    name='discretization_longitude')(longitude)