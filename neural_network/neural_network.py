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

## Cross latitude and longitude boundaries into feature cross
feature_cross = tf.keras.layers.HashedCrssing(
    num_bins=len(latitude_boundaries) * len(longitude_boundaries),
    output_mode='one_hot',
    name='cross_latitude_longitude')([latitude, longitude])

preprocessing_layers = tf.keras.layers.Concatenate()(
    [feature_cross, median_income, population]
)

def plot_the_loss_curve(epochs, mse_training, mse_validation):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse_training, label="Training Loss")
    plt.plot(epochs, mse_validation, label="Validation Loss")

    merged_mse_lists = mse_training.tolist() + mse_validation
    highest_loss = max(merged_mse_lists)
    lowest_loss = min(merged_mse_lists)
    top_of_y_axis = highest_loss * 1.03
    bottom_of_y_axis = lowest_loss * 0.97

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.legend()
    plt.show()

def create_model(my_inputs, my_outputs, my_learning_rate):
    model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=my_learning_rate
    ),
    loss="mean_squared_error",
    metrics=[tf.keras.metrics.MeanSquaredError()]
    )

    return model



