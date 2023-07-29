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
    'median_income':
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='median_income'),
    'population': 
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='population')
}

## Create normalization layer for median income
median_income = tf.keras.layers.Normalization(
    name='normalization_median_income',
    axis=None
)

median_income.adapt(train_df['median_income'])
median_income = median_income(inputs.get('median_income'))

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
# Cross the latitude and longitude features into a single one-hot vector.
feature_cross = tf.keras.layers.HashedCrossing(
    # num_bins can be adjusted: Higher values improve accuracy, lower values
    # improve performance.
    num_bins=len(latitude_boundaries) * len(longitude_boundaries), 
    output_mode='one_hot',
    name='cross_latitude_longitude')([latitude, longitude])

# Concatenate our inputs into a single tensor.
preprocessing_layers = tf.keras.layers.Concatenate()(
    [feature_cross, median_income, population])

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


## Create normalization layers for median_house_value
train_median_house_value_normalized = tf.keras.layers.Normalization(axis=None)
train_median_house_value_normalized.adapt(
    np.array(train_df['median_house_value'])
)

test_median_house_value_normalized = tf.keras.layers.Normalization(axis=None)
test_median_house_value_normalized.adapt(
    np.array(test_df['median_house_value'])
)

def train_model(model, dataset,epochs, batch_size, label_name, validation_split=0.1):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = train_median_house_value_normalized(
        np.array(features.pop(label_name))
    )
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist['mean_squared_error']

    return epochs, mse, history.history

def get_outputs_linear_regression():
    dense_output = tf.keras.layers.Dense(units=1, input_shape=(1,), name='dense_output')(preprocessing_layers)

    outputs = {
        'dense_output': dense_output
    }

    return outputs

def get_outputs_dnn():
    dense_output = tf.keras.layers.Dense(units=20, input_shape=(1,),
                                     activation='relu',
                                     name='hidden_dense_layer_1')(preprocessing_layers)

    dense_output = tf.keras.layers.Dense(units=12, input_shape=(1,),
                                     activation='relu',
                                     name='hidden_dense_layer_2')(dense_output)
    dense_output = tf.keras.layers.Dense(units=1, input_shape=(1,),
                                     activation='relu',
                                     name='dense_output')(dense_output)
    outputs = {
        'dense_output': dense_output
    }

    return outputs


## Hyperparameters
learning_rate = 0.01
epochs = 15
batch_size = 10000
label_name = 'median_house_value'

## split og training set into reduced set + validation set
validation_split = 0.2

outputs = get_outputs_linear_regression()

## Establish model topography
my_model = create_model(inputs, outputs, learning_rate)

## Train model on normalized training set
epochs, mse, history = train_model(my_model, train_df, epochs, batch_size, label_name, validation_split)
plot_the_loss_curve(epochs, mse, history['val_mean_squared_error'])

test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = test_median_house_value_normalized(test_features.pop(label_name))
print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size, return_dict=True)