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

# Input lat/long values as float 
inputs = {
    'latitude':
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='latitude'),
    'longitude':
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='longitude')
}

def create_model(my_inputs, my_outputs, my_learning_rate):
    model = tf.keras.model(inputs=my_inputs, outputs=my_outputs)

    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(
      learning_rate=my_learning_rate),
      loss="mean_squared_error",
      metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model
    
def train_model(model, dataset, epochs, batch_size, label_name):
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = history['root_mean_squared_error']

    return epochs, rmse
    

def plot_the_loss_curve():