import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

def build_model(my_learning_rate):
    # Most simple tf.keras models are sequential. 
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Compile model topography
    # Configure training to minimize mean squared error
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model    
    
