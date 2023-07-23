import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
# tf.keras.backend.set_floatx('float32')

train_df = pd.read_csv('california_housing_train.csv')
test_df = pd.read_csv('california_housing_test.csv')
train_df = train_df.reindex(np.random.permutation(train_df.index)) ## shuffle training data

## Normalize data values

train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean) / train_df_std

test_df_mean = test_df.mean()
test_df_std  = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std



# Examine some of the values of the normalized training set. Notice that most 
# Z-scores fall between -2 and +2.
train_df_norm.head()

threshold = 265000
train_df_norm['median_house_value_is_high'] = (train_df['median_house_value'] > threshold).astype(float)
test_df_norm['median_house_value_is_high'] = (test_df['median_house_value'] > threshold).astype(float)

train_df_norm["median_house_value_is_high"].head(8000)
print(train_df_norm["median_house_value_is_high"].head(8000))

## Define features to be used to train model
inputs = {
    'median_income': tf.keras.Input(shape=(1,)),
    'total_rooms': tf.keras.Input(shape=(1,))
}

def create_model(my_inputs, my_learning_rate, METRICS):
    concatenated_inputs = tf.keras.layers.Concatenate()(my_inputs.values())
    dense = layers.Dense(units = 1, input_shape=(1,), name='dense_layer', activation=tf.sigmoid)
    dense_output = dense(concatenated_inputs)
 ##  """Create and compile a simple classification model."""
    my_outputs = {
    'dense': dense_output,
    }
    model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)

  # Call the compile method to construct the layers into a model that
  # TensorFlow can execute.  Notice that we're using a different loss
  # function for classification than for regression.    
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),                                                   
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    return model    

