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
    model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)

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
    rmse = hist['root_mean_squared_error']

    return epochs, rmse
    

def plot_the_loss_curve(epochs, rmse):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.94, rmse.max()* 1.05])
  plt.show()  


## Hyperparameters
learning_rate = 0.04
epochs = 35
batch_size = 100
label_name = 'median_house_value'

preprocessing_layer = tf.keras.layers.Concatenate()(inputs.values())

# # The two Input layers are concatenated so they can be passed as a single
# # tensor to a Dense layer.
# dense_output = layers.Dense(
#    units = 1,
#    input_shape = (1,),
#    name='dense_layer'
# )(preprocessing_layer)

# outputs = {
#    'dense_output': dense_output
# }

resolution_in_degrees = 0.2

latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                      int(max(train_df['latitude'])),
                                      resolution_in_degrees))
print("latitude boundaries:", str(latitude_boundaries))

# Create a Discretization layer to separate the latitude data into buckets.
latitude = tf.keras.layers.Discretization(
    bin_boundaries=latitude_boundaries,
    name='discretization_latitude')(inputs.get('latitude'))


# Create a list of numbers representing the bucket boundaries for longitude.
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), 
                                      int(max(train_df['longitude'])), 
                                      resolution_in_degrees))

print("longitude boundaries: " + str(longitude_boundaries))

longitude = tf.keras.layers.Discretization(
   bin_boundaries = longitude_boundaries,
   name='discretization_longitude'
)(inputs.get('longitude'))

feature_cross = tf.keras.layers.HashedCrossing( ## Create cross_feature by combining lat and long values
   num_bins = len(latitude_boundaries) * len(longitude_boundaries),
   output_mode = 'one_hot',
   name='cross_latitude_longitude'
)([latitude, longitude])

dense_output = layers.Dense(
   units=1, input_shape=(2,), name='dense_layer'
)(feature_cross)

outputs = {
   'dense_output': dense_output
}


my_model = create_model(inputs, outputs, learning_rate)


# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

# Print out the model summary.
my_model.summary(expand_nested=True)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)