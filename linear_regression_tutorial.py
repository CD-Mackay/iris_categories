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

def train_model(model, feature, label, epochs, batch_size):
     # Feed the feature values and the label values to the 
    # model. The model will train for the specified number 
    # of epochs, gradually learning how the feature values
    # relate to the label values. 

    history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epochs = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

     # Specifically gather the model's root mean 
     # squared error at each epoch. 
    rmse = hist['root_mean_squared_error']

    return trained_weight, trained_bias, epochs, rmse
    

def plot_the_model(trained_weight, trained_bias, feature, label):
    plt.xlabel('features')
    plt.ylabel('label')

    ## make scatterplot
    plt.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.show()


