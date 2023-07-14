import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv(filepath_or_buffer="california_housing_train.csv")

training_df['Median_house_value'] = 1000.0
training_df.head()

training_df.describe(include='all')
print(training_df.describe())

def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    

def train_model(model, feature, label, epochs):
    