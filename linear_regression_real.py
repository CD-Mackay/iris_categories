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
    
    return model
    

def train_model(model, df, feature, label, epochs, batch_size):
    history = model.fit(y=df[feature], x=df[label], batch_size=batch_size, epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    
    epochs = history.epoch

    hist = pd.DataFrame(history.history)

    rmse = hist['root_mean_squared_error']

    return trained_weight, trained_bias, epochs, rmse
    
def plot_model(trained_weight, trained_bias, feature, label):
    plt.xlabel=('feature')
    plt.ylabel=('label')

    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
    y1_nested = trained_bias + (trained_weight * x1) 
    y1 = [item for sub_list in y1_nested for item in sub_list]

    plt.plot([x0, x1], [y0, y1], c='r')
  

def plot_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel = 'epoch'
    plt.ylabel = 'root mean squared error'

    plt.plot(epochs, rmse, label='Loss')
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


learning_rate = 0.01
epochs = 30
batch_size = 30

## Specify Feature and Label
my_feature = "total_rooms" 
my_label = 'median_house_value'
## This model will predict house value based solely on total rooms

my_model = None ## remove any previous versions of model

my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df, my_feature, my_label, epochs, batch_size)

plot_model(weight, bias, my_feature, my_label)
plot_loss_curve(epochs, rmse)
