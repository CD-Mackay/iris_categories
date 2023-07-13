import numpy as np
import pandas as pd

my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

my_collumn_names = ['temperature', 'activity']

my_dataframe = pd.DataFrame(data=my_data, columns=my_collumn_names)
print(my_dataframe)

my_dataframe['adjusted'] = my_dataframe['activity'] + 2
print(my_dataframe)

## Selecting subsets of dataset
print("rows 0, 1 and 2")
print(my_dataframe.head(3), '\n')

print("row 2 only")
print(my_dataframe.iloc[[2]], '\n') ## Iloc = integer-location based indexing fr selection by position

print('rows 1, 2 and 3')
print(my_dataframe[1:4], '\n')

print('temperature column only')
print(my_dataframe['temperature'])

## Task 1, Create a Dataframe

rows = np.random.randint(high=101, low=0, size=(3, 4))
print(rows)
good_labels = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
good_dataframe = pd.DataFrame(data=rows, columns=good_labels)
print(good_dataframe)
print(good_dataframe['Eleanor'].iloc[[1]]) ## Print value in row#1 of eleanor column

good_dataframe['Janet'] = good_dataframe['Jason'] + good_dataframe['Tahani']
print(good_dataframe)
