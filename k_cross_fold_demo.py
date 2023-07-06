from numpy import array
from sklearn.model_selection import KFold

dataset = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

kfold = KFold(n_splits=3, shuffle=True, random_state=1)

for train, test in kfold.split(dataset):
    print('train, %s, test, %s' % (dataset[train], dataset[test]))

