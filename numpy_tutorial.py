import numpy as np

one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)

zeroes_array = np.zeros((2, 3)) ## np.zeros creates one or two dimensional arrays composed entirely of zeros
print(zeroes_array)

sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

random_int_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_int_between_50_and_100)

random_floats_between_0_and_1 = np.random.random([3])
print(random_floats_between_0_and_1)

## Task 1, Create Linear Dataset
feature = np.arange(6, 21)
print("feature", feature)
label = (feature * 3) + 4
print("label", label)

## Task 2, create noise
noise = (np.random.random([15]) * 4) -2 ## Generate array of random ints between -2 and 2
print("noise", noise)
label = label + noise ## Add each value of noise array to label
print("label", label)