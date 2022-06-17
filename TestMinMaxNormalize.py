
from random import randint
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from KNearestNeighbors import MyKNearestClassifier

# 1) Generate 2d matrix
rand_arr = [[randint(0, 100) for j in range(5)] for i in range(5)]

# 2) use sklearn's min-max normalize function and my own function
scaler = MinMaxScaler()
scaler.fit(rand_arr)
normalized_arr1 = scaler.transform(rand_arr)

scaler = MyKNearestClassifier()
normalized_arr2 = scaler.minmax_normalize(rand_arr)

# 3) Compare the results
print("sklearn normalized:\n", normalized_arr1)
print("\nmy normalized:\n", normalized_arr2)