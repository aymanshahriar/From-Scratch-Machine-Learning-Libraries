import pandas as pd
import numpy as np
from collections import Counter
import csv

class MyKNearestClassifier:

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    # Get distance between two n-dimensional points
    # (assume they are 1d arrays only containing the features, not target)
    def get_distance(self, datapointA, datapointB):
        # Make sure data type to int64 to prevent overflow. By default its int32
        datapointA = datapointA.astype('int64')
        datapointB = datapointB.astype('int64')
        sum_squared_diff = 0
        for featureA, featureB in zip(datapointA, datapointB):
            squared_diff = (featureA - featureB) ** 2
            sum_squared_diff += squared_diff
        distance = sum_squared_diff ** 0.5
        return distance

    # Min-max normalize:
    # Set the min to be 0 and max to be 1. The other numbers will transform to values between 0 and 1, depending on their distance from the min and max
    # Input will be a 2d matrix of features, where each row corresponds to the features of a single datapoint
    def minmax_normalize(self, X):
        # In order to loop through each column, just loop through the transposed matrix
        # A transposed matrix is just a new matrix where the rows of original matrix are columns and vice versa.
        transposed_X = zip(*X)
        normalized_features = []
        for feature in transposed_X:
            normalized_feature = []
            minimum = min(feature)
            maximum = max(feature)
            for value in feature:
                normalized_value = (value-minimum)/(maximum-minimum)
                normalized_feature.append(normalized_value)
            normalized_features.append(normalized_feature)
        # Transpose again to make sure each row corresponds to the features of a single datapoint
        minmax_normalized_X = np.array(list(zip(*normalized_features)))
        return minmax_normalized_X

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    # For an unknown datapoint, find its nearest k neighbors.
    # Classify the new point based on those neighbors
    def predict_single_datapoint(self, unknown):
        # Find the distance between this datapoint and every other datapoint.
        # Distances is a 2d array, storing [distance, target] of each datapoint.
        distances = []
        for known_datapoint in self.X_train:
            distance = self.get_distance(unknown, known_datapoint)
            distances.append(distance)
        # Sort the distances, return the distance. By default, I think a 2d array will be sorted by the first element
        # in the subarray.
        distances.sort()
        neighbors = distances[0:self.k]

        # Now count the most common class/category among the neighbors, and that will be out predicted category
        # for the unknown datapoint.
        neighbors_features = neighbors[:, 1]
        counter = Counter(neighbors_features)
        most_common_feature = counter.most_common(1)[0][0]
        return most_common_feature





















