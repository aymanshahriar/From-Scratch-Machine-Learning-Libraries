import pandas as pd
import numpy as np

from KNearestNeighbors import MyKNeighborsClassifier

def distance(movieA, movieB):
  movieA = movieA.astype('int64')
  movieB = movieB.astype('int64')
  body = 0
  for feature_A,feature_B in zip(movieA, movieB):
    body = body + ((feature_A - feature_B) ** 2)
  distance = body ** 0.5
  return distance

star_wars = np.array([125, 1977, 11000000])
raiders = np.array([115, 1981, 18000000])
mean_girls = np.array([97, 2004, 17000000])


model = MyKNeighborsClassifier()
print(distance(star_wars, raiders), model.get_distance(star_wars, raiders))
print(distance(star_wars, mean_girls), model.get_distance(star_wars, mean_girls))
