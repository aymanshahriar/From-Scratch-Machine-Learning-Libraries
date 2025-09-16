import pandas as pd
from sklearn.model_selection import train_test_split
from KNearestNeighbors import MyKNeighborsRegressor

########################################################################################################################
'''def predict_single_datapoint(unknown, dataset, movie_ratings, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  sum_neighbor_targets = 0
  for neighbor in neighbors:
    title = neighbor[1]
    target = movie_ratings[title]
    sum_neighbor_targets += target
  avg_neighbor_targets = sum_neighbor_targets / k
  return avg_neighbor_targets'''


def predict_single_datapoint(unknown, dataset, movie_ratings, k):
  distances = []
  # Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    # Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  # Taking only the k closest points
  neighbors = distances[0:k]

  # compute the weighted average
  numerator = 0
  denominator = 0
  dist = 0
  for neighbor in neighbors:
    dist = neighbor[0]
    title = neighbor[1]
    numerator += movie_ratings[title] / dist
    denominator += 1 / dist
  weighted_average = numerator / denominator
  return weighted_average

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(X_test_dict, dataset, labels, k):
  y_pred = []
  for title in X_test_dict:
    y_pred.append(predict_single_datapoint(X_test_dict[title], dataset, labels, k))
  return y_pred
###########################################################################################################################

dataframe = pd.read_csv("data.csv")
X = dataframe[['feature1', 'feature2', 'feature3']].values
y = dataframe['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_dict_train = {}
for title, features in zip(dataframe['title'], X_train):
  X_dict_train[title] = list(features)

X_dict_test = {}
for title, features in zip(dataframe['title'], X_test):
  X_dict_test[title] = list(features)

y_dict_train = {}
for title, target in zip(dataframe['title'], y_train):
  y_dict_train[title] = target

y_pred_codecademy = predict(X_dict_test, X_dict_train, y_dict_train, 3)

model = MyKNeighborsRegressor(k=3, weighted=True)
model.fit(X_train, y_train)
y_pred_me = model.predict(X_test)

for a in y_pred_me:
  print(a)


#for a, b in zip(y_pred_me, y_pred_codecademy):
#  print(a, b)
###########################################################################################################################

#Now test with weighted mean








