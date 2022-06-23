import pandas as pd
from sklearn.model_selection import train_test_split
from KNearestNeighbors import MyKNeighborsClassifier

############################################################################################################
def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #print(distances[0:k+1])
  #print([labels[title] for title in [a[1] for a in distances[0:k+1]]])
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

########################################################################################################
dataframe = pd.read_csv("data.csv")
X = dataframe[['feature1', 'feature2', 'feature3']].values
y = dataframe['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


X_dict = {}
for title, features in zip(dataframe['title'], X_train):
    X_dict[title] = list(features)

y_dict = {}
for title, target in zip(dataframe['title'], y_train):
    y_dict[title] = target

y_pred_sololearn = []
for unknown_datapoint in X_test:
    predicted_target = classify(list(unknown_datapoint), X_dict, y_dict, 3)
    y_pred_sololearn.append(predicted_target)

##################################################################

model = MyKNeighborsClassifier(k=3)
model.fit(X_train, y_train)

y_pred_me = []
for unknown_datapoint in X_test:
    predicted_target = model.predict_single_datapoint(unknown_datapoint)
    y_pred_me.append(predicted_target)

for a, b in zip(y_pred_me, y_pred_sololearn):
    print(a, b)

#print(model.predict_single_datapoint(X_test[-5]), classify(list(X_test[-5]), X_dict, y_dict, 3))
