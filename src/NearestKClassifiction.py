import numpy as np
import pandas as pd
# Decr: This function iterates over all the datapoints from the inputed training data
#	 	and for each datapoint measures its distance to the inputed datapoint from the 
#       test data. It then stores the distance and the class type of the training datapoint
#       in a list, and returns this list sorted by the distance values.
# Input: myTrain: training set pd.Dataframe
#        myTestpoint: single row (aka datapoint) form the test set (pd.Dataframe or pd.Series)
#		 classColName: name of the column with the class info form the training set (string)
# Output: sorted distances with class info
def KNNMeasure(myTrain, myTestpoint, classColName):
	
	distances = []
	for index, datapoint in myTrain.iterrows():
		classType = datapoint[classColName]
		dist = euclideanDistance(myTestpoint, datapoint)
		distances.append([dist, classType])

	
	return sorted(distances, key = lambda tup: tup[0])


# Decr: This function is calcualtes the distnace between a testpoint and trainpoin
#       or any two datapoints really. 
# Input: myTestpoint: single row (aka datapoint) form the test set (pd.Dataframe or pd.Series)
#        myTrainpoint: single row (aka datapoint) form the train set (pd.Dataframe or pd.Series)
# Output: distance value (float)
def euclideanDistance(myTestpoint, myTrainpoint):
	
	totalSoFar=0
	features = myTrainpoint.shape[0] - 1 #as it is a series of just features, we can do this
	for feat in range(0, features):
		totalSoFar += np.power((myTrainpoint.iloc[feat] - myTestpoint.iloc[feat]), 2)

	return np.sqrt(totalSoFar)

# Descr: This function goes through the k closest datapoints and counts how many
#        of them are of class=0. Then using that data it can understand what class
#        the majority of these k closest datapoints are, and then make a class to
#        predict the testdatapoint is of class=0 or class=1.
# Input: sortedDistances: should be produced from KNNMeasure() class (sorted list of tuples)
# 		 k: integer greater than 0
# Output: 0 or 1
def KNNPredict(sortedDistances, k):
	class0count = 0
	for index in range(0, k):
		if (sortedDistances[index][1] == 0):
			class0count += 1

	if (class0count/k >= 0.5):
		return 0
	else:
		return 1