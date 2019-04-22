import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from preperation import mergeData
from preperation import BofW
from preperation import autoLabelEncoding

from LogisticalRegression import LogiRegrTrain
from LogisticalRegression import LogiRegrPredict

from NearestKClassifiction import KNNMeasure
from NearestKClassifiction import KNNPredict


#  --------------------------------------
#    Part One: Prepare data for 
#			   classification           
#  --------------------------------------



#Step Zero: merge data and create bag-of-words representatation of lyrics
#- - - - - - - - - - - - - - - - - -
if (1 == 0): 
	data = mergeData()
	print (data.head())
	data = BofW(data)
else:
	data = pd.read_csv("C:/Users/iamro/Documents/School/AI Class/Project/data/refined.csv")
	print("preperating data for classification...", end='')


#Step Two: turn catigorical data into data the classifieires can use
#- - - - - - - - - - - - - - - - - -
autoLabelEncoding(data)

 
#Step Three: set Features and Labels
#- - - - - - - - - - - - - - - - - -
#For now we will not inlcude lyrics, to see how it imporves the guess...
#'acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature'

X = pd.DataFrame(data[['genre', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature']].copy())
y = pd.DataFrame(data['popularity'].copy())


#Step Four: normalize and threshold label values (making it binary) 
#- - - - - - - - - - - - - - - - - -
#normaize
y_values = y.values.astype(float)
min_max_normalizer = preprocessing.MinMaxScaler()
y_scaled = min_max_normalizer.fit_transform(y_values)
y = pd.DataFrame(y_scaled)

#threshold
for index, row in y.iterrows():
	if (y.iloc[index, 0] >= 0.80):
		y.iloc[index, 0] = 1
	else:
		y.iloc[index, 0] = 0
y.rename(columns={0: 'popularity'}, inplace=True)


print("\t[DONE]")


#  --------------------------------------
#    Part Two: Classify using all
#              three types of 
#              classifiers 
#  --------------------------------------

#Step Zero: create 5-fold cross validation and start classifying
#- - - - - - - - - - - - - - - - - -
fold=1
countPopular=0
cv = KFold(n_splits=5, shuffle=True)
for train_index_array, test_index_array in cv.split(X):

	X_train = X.iloc[train_index_array,:]
	y_train = y.iloc[train_index_array,:]

	X_test  = X.iloc[test_index_array,:]
	y_test  = y.iloc[test_index_array,:]

	X_train = X_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	X_test  = X_test.reset_index(drop=True)
	y_test  = y_test.reset_index(drop=True)	


	#LOGISTICAL REGRESSION
	if (1==0):
		weights = LogiRegrTrain(X_train, y_train)
		predictions = LogiRegrPredict(X_test, weights)
		predictions = pd.DataFrame(data=predictions)

		for index, row in predictions.iterrows():
			if predictions.iloc[index,0] >= 0.5:
				predictions.iloc[index,0] = 1
			else:
				predictions.iloc[index,0] = 0

		correctCount=0
		
		for index, row in y_test.iterrows():
			if predictions.iloc[index,0] == y_test.iloc[index,0]:
				correctCount += 1
			if y_test.iloc[index,0] == 1:
					countPopular+=1

		print("Round: ", fold, "/5, pop:", countPopular)
		print(" ---------------------------------------- ")	
		print("| Logistical Regression accuracy: ", format(correctCount/y_test.shape[0], ".2f"), " |")
		print(" ---------------------------------------- ")



	#NEAREST-K CLASSIFICATION
	if (1==0):
		merged_train = pd.DataFrame(pd.merge(X_train, y_train, left_index=True, right_index=True))
		merged_test = pd.DataFrame(pd.merge(X_test, y_test, left_index=True, right_index=True))

		k = int(np.sqrt(merged_train.shape[0]))
		guesses = []
		for index, row in merged_test.iterrows():
			print(index,"/",merged_test.shape[0])
			distsToTestpoint = KNNMeasure(merged_train, row, 'popularity')
			guess = KNNPredict(distsToTestpoint, k)
			guesses.append(guess)

		correct=0
		for i in range(0, len(guesses)):
			if guesses[i] == y_test[i]:
				correct+=1

		print("knn accuracy: ", correct/len(guesses))
		print(guesses)


	#NAIVE BAYESIAN CLASSIFICATION




	#SUPPORT VECTOR MACHINE CLASSIFICATION
	fold += 1

