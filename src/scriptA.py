import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.utils import resample

#preperation
from preperation import mergeData
from preperation import BofW
from preperation import autoLabelEncoding

#Logistical Regression
from LogisticalRegression import LogiRegrTrain
from LogisticalRegression import LogiRegrPredict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#KNN
from NearestKClassifiction import KNNMeasure
from NearestKClassifiction import KNNPredict
from NearestKClassifiction import KNNMeasureMatrix
from sklearn.neighbors import KNeighborsClassifier

#Naive Bayes
from BayseianClassification import generateSTDTable
from BayseianClassification import generateMeanTable
from BayseianClassification import BayesPredict
from sklearn.naive_bayes import GaussianNB


#  --------------------------------------
#    Part One: Prepare data for 
#			   classification           
#  --------------------------------------

whatWeGuess = 'valence'
ourFeatures = ['genre', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'popularity']

#Step Zero: merge data
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

X = pd.DataFrame(data[ourFeatures].copy())
y = pd.DataFrame(data[whatWeGuess].copy())



#Step Four: normalize and threshold label values (making it binary) 
#- - - - - - - - - - - - - - - - - -
#normaize
y_values = y.values.astype(float)
min_max_normalizer = preprocessing.MinMaxScaler()
y_scaled = min_max_normalizer.fit_transform(y_values)
y = pd.DataFrame(y_scaled)

#threshold
for index, row in y.iterrows():
	if (y.iloc[index, 0] >= 0.5):
		y.iloc[index, 0] = 1
	else:
		y.iloc[index, 0] = 0
y.rename(columns={0: whatWeGuess}, inplace=True)

countpop = 0
for index, row in y.iterrows():
	if (y.iloc[index, 0] == 1):
		countpop += 1

mergedXY = pd.DataFrame(pd.merge(X, y, left_index=True, right_index=True))

print("\t[DONE]")

print("\n % ", whatWeGuess, ": ", countpop/y.shape[0])
print()

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
	merged_train = mergedXY.iloc[train_index_array,:]

	X_test  = X.iloc[test_index_array,:]
	y_test  = y.iloc[test_index_array,:]
	merged_test = mergedXY.iloc[test_index_array,:]



	X_train = X_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	X_test  = X_test.reset_index(drop=True)
	y_test  = y_test.reset_index(drop=True)	
	merged_test = merged_test.reset_index(drop=True)
	merged_train = merged_train.reset_index(drop=True)



	#LOGISTICAL REGRESSION
	if (1==1):
		#We need to even out our data proportions
		#Step Zero: divide the trianing data into class=0 and class=1
		mTrain_majority = merged_train[merged_train[whatWeGuess]==0]
		mTrain_minority = merged_train[merged_train[whatWeGuess]==1]

		#Step One: now we both randomly resample both of these class to the
		#          size of the smaller dataset class
		sampleSize = mTrain_minority.shape[0]
		mTrain_majority_downsampled = resample(mTrain_majority, replace=False, n_samples=sampleSize, random_state=123)

		merged_train = pd.concat([mTrain_majority_downsampled, mTrain_minority])

		X_train = merged_train[ourFeatures]
		y_train = merged_train[whatWeGuess]


		weights = LogiRegrTrain(X_train, y_train)
		predictions = LogiRegrPredict(X_test, weights)
		predictions = pd.DataFrame(data=predictions)

		for index, row in predictions.iterrows():
			if predictions.iloc[index,0] >= 0.5:
				predictions.iloc[index,0] = 1
			else:
				predictions.iloc[index,0] = 0

		correctCount=0
		
		count0=0
		count1=1
		for index, row in y_test.iterrows():
			if predictions.iloc[index,0] == y_test.iloc[index,0]:
				correctCount += 1
			if predictions.iloc[index,0] == 0:
				count0 += 1
			if predictions.iloc[index,0] == 1:
				count1 += 1



		#print("best Wghts: ", weights)
		print("\nRound: ", fold)
		print(" ---------------------------------------- ")	
		print("| Logistical Regression Accuracy: ", format(correctCount/y_test.shape[0], ".2f"), " |")
		print(" ---------------------------------------- ")
		
		model = LogisticRegression(solver='lbfgs')
		model.fit(X_train, y_train.to_numpy().ravel())
		parameters = model.coef_
		predicted_classes = model.predict(X_test)
		SKaccuracy = accuracy_score(y_test.to_numpy().ravel().flatten(), predicted_classes)
		print('sklearn accuracy: ', format(SKaccuracy, ".2f"))

	#NEAREST-K CLASSIFICATION
	if (1==0):
		#We need to even out our data proportions
		#Step Zero: divide the trianing data into class=0 and class=1
		mTrain_majority = merged_train[merged_train[whatWeGuess]==0]
		mTrain_minority = merged_train[merged_train[whatWeGuess]==1]

		#Step One: now we both randomly resample both of these class to the
		#          size of the smaller dataset class
		sampleSize = mTrain_minority.shape[0]
		mTrain_minority_downsampled = resample(mTrain_minority, replace=False, n_samples=sampleSize, random_state=123)
		mTrain_majority_downsampled = resample(mTrain_majority, replace=False, n_samples=sampleSize, random_state=123)

		merged_test = resample(merged_test, replace=sampleSize, n_samples=100, random_state=123)
		merged_train = pd.concat([mTrain_majority_downsampled, mTrain_minority])

		#KNNMeasureMatrix(merged_train, whatWeGuess)		
		k = int(np.sqrt(merged_train.shape[0]))
		#k=1
		
		guesses = []
		colNum = merged_train.columns.get_loc(whatWeGuess)
		merged_test = merged_test.to_numpy()
		merged_train = merged_train.to_numpy()
		for index in range(0, merged_test.shape[0]):
			distsToTestpoint = KNNMeasure(merged_train, merged_test[index], colNum)
			guess = KNNPredict(distsToTestpoint, k)
			guesses.append(guess)

		correct=0
		for i in range(0, len(guesses)):
			if guesses[i] == merged_test[i, colNum]:
				correct+=1

		
		print("\nRound: ", fold)
		print(" ------------------------ ")	
		print("| KNN Accuracy : ", format(correct/len(guesses), ".2f"), " |")
		print(" ------------------------ ")

		#SKLEARN METHOD
		correct=0
		knn = KNeighborsClassifier(n_neighbors=5)
		knn.fit(X_train, y_train.to_numpy().ravel())
		for i in range(0, merged_test.shape[0]):
			if knn.predict(X_test.iloc[i, :].ravel().reshape(1, -1)) == merged_test[i, colNum]:
		 		correct+=1
		print("sklearn knn accuracy: ", correct/merged_test.shape[0])


	#NAIVE BAYESIAN CLASSIFICATION V2
	if(1==0):
		#We need to even out our data proportions
		#Step Zero: divide the trianing data into class=0 and class=1
		mTrain_majority = merged_train[merged_train[whatWeGuess]==0]
		mTrain_minority = merged_train[merged_train[whatWeGuess]==1]

		sampleSize = mTrain_minority.shape[0]

		#Step One: now we both randomly resample both of these class to the
		#          size of the smaller dataset class
		mTrain_majority_downsampled = resample(mTrain_majority, replace=False, n_samples=sampleSize, random_state=123)

		merged_train = pd.concat([mTrain_majority_downsampled, mTrain_minority])



		stdTable  = generateSTDTable(merged_train, whatWeGuess)
		meanTable = generateMeanTable(merged_train, whatWeGuess)

		correctCount=0
		count1=0
		for index, row in merged_test.iterrows():
			if (BayesPredict(row, meanTable, stdTable) == row[whatWeGuess]):
				correctCount+=1
			if (BayesPredict(row, meanTable, stdTable) == 1):
				count1+=1

		print("\nRound: ", fold)
		print(" ------------------------ ")	
		print("| Bayes V2 Accuracy: ", format(correctCount/merged_test.shape[0], ".2f"), " |")
		print(" ------------------------ ")
		print("'1' count in its predictions: ", count1)

		#print("merged_train[merged_train[whatWeGuess]==0].shape[0]: ", merged_train[merged_train[whatWeGuess]==0].shape[0])
		#print("merged_train[merged_train[whatWeGuess]==1].shape[0]: ", merged_train[merged_train[whatWeGuess]==1].shape[0])

		gnb = GaussianNB()
		
		X_train = merged_train[ourFeatures]
		y_train = merged_train[whatWeGuess]


		gnb.fit(X_train, y_train.to_numpy().ravel())
		y_pred = gnb.predict(X_test)
		skcorrect=0
		index=0
		for element in y_pred:
			if (element == y_test.iloc[index, :].item()):
				skcorrect+=1
			index+=1

		print("sklearn accuracy: ", format((skcorrect/merged_test.shape[0]), ".2f"))


	fold += 1

