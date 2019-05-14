import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.utils import resample

#preperation
from preperation import BofW_Giver

#Logistical Regression
from LogisticalRegression import LogiRegrTrain
from LogisticalRegression import LogiRegrPredict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Bayes
from BayseianClassification import probMatrixGen
from BayseianClassification import multinomialProb


whatWeGuess = 'valence'

#  --------------------------------------
#    Part One: Prepare data for 
#			   classification           
#  -------------------------------------

#Step Zero: merge data and create bag-of-words representatation of lyrics
#- - - - - - - - - - - - - - - - - -
data = pd.read_csv("C:/Users/iamro/Documents/School/AI Class/Project/data/refined.csv")
if (1==0):
	bow = BofW_Giver(data)
	print(bow.head())
	bow.to_csv("C:/Users/iamro/Documents/School/AI Class/Project/data/bow.csv")


#Step One: set Features and Labels
#- - - - - - - - - - - - - - - - - -
bow = pd.read_csv("C:/Users/iamro/Documents/School/AI Class/Project/data/bow.csv", index_col=0)
bow = bow.drop(columns=['1', '2', 'Unnamed: 1'])
ourFeatures = list(bow.columns.values)
X =  pd.DataFrame(bow.copy())
bow[whatWeGuess] = pd.DataFrame(data[whatWeGuess].copy())
y = pd.DataFrame(bow[whatWeGuess].copy())

mergedXY = pd.DataFrame(bow.copy())




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


mergedXY = pd.DataFrame(pd.merge(X, y, left_index=True, right_index=True))

countpop = 0
for index, row in y.iterrows():
	if (y.iloc[index, 0] == 1):
		countpop += 1
print("\n % ", whatWeGuess, ": ", countpop/y.shape[0])
print()
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
cv = KFold(n_splits=5, shuffle=False)
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

	one_yy=0
	for index, row in y_test.iterrows():
		if y_test.iloc[index, 0] == 1:
				one_yy += 1


	print("true on_y: ", one_yy/y_test.shape[0])
	#LOGISTICAL REGRESSION
	if (1==0):
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
		print("1 count in preperation: ", count1)
		
		model = LogisticRegression(solver='lbfgs')
		model.fit(X_train, y_train.to_numpy().ravel())
		parameters = model.coef_
		predicted_classes = model.predict(X_test)
		SKaccuracy = accuracy_score(y_test.to_numpy().ravel().flatten(), predicted_classes)
		print('sklearn accuracy: ', format(SKaccuracy, ".2f"))


	#MULTINOMIAL NAIVE BAYES CLASSIFICATION
	if(1==1):
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

		print("Building probability Matrix...", end="")
		probMatrix = probMatrixGen(merged_train, whatWeGuess, ourFeatures)
		print("[DONE]")
		#print("probMatrix.shape: ", probMatrix.shape)

		howManyCorrect=0
		oneCount=0
		one_y =0
		for index, row in y_test.iterrows():
			guess = multinomialProb(merged_train, whatWeGuess, row, ourFeatures, probMatrix)

			if (guess == 1):
				oneCount += 1

			if y_test.iloc[index, 0] == 1:
				one_y += 1
			
			if (guess == y_test.iloc[index, 0]):
				howManyCorrect = howManyCorrect + 1
				
		print("Round: ", fold)
		print(" ------------------------------------ ")	
		print("| Bayes Multinomial Accuracy: ", howManyCorrect/y_test.shape[0], )
		print(" ------------------------------------ ")
		print("'1' count in its predictions: ", oneCount/y_test.shape[0])
		print('one_y: ', one_y/y_test.shape[0])
		print()

	fold += 1