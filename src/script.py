import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
from preperation import mergeData
from preperation import BofW
from preperation import autoLabelEncoding


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
X = pd.DataFrame(data['genre'].copy())
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
cv = KFold(n_splits=5, shuffle=True)
for train_index_array, test_index_array in cv.split(X):

	X_train = X.iloc[train_index_array,:]
	y_train = y.iloc[train_index_array,:]

	X_test  = X.iloc[test_index_array,:]
	y_test  = y.iloc[test_index_array,:]
