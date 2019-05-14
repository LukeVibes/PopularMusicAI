import pandas as pd 
import numpy as np




#Desrc: This functoin simply splits the training data into two dataframes where
#       the split is based on the y values. It is important to note that here
#       the training data must be a one that holds X dimensions and the y 
#       dimension
#Input: myTrain: trainind dataframe WITH X and y (Dataframe)
#	 	yName: name of y column (string) 
def classSplitter(myTrain, yName):

	c1 = myTrain[myTrain[yName] == 1]
	c0 = myTrain[myTrain[yName] == 0]
	return c0, c1

#Descr: this function generates the mean matrix. Essentially each row
#		in this matrix represents the possible y values (0,1) the 
#       values within each row are the means of ALL values in an attribute
#       when the y=0 or y=1 respectively. So to clarify, the column values
#       represent dimensions of the training data, number of rows = number of 
#       possible classes.
#Input: myTrain: trainind dataframe WITH X and y (Dataframe)
#Output: pynum table of mean values (pynum matrix)
def generateMeanTable(myTrain, yName):

	meanTable = np.zeros((2, myTrain.shape[1]-1))

	class0Train, class1Train = classSplitter(myTrain, yName)

	colIndex=0
	for label, values in class0Train.items():

		if(colIndex < myTrain.shape[1]-1):
			mean = meanGenertor(values)
			meanTable[0, colIndex] = mean
		colIndex += 1

	colIndex=0
	for label, values in class1Train.items():
		if(colIndex < myTrain.shape[1]-1):
			mean = meanGenertor(values)
			meanTable[1, colIndex] = mean
		colIndex += 1

	return meanTable

#Decr: Really similar to generateMeanTable but with standard 
#      deviation rather than mean. Just re-read generateMeanTable
#      decription but replace the word mean with std.
#Input: myTrain: trainind dataframe WITH X and y (Dataframe)
#Output: pynum table of std values (pynum matrix)
def generateSTDTable(myTrain, yName):

	stdTable = np.zeros((2, myTrain.shape[1]-1))

	class0Train, class1Train = classSplitter(myTrain, yName)

	colIndex=0
	for label, values in class0Train.items():
		if(colIndex < myTrain.shape[1]-1):
			std = stdGenerator(values)
			stdTable[0, colIndex] = std
		colIndex += 1

	colIndex=0
	for label, values in class1Train.items():
		if(colIndex < myTrain.shape[1]-1):
			std = stdGenerator(values)
			stdTable[1, colIndex] = std
		colIndex += 1

	return stdTable


def GenerateProbClass(datapoint, meanTable, STDTable, c):
	total = 1

	for i in range(datapoint.shape[0]-1):

		total *= gaussian(datapoint[i], meanTable[c,i], STDTable[c,i])

	return total

#Descr: given the datapoint feature, mean and std this function produces the gaussian
#       value
# Input: d:  values of specific feature in testpoint (float)
#		 mean: mean of feature (float)
#		 std: std of feature (float)
# Output: gauasian value (float)
def gaussian(d, mean, std):

	exp = np.exp(-(np.power(d-mean,2)/(2*np.power(std, 2))))
	result = (1/(np.sqrt(2*np.pi)*std))*exp
	return result


# Decr: returns the mean, simple.
def meanGenertor(values):
	# total=0
	# for i in range(0, len(values)):
	# 	total += values[i]

	# total = total/len(values)-1
	total = np.mean(values)
	return total

# Decr: returns the standard devation, simple.
def stdGenerator(values):
	# total=0

	# mean = meanGenertor(values)

	# for i in range(0, len(values)):

	# 	total += np.power((values[i] - mean), 2)

	# total = np.sqrt(total/(len(values)-1))
	total = np.std(values)


	return total

# Desrc: This function produces the probability values of a datapoint being of class=0 and
#		 the probability values of the datapoint being of class=1. It then the class of the larger
#        value.
# Input: datapoint: test point (Series)
#        meanTable: 2 by number-of-feature generated from meanTableGenerator()
#        STDTable: 2 by number-of-feature generated from STDTableGenerator()   
# Output: 1 or 0
def BayesPredict(datapoint, meanTable, STDTable):
	prob0 = GenerateProbClass(datapoint, meanTable, STDTable, 0)
	prob1 = GenerateProbClass(datapoint, meanTable, STDTable, 1)
	#print("prob0: ", prob0, "  prob1: ", prob1)
	if (prob0 > prob1):
		return 0
	else:
		return 1













#----------------------MULTINOMIAL-----------------------------


#Desr: this produces the probability of each class appearing in the data
#      set. because we do the 50/50 split this should be 0.5 and 0.5. But
#      if the code is altered to not do the 50/50 split this will give more
#      reflective data.
#Input: myTrain: trainind dataframe WITH X and y (Dataframe)
#	 	yName: name of y column (string)
def classProb(myTrain, myName):
	c0, c1 = classSplitter(myTrain, myName)

	return (c0.shape[0]/myTrain.shape[0]), (c1.shape[0]/myTrain.shape[0])

#Decr: this simply counts how many times a certain word comes up in a class
#      across all documents.
def countWordinClass(wordPos, classDF):
	count=0
	classDF = classDF.to_numpy()
	for i in range(0, classDF.shape[0]):
		count += classDF[i, wordPos]

	if(count == 0):
		print("COUNT ZERO ERROR A")

	return count

#Decr: this simply the TOTAL number of words across all documents for a class
def countOfAllWordsInClass(classDF):
	count=0
	
	classDF = classDF.to_numpy()
	for i in range(0, classDF.shape[0]):
		for j in range(0, classDF.shape[1]-1): #ADDED THE -1
			count += classDF[i, j]

	if(count == 0):
		print("COUNT ZERO ERROR B")
	return count


def probMatrixGen(myTrain, myName, words):
	c0, c1 = classSplitter(myTrain, myName)
	classes = [c0, c1]
	probMatrix = np.zeros((2, len(words))) #-1 beacause myTrain includes the label yo

	for c in range(0,2):
		v2 = countOfAllWordsInClass(classes[c]) + len(words) #+vectorsize to smooth
		for j in range(0, len(words)):
			v1 = countWordinClass(j, classes[c]) + 1 #+1 to smooth
			probMatrix[c, j] = v1/v2

	return probMatrix


#DO NOT SEND IN TESTPOINT WITH Y VALUE
def multiSumProbs(c, testpoint, probMatrix):

	totalProb=1
	testpoint = testpoint.to_numpy()
	for i in range(0, testpoint.shape[0]):
		f = testpoint[i]
		totalProb *= np.power(probMatrix[c, i], f)

	return totalProb


#DO NOT SEND IN TESTPOINT WITH Y VALUE
def multinomialProb(myTrain, myName, testpoint, words, probMatrix):
	
	prob_c0, prob_c1 = classProb(myTrain, myName) 
	multiSum0 = multiSumProbs(0, testpoint, probMatrix) 
	multiSum1 = multiSumProbs(1, testpoint, probMatrix) 
	prob_class0_testpoint = prob_c0 * multiSum0
	prob_class1_testpoint = prob_c1 * multiSum1 

	#print("prob(0) = ", prob_c0, " * ", multiSum0)
	#print("prob(1) = ", prob_c1, " * ", multiSum1)

	if prob_class0_testpoint >= prob_class1_testpoint:
		return 0
	else:
		return 1



