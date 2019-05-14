import pandas as pd
import numpy as np
from scipy.optimize import fmin_tnc

#Decr: This function trains loops through the costFunction, constantly measuring its changes using
#      the measure from the gradient() function we made. It then adjusts the weights using the data
#      from the cost function. With the help of the fmin_tnc function, this process will
#      continue until we reach an extremely low value is produced by the cost function. Therefore
#      although we have set how to adjust the weights and how to measure there progress the min_tnc()
#      function takes care of the simple process of actually adjusting myTheta and keep track of how
#      many iterations to produce. 
# Input: myX    : coefficents
#		 myY    : vector holding real class values. (Should be flattened btw)
def LogiRegrTrain(myX, myY):
	myY = myY.to_numpy()
	myX = myX.to_numpy()

	myY = myY.flatten()
	
	myX = np.c_[np.ones((myX.shape[0], 1)), myX]
	myTheta = np.zeros((myX.shape[1], 1))

	best_weights = fmin_tnc(func=costFunction, x0=myTheta, fprime=gradient, args=(myX, myY), disp=0)

	results = best_weights[0]

	return results


def LogiRegrPredict(myX, weights):
	myX = myX.to_numpy()
	myX = np.c_[np.ones((myX.shape[0], 1)), myX]

	return sigmoid(myX, weights)

#Descr: This is the sigmoid function, the core of the Logistical Regression
#       alogrithm. It returns a values between 0 and 1, which represents the 
#       probability of a given datapoint being a certian class. 
#Input: myX    : this this the coefficents, it can also be a bunch of coefficents 
#			     a matrix if you want to caluculate a bunch of probabilities at once.
#       myTheta: these are the weights of the coeffiecents, it can also be the weights
#                for a bunch of coeffiecents if you want to calculate a bunch at once
#Output: value between 0 and 1.
def sigmoid(myX, myTheta):
	weightsWithCoefs = np.dot(myX, myTheta)
	probability = 1 / (1 + np.exp(- weightsWithCoefs))
	return probability

#Desrc: This costFunction does the following:
#           I.   Get the sigmoid probability for each datapoint (stored in vector)
#           II.  Do I. for normal case and inverse case
#           III. Input these vectors into the traditional cost function for Linear
#				 Regression
#           IV.  What will result is a matrix of shape = (num datapoints, numdatapoints)
#            V.  All these values are then summed together and divided by the number of datapoints
#
# What does it do?: It essentially adjusts the weights based on their predicted results and the actual results.
# 
# Input: myX    : coefficents
#		 myTheta: weights
#		 myY    : vector holding real class values. (Should be flattened btw)
def costFunction(myTheta, myX, myY):
	size = myX.shape[0]

	#TESTING
	# print("myX.shape: ", myX.shape)
	# print("myTheta.shape: ", myTheta.shape)
	# print("sigmoid(myX, myTheta): ", sigmoid(myX, myTheta))

	cost = -(1/size) * np.sum(  (myY) * np.log(sigmoid(myX, myTheta)) + (1-myY) * np.log(1 - sigmoid(myX, myTheta))  )
	return cost

#Descr: The gradient is meant to measure the cost functions changes in
#       order to better understand if it is imporving or getting worse.
#		Because we are measuring the differnce of the cost function, the
#       the gradient function is simply the deriviative of the cost
#       function. 
# Input: myX    : coefficents
#		 myTheta: weights
#		 myY    : vector holding real class values. (Should be flattened btw)
def gradient(myTheta, myX, myY):
	return derivativeCostFunction(myX, myTheta, myY)


# Decr: Exactly what the name suggests, the partical derivative
#       of the cost function.
# Input: myX    : coefficents
#		 myTheta: weights
#		 myY    : vector holding real class values. (Should be flattened btw)
def derivativeCostFunction(myX, myTheta, myY):
	size = myX.shape[0] #aka, how many rows in X, (aka, num datapoints)
	measure = (1/size) * np.dot(myX.T, sigmoid(myX, myTheta)- myY)
	return measure
