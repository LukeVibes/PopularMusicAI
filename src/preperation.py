import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
stopwords = set(stopwords.words('english'))
stopwords.add('')
stopwords.add('im')
stopwords.add('x2')
stopwords.add('x3')
stopwords.add('x4')





#Import Datasets
def mergeData():
	songLyrics  = pd.read_csv('C:/Users/iamro/Documents/School/AI Class/Project/data/raw/songdata.csv')
	spotifyInfo = pd.read_csv('C:/Users/iamro/Documents/School/AI Class/Project/data/raw/SpotifyFeatures.csv')


	mergeredSet = pd.merge(songLyrics, spotifyInfo, left_on=['artist', 'song'], right_on=['artist_name', 'track_name'], how='inner')

	mergeredSet.to_csv('C:/Users/iamro/Documents/School/AI Class/Project/data/merged.csv')

	return mergeredSet


def BofW_Giver(ds):

	#Create global wordset + first round of cleaning data
	wordSet = totalWordSet(ds)

	#Second round of cleaning data
	minMaxBOWTrimmer(ds, wordSet)

	#Create each songs(row) bag-of-words
	bow = bagOfWordsGenerator2(ds, wordSet)

	return bow


def BofW(ds):

	#Create global wordset + first round of cleaning data
	wordSet = totalWordSet(ds)

	#Second round of cleaning data
	minMaxBOWTrimmer(ds, wordSet)

	#Create each songs(row) bag-of-words
	bagOfWordsGenerator(ds, wordSet)

	ds.to_csv('C:/Users/iamro/Documents/School/AI Class/Project/data/refined.csv')
	print(ds.ix[2, 'bow'])

	return ds


def punctuationKiller(word):
	newword = ''.join(c for c in word if c not in string.punctuation)
	return newword

#Desrc: Remvoes tokens that appear in less than 10 songs, 
#       removes tokens that appear in more than 70% of songs.
def minMaxBOWTrimmer(ds, ws):
	print("trimming wordSet...          ", end =" ")

	zeros = [0] * len(ws)
	indoc = dict(zip(ws, zeros))
	
	wordsWeCounted = []

	numberOfDocs = len(ds.index)

	#create bow and indoc
	for index, row in ds.iterrows():
		lyrics = row['text'].strip()
		lyrics.translate(str.maketrans('','', string.punctuation))
		lyrics = lyrics.lower()
		lyrics = lyrics.replace('\n', '')
		lyrics = lyrics.split(' ')
		lyrics = [word for word in lyrics if word not in stopwords]
		lyrics = [PorterStemmer().stem(word) for word in lyrics]
		lyrics = [punctuationKiller(w) for w in lyrics]

		wordsWeCounted = []
		for word in lyrics:
			if (word not in wordsWeCounted):
				indoc[word] += 1
				wordsWeCounted.append(word)

	#remove words with counts 
	wordsToRemove = []
	
	for word in indoc:
		if indoc[word] <= 200:
			wordsToRemove.append(word)
			
		elif (indoc[word] / numberOfDocs) >= 0.7:
			wordsToRemove.append(word)
		

	for killWord in wordsToRemove:
	 	ws.remove(killWord)

	print("[DONE]")
	print("\tWordset Size: ", len(ws))


def totalWordSet(ds):

	wordSet = []
	wordSet = set(wordSet)

	print("generating wordSet...        ", end =" ")
	for index, row in ds.iterrows():
		lyrics = row['text'].strip()
		lyrics.translate(str.maketrans('','', string.punctuation))
		lyrics = lyrics.lower()
		lyrics = lyrics.replace('\n', '')
		lyrics = lyrics.split(' ')
		lyrics = [word for word in lyrics if word not in stopwords]
		lyrics = [PorterStemmer().stem(word) for word in lyrics]
		lyrics = [punctuationKiller(w) for w in lyrics]
		

		wordSet = wordSet.union(set(lyrics))


		# if(index < 2):
		# 	print(trimmedLyrics)

	print("[DONE]")
	print("\tWordset Size: ", len(wordSet))
	return wordSet


def tfCalculator(bow):

	tfDict = {}
	count = len(bow)
	for word in bow:
		tfDict[word] = count/float(bowCount)


def bagOfWordsGenerator(ds, ws):
	print("generating each songs BofW...", end =" ")
	ds["bow"] = np.zeros

	for index, row in ds.iterrows():
		print(index)
		bow = {}
		zeros = [0] * len(ws)
		bow = dict(zip(ws, zeros))

		lyrics = row['text'].strip()
		lyrics.translate(str.maketrans('','', string.punctuation))
		lyrics = lyrics.lower()
		lyrics = lyrics.replace('\n', '')
		lyrics = lyrics.split(' ')
		lyrics = [word for word in lyrics if word not in stopwords]
		lyrics = [PorterStemmer().stem(word) for word in lyrics]
		lyrics = [punctuationKiller(w) for w in lyrics]

		for word in lyrics:
			if (word in bow):
				bow[word] += 1


		ds.at[index, 'bow'] = str(bow)
	print("[DONE]")

def bagOfWordsGenerator2(ds, ws):
	print("generating each songs BofW...", end =" ")
	ds["bow"] = np.zeros

	simpleDF = [[0]*len(ws)]

	for index, row in ds.iterrows():
		bow = {}
		zeros = [0] * len(ws)
		bow = dict(zip(ws, zeros))

		lyrics = row['text'].strip()
		lyrics.translate(str.maketrans('','', string.punctuation))
		lyrics = lyrics.lower()
		lyrics = lyrics.replace('\n', '')
		lyrics = lyrics.split(' ')
		lyrics = [word for word in lyrics if word not in stopwords]
		lyrics = [PorterStemmer().stem(word) for word in lyrics]
		lyrics = [punctuationKiller(w) for w in lyrics]

		for word in lyrics:
			if (word in bow):
				bow[word] += 1

		simpleDF.append(list(bow.values()))

	
	print("[DONE]")

	newDf = pd.DataFrame(simpleDF, columns=list(bow.keys()))

	return newDf



def autoLabelEncoding(ds):
	labelEncoding(ds, 'key')
	labelEncoding(ds, 'mode')
	labelEncoding(ds, 'time_signature')
	labelEncoding(ds, 'genre')

def labelEncoding(ds, columnName):
	ds[columnName] = ds[columnName].astype('category').cat.codes
