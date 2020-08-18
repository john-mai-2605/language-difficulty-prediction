import pandas as pd
import re
import numpy as np
import cmudict
import textdistance
import sklearn 
import pickle
import nltk
# Load vocab
# file = open('vocab', 'rb')
# vocab = pickle.load(file)
unigrams_df = pd.read_csv('unigram_freq.csv', index_col = 'word')
pron_dict = cmudict.dict()

# The class for Word
class Word:
	# Constructor
	def __init__(self, token):
		self.token = re.sub(r'\W+', '', str(token)).lower()
	# Basic features
	def length(self):
		if not self.token:
			return 0
		return len(self.token)

	def frequency(self):
		try:
			return np.log(unigrams_df.loc[self.token]['count'])
		except:
			return 0
	def orthogonal_depth(self):
		try:
			phoneme = pron_dict[self.token]
			return self.length()/len(phoneme[0])
		except:
			return 1
	def phonetic_density(self):
		cons = [0 if char in {'u', 'e', 'o', 'a', 'i'} else 1 for char in self.token]
		num_cons = sum(cons)
		if sum(cons) == 0:
			return 1
		return self.length()/num_cons - 1

# class Vocab:
# 	# Constructor
# 	def __init__(self, token):

# Test basic features

# word = Word("this, ")
# print(word.length())
# print(word.frequency())
# print(word.orthogonal_depth())
# print(word.phonetic_density())

# The class for Text (text fragments)
class Text:
	# Constructor
	def __init__(self, text):
		self.text = text
		self.words = [Word(token) for token in text.strip().split(' ')]
		self.tokens = [Word(token).token for token in text.strip().split(' ')]
		self.pos_tags = [tag[:2] for (word, tag) in nltk.pos_tag(nltk.word_tokenize(text))]
		self.noun = 0
		self.verb = 0
		self.adj = 0
		self.adv = 0
		for pos in self.pos_tags:
			if pos == 'NN':
				self.noun += 1
			if pos == 'VB':
				self.verb += 1
			if pos == 'JJ':
				self.adj += 1
			if pos == 'RB':
				self.adv += 1
		self.length = np.asarray([word.length() for word in self.words])
		self.frequency = np.asarray([word.frequency() for word in self.words])
		self.orthogonal_depth = np.asarray([word.orthogonal_depth() for word in self.words])
		self.phonetic_density = np.asarray([word.phonetic_density() for word in self.words])
		self.pron_ambiguity = np.asarray([self.distance(word) for word in self.words])
	def distance(self, word):
		return np.max([0 if word.token == another_word.token else textdistance.levenshtein.normalized_similarity(word.token, another_word.token) for another_word in self.words])

	def extract_features(self):
		features = [len(self.words), 
			self.noun/len(self.words), 
			self.verb/len(self.words), 
			self.adj/len(self.words), 
			self.adv/len(self.words)]
		for prop in [self.length, self.frequency, self.orthogonal_depth, self.phonetic_density, self.pron_ambiguity]:
			features.append(np.average(prop))
			features.append(np.max(prop))
			features.append(np.min(prop))
		return np.asarray(features)

# Test text

# text = Text(" I don't know where a headset ties into patriot")
# print(text.orthogonal_depth)
# print(text.extract_features())

# Class Dataset
class Preprocess:
	def __init__(self, path):
		self.data = pd.read_csv(path)
		self.X = np.asarray([Text(text) for text in self.data['text']])

	def transform(self):
		features = [Text(text).extract_features() for text in self.data['text']]
		self.data[[
			'length', 'noun', 'verb', 'adj', 'adv',
			'aveLength', 'maxLength', 'minLength', 
			'aveFreq', 'maxFreq', 'minFreq', 
			'aveDepth', 'maxDepth', 'minDepth', 
			'aveDensity', 'minDensity', 'maxDensity',
			'aveAmbiguity', 'minAmbiguity', 'maxAmbiguity'
			]] = features

		self.data['speed'] = np.divide(self.data['length'], self.data['elapse_time'])

		self.data.to_csv('../processed_data.csv')

data = Preprocess('../big_data.csv')
data.transform()
print("Finish pre-processing data")

