from string import punctuation
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re

import nltk
nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def clean(text):
	# split into words without punctuation
	# there's becomes "there" "s"
	tokens = tokenizer.tokenize(text)
	# convert to lower case
	tokens = [w.lower() for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	tokens = [word for word in tokens if not word in stop_words]
	# filter out words less than 1 character
	tokens = [word for word in tokens if len(word) > 1]
	return tokens


negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def cleanv2(text):
	# convert to lower case
	text = text.lower()
	# change don't to do not, doesn't to does not
	text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
	# remove remaining tokens that are not alphabetic
	text = re.sub("[^a-zA-Z]", " ", text)
	# tokenize
	tokens = tokenizer.tokenize(text)
	# filter out stop words
	tokens = [word for word in tokens if not word in stop_words]
	# filter out words less than 1 character
	tokens = [word for word in tokens if len(word) > 1]
	return tokens	

# Function to create or fetch vocabulary
def make_vocab(vocabFile,directory='./sample'): #floyd
	vocabExists = os.path.isfile(vocabFile)
	if vocabExists:
		# Read and return vocab
		print("Found vocab file {}").format(vocabFile)
		vocab = load_doc(vocabFile)
		vocab = vocab.split('\n')
		print("Vocabulary has {} words").format(len(vocab))
	else:
		print("Did not find vocab file {}").format(vocabFile)
		vocab = Counter()
		# walk through all files in the folder
		for path, subdirs, files in os.walk(directory):
			for filename in files:
				# create the full path of the file to open
				filepath =  os.path.join(path, filename)
				# load and clean the doc
				doc = load_doc(filepath)
				tokens = cleanv2(doc)
				vocab.update(tokens)

		print("Number of tokens before filtering freqeuncy of occurance: {}").format(len(vocab))
		vocab = [word for word,freq in vocab.most_common() if freq>2]
		print("Number of tokens occuring more than 2 times: {}").format(len(vocab))
		
		# Save the vocabulary file
		# convert lines to a single blob of text
		data = '\n'.join(vocab)
		# open file
		file = open(vocabFile, 'w+')
		# write text
		print("Saving vocabulary to {}").format(vocabFile)
		file.write(data)
		# close file
		file.close()
	return vocab


# change all files to BoW representation 
# based on frequency of words in each review
# load all docs in a directory into memory
def process_reviews(directory,vocab):
	reviews = list()
	sentiment = list()
	# walk through all files in the folder
	for path, subdirs, files in os.walk(directory):
			for filename in files:
				# create the full path of the file to open
				filepath =  os.path.join(path, filename)
				# load the doc
				doc = load_doc(filepath)
				# clean doc
				tokens = cleanv2(doc)
				# filter by vocab
				tokens = [word for word in tokens if word in vocab]
				review = ' '.join(tokens)
				# append review to reviews
				reviews.append(review)
				# Get the sentiment as well
				sentiment.append(1 if 'pos' in filepath else 0)

	return reviews,sentiment


def get_data(data_file,isTrain=True):
	dataset_type = 'train' if isTrain else 'test'
	if os.path.isfile(data_file):
		print("Found "+dataset_type+" File {}.").format(data_file)
		data = pickle.load(open(data_file, 'rb'))
		X,y = zip(*data)


	else:
		print("Did not find "+dataset_type+" file.")
		vocab = make_vocab('./data/polar.vocab','./dataset/train')#floyd
		print("Saved Vocabulary")
		print("processing reviews...")
		X,y = process_reviews('./dataset/'+dataset_type,vocab) #floyd
		data = zip(np.array(X),np.array(y))
		np.array(data).dump('./output/'+dataset_type+'.data') #floyd

	X = np.array(X)
	y = np.array(y)
	print("Found {} samples for "+dataset_type).format(X.shape[0])

	return X,y

def tokenize(X_train,X_test):
	from keras.preprocessing.text import Tokenizer
	
	keras_tokenizer =  Tokenizer()
	keras_tokenizer.fit_on_texts(X_train)
	X_train = keras_tokenizer.texts_to_matrix(X_train, mode='binary')
	X_test = keras_tokenizer.texts_to_matrix(X_test, mode='binary')
	return X_train,X_test


def get_model(input_shape):

	from keras.models import Model
	from keras.layers import Input, Dense, Dropout

	input_layer = Input(shape=(input_shape,))
	x = Dense(50,activation='relu')(input_layer)
	x = Dropout(0.5)(x)
	# x = Dense(128,activation='relu')(x)
	# x = Dropout(0.5)(x)
	output_layer = Dense(1,activation='sigmoid')(x)
	model = Model(inputs=input_layer, outputs=output_layer)

	model.summary()
	return model


def train_model(model, X,y,epochs=10):
	from keras.callbacks import ModelCheckpoint
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=30)
	save_model = ModelCheckpoint('./output/weights.hdf5', monitor='val_loss',save_best_only=True) #floyd
	hist = model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=2, callbacks=[save_model],validation_data=(X_val,y_val),shuffle=True )
	return hist

def test_model(model, X_test, y_test):
	print("Testing model on {}").format(X_test.shape[0])
	model.load_weights('./output/weights.hdf5') #floyd
	loss, acc = model.evaluate(X_test, y_test, verbose=0)
	print('Test Accuracy: %f' % (acc*100))

def plot_loss(hist):
	import matplotlib.pyplot as plt
	loss = hist.history['loss'] #np.loadtxt('my_cnn_model_loss.csv')
	val_loss = hist.history['val_loss'] #np.loadtxt('my_cnn_model_val_loss.csv')

	plt.plot(loss, linewidth=3, label='train')
	plt.plot(val_loss, linewidth=3, label='valid')
	plt.grid()
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('loss')
	#plt.ylim(1e-3, 1e-2)
	plt.yscale('log')
	plt.show()


if __name__ == '__main__':
	train_file = './data/train.data' #floyd
	test_file = './data/test.data' #floyd

	X,y = get_data(train_file,True)
	X_test,y_test = get_data(test_file,False)
	
	X,X_test = tokenize(X,X_test)

	model = get_model(input_shape=X.shape[1])
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	hist = train_model(model,X,y,epochs=30)
	test_model(model,X_test,y_test)
	plot_loss(hist)