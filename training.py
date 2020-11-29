from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
import cPickle as pickle
import numpy as np
import pandas as pd


class CaptionGenerator():

	def __init__(self):

		print("Initializing training data")

		self.max_caption_length = None
		self.vocab_size = 0
		self.index_word = {}
		self.word_index = {}
		self.num_samples = 0
		self.encoded_images = pickle.load(open("encoded_images.p", "rb"))

		df = pd.read_csv('Flickr_Data/Flickr_Dataset/flickr_8k_train_dataset.txt', delimiter='\t')

		captions = []
		for index, row in df.iterrows():
			captions.append(row[1][1])

		for text in captions:
			self.num_samples += len(text.split()) - 1

		words = [caption.split() for caption in captions]

		unique = set()
		for word in words:
			unique.add(word)

		unique = list(unique)
		self.vocab_size = len(unique)

		for i, word in enumerate(unique):
			self.word_index[word] = i
			self.index_word[i] = word

		max_caption_length = 0
		for caption in captions:
			if (len(caption.split()) > max_caption_length):
				max_caption_length = len(caption.split())
		self.max_caption_length = max_caption_length

		print("Training initialization finished")


	def make_model(self):
		print("Making model")

		image_model = Sequential()
		image_model.add(Dense(128, input_dim=4096, activation='relu'))

		caption_model = Sequential()
		caption_model.add(Embedding(self.vocab_size, 256, input_length=self.max_caption_length))
		caption_model.add(LSTM(128, return_sequences=True))
		caption_model.add(TimeDistributed(Dense(128)))

		main_model = Sequential()
		main_model.add(Merge([image_model, caption_model], mode='concat'))
		main_model.add(LSTM(1000, return_sequences=False))
		main_model.add(Dense(self.vocab_size))
		main_model.add(Activation('softmax'))

		main_model.compile(loss='categorical_crossentropy', optimizer='adamoptimizer', metrics=['accuracy'])

		print("Model making finished")
		return main_model
