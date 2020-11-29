from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
import cPickle as pickle
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 32
EPOCHS = 64


class CaptionGenerator():

	def __init__(self):

		print("Initializing training data")

		self.max_caption_length = None
		self.vocab_size = 0
		self.index_word = {}
		self.word_index = {}
		self.num_samples = 0
		self.encoded_images = pickle.load(open("Flickr_Data/encoded_images.p", "rb"))

		df = pd.read_csv('Flickr_Data/Flickr_Dataset/flickr_8k_train_dataset.txt', delimiter='\t')

		captions = []
		for index, row in df.iterrows():
			captions.append(row[1][1])

		for text in captions:
			self.num_samples += len(text.split()) - 1

		captions_split = [caption.split() for caption in captions]

		unique = set()
		for words in captions_split:
			for word in words:
				unique.add(word)

		unique = list(unique)
		self.vocab_size = len(unique)

		for i, words in enumerate(unique):
			self.word_index[words] = i
			self.index_word[i] = words

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
		image_model.add(RepeatVector(self.max_caption_length))

		caption_model = Sequential()
		caption_model.add(Embedding(self.vocab_size, 256, input_length=self.max_caption_length))
		caption_model.add(LSTM(128, return_sequences=True))
		caption_model.add(TimeDistributed(Dense(128)))

		main_model = Sequential()
		main_model.add(Merge([image_model, caption_model], mode='concat'))
		main_model.add(LSTM(1000, return_sequences=False))
		main_model.add(Dense(self.vocab_size))
		main_model.add(Activation('softmax'))

		main_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		print("Model making finished")
		return main_model

	def generate_data(self, batch_size=32):

		partial_captions = []
		next_words = []
		images = []
		df = pd.read_csv('Flickr_Data/Flickr_Dataset/flickr_8k_train_dataset.txt', delimiter='\t')

		captions = []
		imgs = []
		for index, row in df.iterrows():
			captions.append(row[1][1])
			imgs.append(row[1][0])

		total_count = 0

		while 1:
			image_counter = -1

			for caption in captions:
				image_counter += 1

				if imgs[image_counter] in self.encoded_images:
					current_image = self.encoded_images[imgs[image_counter]]

					for i in range(len(caption.split()) - 1):
						total_count += 1

						partial = [self.word_index[word] for word in caption.split()[:i + 1]]
						partial_captions.append(partial)

						next = np.zeros(self.vocab_size)
						next[self.word_index[caption.split()[i + 1]]] = 1
						next_words.append(next)

						images.append(current_image)

						if total_count >= batch_size:
							next_words = np.asarray(next_words)
							images = np.asarray(images)
							partial_captions = sequence.pad_sequences(partial_captions, maxlen=self.max_caption_length,
																	padding='post')
							total_count = 0

							yield [[images, partial_captions], next_words]

							partial_captions = []
							next_words = []
							images = []


if __name__ == '__main__':

	print("Starting training")

	caption_generator = CaptionGenerator()
	model = caption_generator.make_model()

	counter = 0
	file = "Flickr_Data/weights-{epoch:02d}.hdf5"

	checkpoint = ModelCheckpoint(file, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	model.fit_generator(caption_generator.generate_data(batch_size=BATCH_SIZE),
						steps_per_epoch=caption_generator.num_samples / BATCH_SIZE,
						epochs=EPOCHS, verbose=2, callbacks=callbacks_list)
	try:
		model.save("Model/model.h5", overwrite=True)
		model.save_weights("Model/weights.h5", overwrite=True)
	except:
		print("Could not save model")

	print("Training finished, model and weights saved in folder \'Model\'")
