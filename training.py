from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten

VOCAB_SIZE = 1000
MAX_CAPTION_LEN = 300

def make_model():

	print("Making model")

	image_model = Sequential()
	image_model.add(Dense(128, input_dim = 4096, activation='relu'))

	caption_model = Sequential()
	caption_model.add(Embedding(VOCAB_SIZE, 256, input_length=MAX_CAPTION_LEN))
	caption_model.add(LSTM(128,return_sequences=True))
	caption_model.add(TimeDistributed(Dense(128)))

	main_model = Sequential()
	main_model.add(Merge([image_model, caption_model], mode='concat'))
	main_model.add(LSTM(1000, return_sequences=False))
	main_model.add(Dense(VOCAB_SIZE))
	main_model.add(Activation('softmax'))

	main_model.compile(loss='categorical_crossentropy', optimizer='adamoptimizer', metrics=['accuracy'])

	print("Model making finished")
	return main_model