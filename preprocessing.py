import cPickle as pickle
from keras.preprocessing import image
from keras.applications import vgg16
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input
import string

IH = 224
IW = 224
IZ = 3

INPUT_SHAPE = (IH, IW, IZ)

def load_image(path):
    img = image.load_img(path, target_size=(IH,IW))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)

def load_encoding_model():
	model = vgg16.VGG16(weights='imagenet', include_top=True, input_shape = INPUT_SHAPE)
	return model

def get_encoding(model, img):
	image = load_image('Flickr_Data/Flickr_Images/'+str(img))
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[1])
	return pred

table = string.maketrans('', '')

def clean_caption(capt):
	global table
	capt = capt.split()
	capt = [word.lower() for word in capt]
	capt = [w.translate(table, string.punctuation) for w in capt]
	capt = [word for word in capt if len(word)>1]
	capt = [word for word in capt if word.isalpha()]
	capt =  ' '.join(capt)
	return capt

def prepare_dataset():
	train_images_file = open('Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt','rb')
	train_images = train_images_file.read().strip().split('\n')

	test_images_file = open('Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt','rb')
	test_images = test_images_file.read().strip().split('\n')

	test_images_file.close()
	train_images_file.close()

	# Saving files with corresponding captions
	train_dataset_file = open('Flickr_Data/Flickr_Dataset/flickr_8k_train_dataset.txt','wb')
	test_dataset_file = open('Flickr_Data/Flickr_Dataset/flickr_8k_test_dataset.txt','wb')

	captions_file = open('Flickr_Data/Flickr_TextData/Flickr8k.token.txt', 'rb')
	captions = captions_file.read().strip().split('\n')
	data = {}
	for caption_row in captions:
		caption_row = caption_row.split("\t")
		caption_row[0] = caption_row[0][:len(caption_row[0])-2]
		try:
			data[caption_row[0]].append(caption_row[1])
		except:
			data[caption_row[0]] = [caption_row[1]]
	captions_file.close()

	encoded_images = {}
	encoding_model = load_encoding_model()

	for img in train_images:
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in data[img]:			
			capt = clean_caption(capt)
			caption = "<start> "+capt+" <end>"
			train_dataset_file.write(img+"\t"+caption+"\n")
			train_dataset_file.flush()
	train_dataset_file.close()

	for img in test_images:
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in data[img]:
			capt = clean_caption(capt)
			caption = "<start> "+capt+" <end>"
			test_dataset_file.write(img+"\t"+caption+"\n")
			test_dataset_file.flush()
	test_dataset_file.close()
	with open( "Flickr_Data/encoded_images.p", "wb" ) as pickle_f:
		pickle.dump( encoded_images, pickle_f )  

if __name__ == '__main__':
	prepare_dataset()
	print('Preprocessing Complete!\n')