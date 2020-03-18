import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

classifier = ResNet50(weights='imagenet')

def classify(path=None):
	if path:
		img = image.img_to_array(image.load_img(path, target_size=(224, 224)))
		img = preprocess_input(np.expand_dims(img, axis=0))
		predictions = np.squeeze(decode_predictions(classifier.predict(img)), axis=0)
		for label in predictions:
			print("prediction: {} with confidence: {}\n".format(label[1], label[2]))

classify('test.jpg')