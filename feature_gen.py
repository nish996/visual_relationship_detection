import keras
import os
import sys
scriptpath = "/keras-retinanet/"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))
# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time


from utils.csv_gen import labelsToNames

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def getUnionBBox(aBB, bBB, ih, iw):
	margin = 10
	return [max(0, min(aBB[0], bBB[0]) - margin), \
		max(0, min(aBB[1], bBB[1]) - margin), \
		min(iw, max(aBB[2], bBB[2]) + margin), \
		min(ih, max(aBB[3], bBB[3]) + margin)]

def getAppr(im, bb):
	subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
	subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
	pixel_means = np.array([[[103.939, 116.779, 123.68]]])
	subim -= pixel_means
	subim = subim.transpose((2, 0, 1))
	return subim	

def visualize_pred(img_path,bboxes):
	# load image
	image = read_image_bgr(img_path)

	# copy to draw on
	draw = image.copy()
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
	#print(bboxes)
	labels_to_names = labelsToNames()
	for i in range(bboxes.shape[0]):		
		b = bboxes[i,0:4].astype(int)
		print(i,b) 	
		label = bboxes[i,4]
		score = bboxes[i,5]
		cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
		caption = "{} {:.3f}".format(labels_to_names[label], score)
		cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
		cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
	
	plt.figure(figsize=(15, 15))
	plt.axis('off')
	plt.imshow(draw)
	plt.show()	

	

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)

def load_model(model_name):

	# use this environment flag to change which GPU to use
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	# set the modified tf session as backend in keras
	keras.backend.tensorflow_backend.set_session(get_session())


	# adjust this to point to your downloaded/trained model
	model_path = os.path.join('snapshots', model_name)

	# load retinanet model
	model = keras.models.load_model(model_path, custom_objects=custom_objects)
	
	return model

def image_pred(model,img_path):


	#files = os.walk(img_dir).next()[2]
	
	#for file in files:
	#print os.path.join(img_dir, file)
	# load image
	image = read_image_bgr(os.path.join(img_path))

	# preprocess image for network
	image = preprocess_image(image)
	image, scale = resize_image(image)

	# process image
	_, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

	# compute predicted labels and scores
	predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
	scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

	# correct for image scale
	detections[0, :, :4] /= scale
	bb_score= np.empty((1,6))
	# store detections with confidence > 0.5
	flag = 0;		
	for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
		if score < 0.5:
   			 continue
		bb = detections[0, idx, :4].astype(int)	
		bb_temp =  np.append(bb,[label,score])
		if(flag == 0):
			flag =1
			bb_score[0,:]=np.expand_dims(bb_temp,axis=0)
		#print(np.expand_dims(bb_temp,axis=0).shape)
		else:
			bb_score = np.append(np.expand_dims(bb_temp,axis = 0),bb_score,axis=0)
	
	print(bb_score.shape)
	if(flag  == 0):
		return []
	else:
		return bb_score 

def main():
	#resnet50_coco_best_v2.0.1.h5
	model = load_model('resnet50_coco_best_v2.0.1.h5')
	bb_score = image_pred(model,'objdet/test_images/image3.jpg')
	visualize_pred('objdet/test_images/image3.jpg',bb_score)
