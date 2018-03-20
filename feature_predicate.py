import keras
import os
import sys
scriptpath = "/keras-retinanet/"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))
# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input




# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time


from utils.csv_gen import train_annotations,test_annotations,labelsToNames, imageToLabels, predicatesToNames

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


GLOVE_FILE_PATH = '/media/data/nishanth/datasets/glove_data/glove.840B.300d.txt'


'''
Function for generating union bounding box given subject and object bounding boxes
'''
def getUnionBBox(aBB, bBB, ih, iw):
	margin = 10
	return [max(0, min(aBB[0], bBB[0]) - margin), \
		max(0, min(aBB[1], bBB[1]) - margin), \
		min(iw, max(aBB[2], bBB[2]) + margin), \
		min(ih, max(aBB[3], bBB[3]) + margin)]


'''
Preprocessing before giving it it Appearance Subnet
input : image , union bounding box
output : new union bounding box of size (224, 224,3)
'''
def getAppr(im, bb):
	subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
	subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
	pixel_means = np.array([[[103.939, 116.779, 123.68]]])
	subim -= pixel_means
	subim = subim.transpose((2, 0, 1))
	return subim

'''
Network for appearance module ( used VGGNet)
input  : union bounding box
output  : Visual features of bounding box
'''
def apprSubnet(bbox):
	model = VGG16(weights='imagenet', include_top=False)
	img = image.load_img(bbox, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)
	return features


'''
Function for getting spatial features of a object
'''
def getSimpleSpatialFeatures(iw,ih,bbox):
	iarea = iw*ih
	spat_ft = np.empty(5)
	spat_ft[0] = bbox[0]/iw
	spat_ft[1] = bbox[1]/ih
	spat_ft[2] = bbox[2]/iw
	spat_ft[3] = bbox[3]/ih
	spat_ft[4] = (bbox[3]-bbox[1])*(bbox[2]-bbox[1])/iarea
	return spat_ft



def spatialSubnet(mask):
	model = Sequential()
#	model.add(Conv2D(96,(5,5),strides=(2,2) ,padding='same',activation='relu'),input_shape=)
#	model.add(Conv2D(128,(5,5),strides=(2,2) ,padding='same',activation='relu'))
#	model.add(Conv2D(64,(8,8),padding='same',activation='relu'))
	return model


def getDualMask(ih, iw, bb):
	rh = 32.0 / ih
	rw = 32.0 / iw
	x1 = max(0, int(math.floor(bb[0] * rw)))
	x2 = min(32, int(math.ceil(bb[2] * rw)))
	y1 = max(0, int(math.floor(bb[1] * rh)))
	y2 = min(32, int(math.ceil(bb[3] * rh)))
	mask = np.zeros((32, 32))
	mask[y1 : y2, x1 : x2] = 1
	assert(mask.sum() == (y2 - y1) * (x2 - x1))
	return mask

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(os.path.join(gloveFile),'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


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

def  visualize_bboxes(img_path,bboxes):
	# load image
	image = read_image_bgr(img_path)

	# copy to draw on
	draw = image.copy()
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
	#print(bboxes)
	labels_to_names = labelsToNames()
	for i in range(bboxes.shape[0]):
		b = bboxes[i,0:4].astype(int)
		label = bboxes[i,4]
		score = bboxes[i,5]
		cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
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


'''
returns predicted bounding boxes with label index and scire for that index
output format : [[x1,y1,x2,y2,label,score],...]
'''
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
		return bb_score  #format :[x1,y1,x2,y2,label,score]



#Removes redundant <sub,obj> pairs
def pair_filtering():
	return



def feature_extraction(img_file):
	im = cv2.imread(os.path.join('datasets/sg_dataset/sg_train_images',img_file)).astype(np.float32, copy=False)
	ih = im.shape[0]
	iw = im.shape[1]
	glove_model = loadGloveModel(GLOVE_FILE_PATH)
	labels_to_names = labelsToNames()
	gt_spo,gt_bb = imageToLabels(train_annotations, img_file)
	tot_features = []
	appr_features = []
	for i in range(gt_spo.shape[0]):
		sub_bbox = gt_bb[i,0,:4].astype(int)
		obj_bbox = gt_bb[i,1,:4].astype(int)
		sub_wordvec = glove_model[labels_to_names[gt_spo[i,0]]]
		obj_wordvec = glove_model[labels_to_names[gt_spo[i,2]]]
		sem_feat = np.concatenate((sub_wordvec,obj_wordvec),axis=0)
		union_bbox = getUnionBBox(sub_bbox, obj_bbox, ih, iw)
		appr_feat_temp = getAppr(im, union_bbox)
		#appr_feat = apprSubnet(appr_feat_temp)
		spat_feat = np.concatenate((getSimpleSpatialFeatures(ih, iw, sub_bbox), getSimpleSpatialFeatures(ih, iw, obj_bbox)),axis=0)
		tot_feat = np.concatenate((sem_feat,spat_feat), axis=0)
		appr_features.append(appr_feat_temp)
		tot_features.append(tot_feat)
	print(np.array(tot_features).shape,np.array(appr_features).shape)
	return np.array(tot_features)

def feature_extraction_tot():
	glove_model = loadGloveModel(GLOVE_FILE_PATH)
	labels_to_names = labelsToNames()
	tot_features = []
	tot_appr_features =[]
	tot_pred = []
	for img_file in train_annotations:
		im = cv2.imread(os.path.join('datasets/sg_dataset/sg_train_images',img_file)).astype(np.float32, copy=False)
		ih = im.shape[0]
		iw = im.shape[1]
		gt_spo,gt_bb = imageToLabels(train_annotations, img_file)
		for i in range(gt_spo.shape[0]):
			sub_bbox = gt_bb[i,0,:4].astype(int)
			obj_bbox = gt_bb[i,1,:4].astype(int)
			sub_wordvec = glove_model[labels_to_names[gt_spo[i,0]]]
			obj_wordvec = glove_model[labels_to_names[gt_spo[i,2]]]
			sem_feat = np.concatenate((sub_wordvec,obj_wordvec),axis=0)
			union_bbox = getUnionBBox(sub_bbox, obj_bbox, ih, iw)
			appr_feat_temp = getAppr(im, union_bbox)
			#appr_feat = apprSubnet(appr_feat_temp)
			spat_feat = np.concatenate((getSimpleSpatialFeatures(ih, iw, sub_bbox), getSimpleSpatialFeatures(ih, iw, obj_bbox)),axis=0)
			tot_feat = np.concatenate((sem_feat,spat_feat), axis=0)
			tot_appr_features.append(appr_feat_temp)
			tot_features.append(tot_feat)
			tot_pred.append(gt_spo[i,1])
	print(np.array(tot_features).shape,np.array(tot_appr_features).shape,np.array(tot_pred).shape)
	return np.array(tot_features),np.array(tot_appr_features),np.array(tot_pred)


def feature_model():
	features_x,pred_y =  feature_extraction_tot()
	inputs = Input(shape=(4096,))
	x = Dense(4096, activation='linear')(inputs)

	return



def main():
	#resnet50_coco_best_v2.0.1.h5
	#model = load_model('resnet50_csv_07.h5')
	#bb_score = image_pred(model,'objdet/test_images/image3.jpg')
	#visualize_pred('objdet/test_images/image3.jpg',bb_score)
	#glove_model = loadGloveModel('datasets/glove_data/glove.6B.300d.txt')
	#feat= feature_extraction('1602315_961e6acf72_b.jpg')
	feat = feature_extraction_tot()

if __name__ == "__main__":
		main()


