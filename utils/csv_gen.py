import json
import numpy as np
from pprint import pprint
import csv
import subprocess


OBJECT_JSON_PATH='/media/data/nishanth/datasets/json_dataset/objects.json'
PREDICATE_JSON_PATH='/media/data/nishanth/datasets/json_dataset/predicates.json'
TRAIN_JSON_PATH='/media/data/nishanth/datasets/json_dataset/annotations_train.json'
#'/media/data/nishanth/datasets/sg_dataset/sg_train_annotations.json'
TEST_JSON_PATH='/media/data/nishanth/datasets/json_dataset/annotations_test.json'
#'/media/data/nishanth/datasets/json_dataset/annotations_test.json'
APPEND_TRAIN_PATH='sg_dataset/sg_train_images/'
APPEND_TEST_PATH='sg_dataset/sg_test_images/'

# format: path,x1,y1,x2,y2,class_name

objects=json.load(open(OBJECT_JSON_PATH))
predicates=json.load(open(PREDICATE_JSON_PATH))
train_annotations=json.load(open(TRAIN_JSON_PATH))
test_annotations=json.load(open(TEST_JSON_PATH))


def jsonToCSV(inputData,filename,train_flag):
	inFile = open(filename,'w')
	for image in inputData:
		for spo in inputData[image]:
			x = []
			if(train_flag):
				inFile.write(APPEND_TRAIN_PATH)
			else :
				inFile.write(APPEND_TEST_PATH)
			inFile.write(image)
			inFile.write(',')
			for i in spo['object']['bbox']:
				x.append(str(i))
			inFile.write(x[2]+','+x[0]+','+x[3]+','+x[1])
			inFile.write(',')
			inFile.write(objects[spo['object']['category']])
			inFile.write('\n')
			x=[]
			if(train_flag):
				inFile.write(APPEND_TRAIN_PATH)
			else :
				inFile.write(APPEND_TEST_PATH)
			inFile.write(image)
			inFile.write(',')
			for i in spo['subject']['bbox']:
				x.append(str(i))
			inFile.write(x[2]+','+x[0]+','+x[3]+','+x[1])
			inFile.write(',')
			inFile.write(objects[spo['subject']['category']])
			inFile.write('\n')

def objectjsonToCSV(inputData,filename):
	inFile = open(filename,'w')
	for i in range(len(inputData)):
		inFile.write(inputData[i])
		inFile.write(',')
		inFile.write(str(i))
		inFile.write('\n')


def labelsToNames():
	dict = {}
	for i in range(len(objects)):
		if(objects[i] == 'trash can'):
			objects[i] = 'trashcan'
		if(objects[i] == 'traffic light'):
			objects[i] = 'traffic-light'
		dict[i]=objects[i]
	return dict


def predicatesToNames():
	dict = []
	for i in range(len(predicates)):
		x  = predicates[i].split(" ")
		dict.append(x)
	return dict

def imageToLabels(inputData,image_file):
	img_list = inputData[image_file]
	gt_spo =[]
	gt_bb = []
	for spo in img_list:
		spo_tuple = np.empty(3,dtype=int)
		spo_tuple[0]= spo['subject']['category']
		spo_tuple[1] = spo['predicate']
		spo_tuple[2] = spo['object']['category']
		gt_spo.append(spo_tuple)
		gt_bbox = np.empty([2,4],dtype=int)
		itr=[2,0,3,1]
		itr1 = 0
		for i in itr:
			gt_bbox[0,itr1] = spo['subject']['bbox'][i]
			itr1 = itr1+1
		itr1 = 0
		for i in itr:
			gt_bbox[1,itr1] = spo['object']['bbox'][i]
			itr1 = itr1+1
		gt_bb.append(gt_bbox)
	gt_spo_out = np.array(gt_spo)
	gt_bb_out = np.array(gt_bb)
#	print(gt_bb_out.shape,gt_spo_out.shape)
	return gt_spo_out,gt_bb_out


def main():
	jsonToCSV(test_annotations,'test_temp.csv',0)
	jsonToCSV(train_annotations,'train_temp.csv',1)
	subprocess.call("./dup_removal.sh",shell=True)
	objectjsonToCSV(objects,'/media/data/nishanth/datasets/objects.csv')



#not needed for obj detection
#objectjsonToCSV(predicates,'predicates.csv')

#pprint(objects)

