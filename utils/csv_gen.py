import json
import numpy
from pprint import pprint
import csv
import subprocess


OBJECT_JSON_PATH='/media/data/nishanth/dataset/json_dataset/objects.json'
PREDICATE_JSON_PATH='/media/data/nishanth/dataset/json_dataset/predicates.json'
TRAIN_JSON_PATH='/media/data/nishanth/dataset/json_dataset/annotations_train.json'
TEST_JSON_PATH='/media/data/nishanth/dataset/json_dataset/annotations_test.json'
APPEND_TRAIN_PATH='sg_dataset/sg_train_images/'
APPEND_TEST_PATH='sg_dataset/sg_test_images/'

# format: path,x1,x2,y1,y2,class_name

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
		dict[i]=objects[i]	
	return dict

def main():
	jsonToCSV(test_annotations,'test_temp.csv',0)
	jsonToCSV(train_annotations,'train_temp.csv',1)
	subprocess.call("./dup_removal.sh",shell=True)
	objectjsonToCSV(objects,'/media/data/nishanth/dataset/objects.csv')

#not needed for obj detection
#objectjsonToCSV(predicates,'predicates.csv')

#pprint(objects)
 
