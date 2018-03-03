import json
import numpy
from pprint import pprint
import csv
import subprocess


OBJECT_JSON_PATH='json_dataset/objects.json'
PREDICATE_JSON_PATH='json_dataset/predicates.json'
TRAIN_JSON_PATH='json_dataset/annotations_train.json'
TEST_JSON_PATH='json_dataset/annotations_test.json'

# format: path,x1,x2,y1,y2,class_name

objects=json.load(open(OBJECT_JSON_PATH))
predicates=json.load(open(PREDICATE_JSON_PATH))
train_annotations=json.load(open(TRAIN_JSON_PATH))
test_annotations=json.load(open(TEST_JSON_PATH))


def jsonToCSV(inputData,filename):
	inFile = open(filename,'w')
	for image in inputData:
		for spo in inputData[image]:
			inFile.write(image)
			inFile.write(',')
			for i in spo['object']['bbox']:
				inFile.write(str(i))
				inFile.write(',')
			inFile.write(objects[spo['object']['category']])
			inFile.write('\n')
			inFile.write(image)
			inFile.write(',')			
			for i in spo['subject']['bbox']:
				inFile.write(str(i))
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


jsonToCSV(test_annotations,'test_temp.csv')
jsonToCSV(train_annotations,'train_temp.csv')
subprocess.call("./dup_removal.sh",shell=True)
objectjsonToCSV(objects,'objects.csv')

#not needed for obj detection
#objectjsonToCSV(predicates,'predicates.csv')

#pprint(objects)
 
