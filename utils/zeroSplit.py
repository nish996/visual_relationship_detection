import json
import numpy as np
from pprint import pprint
import csv
import subprocess

from utils.csv_gen import train_annotations,test_annotations


def loadTrainDict():
    train_dict ={}
    count  =0
    for image in train_annotations:
        for spo in train_annotations[image]:
            spo_list = np.empty(3, dtype=int)
            spo_list[0]= spo['subject']['category']
            spo_list[1] = spo['predicate']
            spo_list[2] = spo['object']['category']
            spo_tuple = tuple(spo_list)
            count = count +1
            if (bool(train_dict.get(spo_tuple)) == False):
                train_dict[spo_tuple] = 1
    #print(count,len(train_dict))
    return train_dict


def getZeroSplit(train_dict):
    zero_dict= {}
    count  =0
    ncount = 0
    gt_testz_spo =[]
    gt_testz_bb = []
    gt_test_spo =[]
    gt_test_bb = []
    for image in test_annotations:
        for spo in test_annotations[image]:
            spo_list = np.empty(3, dtype=int)
            spo_list[0] = spo['subject']['category']
            spo_list[1] = spo['predicate']
            spo_list[2] = spo['object']['category']
            spo_tuple = tuple(spo_list)
            #if (bool(zero_dict.get(spo_tuple))):
            #   count = count +1;
            if (bool(train_dict.get(spo_tuple)) == False):
                zero_dict[spo_tuple] = 1
                #print(spo_tuple,image)
                count =count +1
            if (bool(train_dict.get(spo_tuple))):
                ncount  = ncount +1
    #print(count,ncount)
    return zero_dict


def zeroTestImageToLabels(image_file):
    train_dict = loadTrainDict()
    zero_dict = getZeroSplit(train_dict)
    img_list = test_annotations[image_file]
    gt_spo =[]
    gt_bb = []
    for spo in img_list:
        spo_list = np.empty(3,dtype=int)
        spo_list[0]= spo['subject']['category']
        spo_list[1] = spo['predicate']
        spo_list[2] = spo['object']['category']
        spo_tuple = tuple(spo_list)
        if (bool(zero_dict.get(spo_tuple))):
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
#   print(gt_bb_out.shape,gt_spo_out.shape)
    return gt_spo_out,gt_bb_out



def nonZeroTestImageToLabels(image_file):
    train_dict = loadTrainDict()
    zero_dict = getZeroSplit(train_dict)
    img_list = test_annotations[image_file]
    gt_spo =[]
    gt_bb = []
    for spo in img_list:
        spo_list = np.empty(3,dtype=int)
        spo_list[0]= spo['subject']['category']
        spo_list[1] = spo['predicate']
        spo_list[2] = spo['object']['category']
        spo_tuple = tuple(spo_list)
        if(not(bool(zero_dict.get(spo_tuple)))):
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
#   print(gt_bb_out.shape,gt_spo_out.shape)
    return gt_spo_out,gt_bb_out

def main():
    train_dict = loadTrainDict()
    zero_dict = getZeroSplit(train_dict)
    print(len(zero_dict))

if __name__ =="__main__":
    main()


