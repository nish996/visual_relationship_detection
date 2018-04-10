import json
import numpy as np
from pprint import pprint
import csv
import subprocess

from csv_gen import train_annotations,test_annotations

# gt file format: [ gt_label, gt_box ]
# 	gt_label: list [ gt_label(image_i) for image_i in images ]
# 		gt_label(image_i): numpy.array of size: num_instance x 3
# 			instance: [ label_s, label_r, label_o ]
# 	gt_box: list [ gt_box(image_i) for image_i in images ]
#		gt_box(image_i): numpy.array of size: num_instance x 2 x 4
#			instance: [ [x1_s, y1_s, x2_s, y2_s], 
#				    [x1_o, y1_o, x2_o, y2_o]]

def loadEvalData(train_flag):
	gt_label = []
	gt_box = []
	if(train_flag):
		l  = train_annotations
	else:
		l = test_annotations
	for img in l:
		gt_label_tmp = []
		gt_bbox_tmp = []
		for spo in l[img]:
			spo_list = np.empty(3, dtype=int)
			spo_list[0] = spo['subject']['category']
			spo_list[1] = spo['predicate']
			spo_list[2] = spo['object']['category']
			gt_label_tmp.append(spo_list)
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
			gt_bbox_tmp.append(gt_bbox)
		gt_label.append(np.array(gt_label_tmp))
		gt_box.append(np.array(gt_bbox_tmp))
	return np.array(gt_label),np.array(gt_box)


# det file format: [ det_label, det_box ]
# 	det_label: list [ det_label(image_i) for image_i in images ]
# 		det_label(image_i): numpy.array of size: num_instance x 6
# 			instance: [ prob_s, prob_r, prob_o, label_s, label_r, label_o ]
# 	det_box: list [ det_box(image_i) for image_i in images ]
#		det_box(image_i): numpy.array of size: num_instance x 2 x 4
#			instance: [ [x1_s, y1_s, x2_s, y2_s], 
#				    [x1_o, y1_o, x2_o, y2_o]]

def main():
	x,y = loadEvalData(False)
	print(x.shape,y)


if __name__ =="__main__":
    main()
