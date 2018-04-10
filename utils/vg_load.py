import numpy as np
import _pickle as pk

'''
Number of test images = 25858
'''
def load_testvg_ann():
	test_ann = '/media/data/nishanth/datasets/vg/test.pkl'
	f = open(test_ann,'rb')
	testvg_annotations = pk.load(f,encoding = 'bytes')
	return testvg_annotations


'''
Number of train images = 73794
'''
def load_trainvg_ann():
	train_ann = '/media/data/nishanth/datasets/vg/train.pkl'
	f = open(train_ann,'rb')
	trainvg_annotations = pk.load(f,encoding = 'bytes')
	return trainvg_annotations


def load_gt():
    train_ann = '/media/data/nishanth/datasets/vg/gt.pkl'
    f = open(train_ann,'rb')
    trainvg_annotations = pk.load(f,encoding = 'bytes')
    return trainvg_annotations


def load_zeroshot():
    zero_ann = '/media/data/nishanth/datasets/vg/zs_gt.pkl'
    f = open(zero_ann,'rb')
    zerovg_annotations = pk.load(f,encoding = 'bytes')
    test_ann = load_testvg_ann()
    sub_bboxes = zerovg_annotations[b'sub_bboxes']
    obj_bboxes = zerovg_annotations[b'obj_bboxes']
    spo_tuple = zerovg_annotations[b'tuple_label']
    gt_bb =[]
    spo_tuple_zero = []
    index =[]
    for i in range(len(spo_tuple)):
        if(spo_tuple[i].size != 0):
            gt_bbox = np.empty([len(spo_tuple[i]),2,4],dtype=int)
            for j in range(len(spo_tuple[i])):
                gt_bbox[j,0,:] = sub_bboxes[i][j]
                gt_bbox[j,1,:] = obj_bboxes[i][j]
            gt_bb.append(gt_bbox)
            spo_tuple_zero.append(spo_tuple[i])
            index.append(test_ann[i][b'img_path'].decode('utf-8').split('/')[4])
    return index,np.array(spo_tuple_zero),np.array(gt_bb)

def vgPredicatesToNames():
    f  = open('/media/data/nishanth/datasets/vg/rel.txt','r')
    pred = [line.rstrip() for line in f]
    return pred

def vgObjectsToNames():
    f  = open('/media/data/nishanth/datasets/vg/obj.txt','r')
    obj = [line.rstrip() for line in f]
    return obj



def vgImageToLabels(img_annot):
    img_file_old = img_annot[b'img_path']
    img_file = img_file_old.decode('utf-8').split('/')[4]
    sub_index = img_annot[b'ix1']
    obj_index  = img_annot[b'ix2']
    rel = img_annot[b'rel_classes']
    classes = img_annot[b'classes']
    bboxes = img_annot[b'boxes']
    gt_bb  = []
    gt_spo = []
    for i in range(len(sub_index)):
        spo_tuple = np.empty(3,dtype=int)
        spo_tuple[0] = classes[sub_index[i]]
        spo_tuple[1] = rel[i][0]
        spo_tuple[2] = classes[obj_index[i]]
        gt_spo.append(spo_tuple)
        gt_bbox = np.empty([2,4],dtype=int)
        gt_bbox[0,:] = bboxes[sub_index[i]]
        gt_bbox[1,:] = bboxes[obj_index[i]]
        # itr=[2,0,3,1]
        # itr1 = 0
        # for j in itr:
        #     gt_bbox[0,itr1] = bboxes[sub_index[i]][j]
        #     gt_bbox[1,itr1] = bboxes[obj_index[i]][j]
        #     itr1 = itr1+1
        gt_bb.append(gt_bbox)
    gt_spo_out = np.array(gt_spo)
    gt_bb_out = np.array(gt_bb)
    # print(gt_bb_out.shape,gt_spo_out.shape)
    return img_file,gt_spo_out,gt_bb_out


def main():
    load_testvg_ann()

if __name__ == '__main__':
	main()