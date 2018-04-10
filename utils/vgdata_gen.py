from keras.utils import Sequence
from keras.utils import to_categorical
import os
import numpy as np
import cv2
import random
from utils.vg_load import vgImageToLabels,load_testvg_ann,load_trainvg_ann
from utils.appr_utils import getAppr,getUnionBBox,getSimpleSpatialFeatures


def vg_preprocess(img,glove_model,labels_to_names):
    filepath = '/media/data/nishanth/datasets/vg/VG_100K/'
    tot_features = []
    tot_appr_features =[]
    tot_pred = []
    img_file,gt_spo,gt_bb = vgImageToLabels(img)
    im = cv2.imread(os.path.join(filepath,img_file)).astype(np.float32, copy=False)
    ih = im.shape[0]
    iw = im.shape[1]
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
        tot_appr_features.append(np.transpose(appr_feat_temp))
        tot_features.append(tot_feat)
        tot_pred.append(gt_spo[i,1])
    return np.array(tot_features),np.array(tot_appr_features),np.array(tot_pred)
    # return tot_features,tot_appr_features,tot_pred

def fetch_batch_vg(annotations,batch_size,glove_model,labels_to_names,num_classes):
    batch_features = []
    batch_labels = []
    for i in range(batch_size):
        x  =[]
        index = random.choice(annotations)
        sem_feat,appr_feat,pred = vg_preprocess(index,glove_model,labels_to_names)
        x.append(appr_feat)
        x.append(sem_feat)
        batch_features.append(x)
        batch_labels.append(to_categorical(pred, num_classes))
    return np.array(batch_features),np.array(batch_labels)



def vg_data_gen(train_flag,glove_model,labels_to_names,batch_size,num_classes):
    if train_flag:
        annotations = load_trainvg_ann()
    else:
        annotations = load_testvg_ann()

    while True:
        batch_features,batch_labels=fetch_batch_vg(annotations,batch_size,glove_model,labels_to_names,num_classes)
        yield batch_features, batch_labels


class vgDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, glove_model,labels_to_names, batch_size=32,
                 n_classes=100, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.glove_model = glove_model
        self.labels_to_names = labels_to_names
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        init_size= len(list_IDs_temp[0][b'rel_classes'])
        X = np.empty((init_size,224,224,3))
        X1= np.empty((init_size,610))
        y= np.empty(init_size, dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # get sample
            if(i == 0):
                X1,X,y= vg_preprocess(ID,self.glove_model,self.labels_to_names)
            else:
                sem_feat,appr_feat,pred= vg_preprocess(ID,self.glove_model,self.labels_to_names)
                X=np.concatenate((X,appr_feat),axis=0)
                X1=np.concatenate((X1,sem_feat),axis=0)
                y=np.concatenate((y,pred),axis=0)
        if(X.shape[0] > 32):
            X=X[:32]
            X1=X1[:32]
            y=y[:32]
        return [X,X1],to_categorical(y, num_classes=self.n_classes)
