import keras
import os
import sys
scriptpath = "/keras-retinanet/"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

#import required keras models
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import random
import pickle as pk

#import utils modules
from utils.csv_gen import train_annotations,test_annotations,labelsToNames, imageToLabels, predicatesToNames
from utils.zeroSplit import zeroTestImageToLabels
from utils.vg_load import load_testvg_ann,load_trainvg_ann,load_zeroshot,vgPredicatesToNames,vgObjectsToNames,vgImageToLabels
from utils.appr_utils import getAppr,getUnionBBox,getDualMask,getSimpleSpatialFeatures
from utils.sem_utils import loadGloveModel, saveVgObjGloveFeat,loadVgObjGloveFeat,GLOVE_FILE_PATH,loadObjGloveFeat
from utils.vgdata_gen import vgDataGenerator,vg_preprocess,fetch_batch_vg,vg_data_gen


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

cpu_id = "2"


def load_session():
    # use this environment flag to change which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = cpu_id

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)





def load_featmodel(model_name):


    # adjust this to point to your downloaded/trained model
    model_path = os.path.join('snapshots', model_name)

    # load retinanet model
    model = keras.models.load_model(model_path)

    return model


###########################################################################################
#Models



def vgg_model(fc_flag):
    if(fc_flag):
        base_model = VGG16(weights='imagenet')
        model = Model(base_model.inputs, base_model.get_layer('fc2').output, name=None)
        model.summary()
        return model
    else:
        appr_inputs = Input(shape=(224,224,3),name='appr_inputs')
        base_model = VGG16(include_top=False, weights='imagenet')(appr_inputs)
        x = Flatten()(base_model)
        model= Model(appr_inputs,x, name=None)
        model.summary()
        return model


def feature_model_novgg(fc_flag):
    if(fc_flag):
        appr_inputs = Input(shape=4096,name='appr_inputs')
    else:
        appr_inputs = Input(25088,name='appr_inputs')
    x = Dense(4096, activation='relu')(app)
    x = Dense(4096, activation='relu')(x)
    appr_feats = Dense(256, activation='relu',name='appr_feat')(x)
    sem_feat_input = Input(shape=(610,), name = 'semantic_feat')
    merged_features  = keras.layers.concatenate([appr_feats,sem_feat_input],axis=-1)
    fclayer = Dense(256, activation='relu',name = 'fcfeatures')(merged_features)
    pred_layer = Dense(num_classes, activation='softmax', name='predictions')(fclayer)
    model = Model(inputs=[appr_inputs,sem_feat_input], outputs=pred_layer, name='feature model')
    model.summary()
    return model



'''
Model for combining spatial, semantic and appearance features
'''
def feature_model(num_classes):
    appr_inputs = Input(shape=(224,224,3),name='appr_inputs')
    appr_model = VGG16(include_top=False, weights='imagenet')(appr_inputs)
    x= Flatten()(appr_model)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    appr_feats = Dense(256, activation='relu',name='appr_feat')(x)
    sem_feat_input = Input(shape=(610,), name = 'semantic_feat')
    merged_features  = keras.layers.concatenate([appr_feats,sem_feat_input],axis=-1)
    fclayer = Dense(256, activation='relu',name = 'fcfeatures')(merged_features)
    pred_layer = Dense(num_classes, activation='softmax', name='predictions')(fclayer)
    model = Model(inputs=[appr_inputs,sem_feat_input], outputs=pred_layer, name='feature model')
    model.summary()
    return model




##############################################################################################
#Feature Extraction Modules for VRD


'''
extracts spatial, semantic and preprocessed appearance features for a given image
'''
def feature_extraction(img_file):
	im = cv2.imread(os.path.join('datasets/sg_dataset/sg_train_images',img_file)).astype(np.float32, copy=False)
	ih = im.shape[0]
	iw = im.shape[1]
	glove_model = loadObjGloveFeat()
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


'''
extracts spatial, semantic and preprocessed appearance features for whole dataset
if flag is set to true: extracts features from train data
else extracts features from test data
'''
def feature_extraction_tot(train_flag,zero_shot_flag):
    glove_model = loadObjGloveFeat()
    labels_to_names = labelsToNames()
    tot_features = []
    tot_appr_features =[]
    tot_pred = []
    if train_flag:
        annotations = train_annotations
        filepath = 'datasets/sg_dataset/sg_train_images'
    else:
        annotations = test_annotations
        filepath = 'datasets/sg_dataset/sg_test_images'
    for img_file in annotations:
        im = cv2.imread(os.path.join(filepath,img_file)).astype(np.float32, copy=False)
        ih = im.shape[0]
        iw = im.shape[1]
        if (zero_shot_flag != True):
            gt_spo,gt_bb = imageToLabels(annotations, img_file)
        else :
            gt_spo,gt_bb = zeroTestImageToLabels(img_file)
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
    print(np.array(tot_features).shape,np.array(tot_appr_features).shape,np.array(tot_pred).shape)
    return np.array(tot_features),np.array(tot_appr_features),np.array(tot_pred)


'''
training the feature model
'''
def train_model(load_flag):
    load_session()
    if not(load_flag):
        print("Loading Feature Extraction Module")
        sem_feat,appr_feat,pred = feature_extraction_tot(True, False)
        print("Loaded Features :)")
    else:
        print("Loading Features ...")
        sem_feat,appr_feat,pred = load_features_tot(True,False)
        print("Loaded Features")
    pred_encoded = to_categorical(pred, num_classes=70)
    model = feature_model(70)
    model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])
    modelsave_filepath="snapshots/feat_models/weights-{epoch:02d}.hdf5"
    checkpointer = ModelCheckpoint(modelsave_filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(x=[appr_feat,sem_feat], y=pred_encoded, epochs=30, callbacks= [checkpointer],batch_size=32)
    return


def resume_model():
    load_session()
    print("Loading Feature Extraction Module")
    sem_feat,appr_feat,pred = feature_extraction_tot(True, False)
    print("Loaded Features :)")
    pred_encoded = to_categorical(pred, num_classes=70)
    model = load_model('weights-30.hdf5')
    model.summary()
    model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])
    modelsave_filepath="snapshots/feat_models/weights-30-{epoch:02d}.hdf5"
    checkpointer = ModelCheckpoint(modelsave_filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)
    model.fit(x=[appr_feat,sem_feat], y=pred_encoded, epochs=30, callbacks= [checkpointer],batch_size=32)
    return

'''
testing the feature model
'''
def test_model(zero_shot_flag):
    print('Loading Model')
    feature_model = load_featmodel('feat_models/weights-30.hdf5')
    feature_model.summary()
    print('Loaded Model')
    print("Loading Feature Extraction Module")
    sem_feat,appr_feat,pred = feature_extraction_tot(False,zero_shot_flag)
    print("Loaded Features :)")
    pred_encoded = to_categorical(pred, num_classes=70)
    scores = feature_model.evaluate([appr_feat,sem_feat],pred_encoded)
    print(scores)



def save_features(train_flag,zero_shot_flag):
    if(train_flag):
        sem_feat,appr_feat,pred =feature_extraction_tot(train_flag, zero_shot_flag)
        np.save(open('snapshots/sem_feat','wb'),sem_feat)
        np.save(open('snapshots/appr_feat','wb'),appr_feat)
        np.save(open('snapshots/pred','wb'),pred)
    else:
        if not(zero_shot_flag):
            sem_feat,appr_feat,pred =feature_extraction_tot(train_flag, zero_shot_flag)
            np.save(open('snapshots/sem_feat_test','wb'),sem_feat)
            np.save(open('snapshots/appr_feat_test','wb'),appr_feat)
            np.save(open('snapshots/pred_test','wb'),pred)
        else:
            sem_feat,appr_feat,pred =feature_extraction_tot(train_flag, zero_shot_flag)
            np.save(open('snapshots/sem_feat_zero','wb'),sem_feat)
            np.save(open('snapshots/appr_feat_zero','wb'),appr_feat)
            np.save(open('snapshots/pred_zero','wb'),pred)

def load_features_tot(train_flag,zero_shot_flag):
    if(train_flag):
        sem_feat = np.load('snapshots/sem_feat')
        appr_feat = np.load('snapshots/appr_feat')
        pred = np.load('snapshots/pred')
    else:
        if not(zero_shot_flag):
            sem_feat = np.load('snapshots/sem_feat_test')
            appr_feat = np.load('snapshots/appr_feat_test')
            pred = np.load('snapshots/pred_test')
        else:
            sem_feat = np.load('snapshots/sem_feat_zero')
            appr_feat = np.load('snapshots/appr_feat_zero')
            pred = np.load('snapshots/pred_zero')
    return sem_feat,appr_feat,pred





######################################################################################################
#model and features modules for Visual Genome
def save_featuresVG(train_flag,zero_shot_flag,glove_model,fc_flag):
    if(train_flag):
        sem_feat,appr_feat,pred =feature_extraction_vgtot1(train_flag, zero_shot_flag,glove_model,fc_flag)
        np.save(open('snapshots/sem_featvg1_1','wb'),sem_feat)
        np.save(open('snapshots/appr_featvg1_1','wb'),appr_feat)
        np.save(open('snapshots/predvg1_1','wb'),pred)
    else:
        if not(zero_shot_flag):
            sem_feat,appr_feat,pred =feature_extraction_vgtot(train_flag, zero_shot_flag)
            np.save(open('snapshots/sem_feat_test','wb'),sem_feat)
            np.save(open('snapshots/appr_feat_test','wb'),appr_feat)
            np.save(open('snapshots/pred_test','wb'),pred)
        else:
            sem_feat,appr_feat,pred =feature_extraction_vgtot(train_flag, zero_shot_flag)
            np.save(open('snapshots/sem_feat_zero','wb'),sem_feat)
            np.save(open('snapshots/appr_feat_zero','wb'),appr_feat)
            np.save(open('snapshots/pred_zero','wb'),pred)


def train_vg(train_flag,zero_shot_flag,batch_size,epoch):
    if(not(zero_shot_flag)): 
        load_session()
        model = feature_model(100)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        modelsave_filepath="snapshots/feat_models/weights-newvg-{0:02d}.hdf5"
    glove_model = loadVgObjGloveFeat()
    labels_to_names = vgObjectsToNames()
    tot_features = []
    tot_appr_features =[]
    tot_pred = []
    filepath = 'datasets/vg/VG_100K/'
    if train_flag:
        annotations = load_trainvg_ann()
    else:
        annotations = load_testvg_ann()
    if zero_shot_flag:
        img_files,gt_spo,gt_bb = load_zeroshot()
        for j in range(len(img_files)):
            im = cv2.imread(os.path.join(filepath,img_files[j])).astype(np.float32, copy=False)
            ih = im.shape[0]
            iw = im.shape[1]
            for i in range(gt_spo[j].shape[0]):
                sub_bbox = gt_bb[j][i,0,:4].astype(int)
                obj_bbox = gt_bb[j][i,1,:4].astype(int)
                sub_wordvec = glove_model[labels_to_names[gt_spo[j][i,0]]]
                obj_wordvec = glove_model[labels_to_names[gt_spo[j][i,2]]]
                sem_feat = np.concatenate((sub_wordvec,obj_wordvec),axis=0)
                union_bbox = getUnionBBox(sub_bbox, obj_bbox, ih, iw)
                appr_feat_temp = getAppr(im, union_bbox)
                #appr_feat = apprSubnet(appr_feat_temp)
                spat_feat = np.concatenate((getSimpleSpatialFeatures(ih, iw, sub_bbox), getSimpleSpatialFeatures(ih, iw, obj_bbox)),axis=0)
                tot_feat = np.concatenate((sem_feat,spat_feat), axis=0)
                tot_appr_features.append(np.transpose(appr_feat_temp))
                tot_features.append(tot_feat)
                tot_pred.append(gt_spo[j][i,1])
        print(np.array(tot_features).shape,np.array(tot_appr_features).shape,np.array(tot_pred).shape)
        return np.array(tot_features),np.array(tot_appr_features),np.array(tot_pred)
    else:
        print('Loaded Annotations')
    for j in range(epoch):
        print("Epoch ",j)
        random.shuffle(annotations, random=random.random)
        sample_count  = 0
        batch_count = 0
        t =0
        cum_x = np.zeros(2)
        for img in annotations:
            sample_count = sample_count+1
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
                t  = t+1
                if(t % batch_size == 0 ):
                    batch_count = batch_count +1
                    X = np.array(tot_appr_features)
                    X1 = np.array(tot_features)
                    Y = to_categorical(np.array(tot_pred),100)
                    x = model.train_on_batch([X,X1],Y)
                    cum_x = cum_x+x
                    tot_features = []
                    tot_appr_features =[]
                    tot_pred = []
                    print(x)
                    avg = cum_x/batch_count
                    print(avg)
            print(sample_count)
        model.save(modelsave_filepath.format(j))
    return model


def feature_extraction_vgtot1(train_flag,zero_shot_flag,fc_flag):
    glove_model =  loadVgObjGloveFeat()
    labels_to_names = vgObjectsToNames()
    tot_features = []
    tot_appr_features =[]
    tot_pred = []
    vgg= vgg_model(fc_flag)
    filepath = 'datasets/vg/VG_100K/'
    if train_flag:
        annotations = load_trainvg_ann()
    else:
        annotations = load_testvg_ann()
    if zero_shot_flag:
        img_files,gt_spo,gt_bb = load_zeroshot()
        for j in range(len(img_files)):
            im = cv2.imread(os.path.join(filepath,img_files[j])).astype(np.float32, copy=False)
            ih = im.shape[0]
            iw = im.shape[1]
            for i in range(gt_spo[j].shape[0]):
                sub_bbox = gt_bb[j][i,0,:4].astype(int)
                obj_bbox = gt_bb[j][i,1,:4].astype(int)
                sub_wordvec = glove_model[labels_to_names[gt_spo[j][i,0]]]
                obj_wordvec = glove_model[labels_to_names[gt_spo[j][i,2]]]
                sem_feat = np.concatenate((sub_wordvec,obj_wordvec),axis=0)
                union_bbox = getUnionBBox(sub_bbox, obj_bbox, ih, iw)
                appr_feat_temp = getAppr(im, union_bbox)
                #appr_feat = apprSubnet(appr_feat_temp)
                spat_feat = np.concatenate((getSimpleSpatialFeatures(ih, iw, sub_bbox), getSimpleSpatialFeatures(ih, iw, obj_bbox)),axis=0)
                tot_feat = np.concatenate((sem_feat,spat_feat), axis=0)
                print(appr_feat_temp.shape)
                tot_appr_features.append(vgg.predict(np.transpose(appr_feat_temp)))
                tot_features.append(tot_feat)
                tot_pred.append(gt_spo[j][i,1])
        print(np.array(tot_features).shape,np.array(tot_appr_features).shape,np.array(tot_pred).shape)
        return np.array(tot_features),np.array(tot_appr_features),np.array(tot_pred)
    else:
        print('Loaded Annotations')
        t =0
        for img in annotations:
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
                x = vgg.predict(np.expand_dims(np.transpose(appr_feat_temp),axis=0))
                x= np.squeeze(x, axis=0)
                tot_appr_features.append(x)
                tot_features.append(tot_feat)
                tot_pred.append(gt_spo[i,1])
        print(np.array(tot_features).shape,np.array(tot_appr_features).shape,np.array(tot_pred).shape)
        return np.array(tot_features),np.array(tot_appr_features),np.array(tot_pred)


'''
training the feature model for visual genome
'''
def train_model_vg():
    load_session()
    model = feature_model(100)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    modelsave_filepath="snapshots/feat_models/weights-vg-{epoch:02d}.hdf5"
    checkpointer = ModelCheckpoint(modelsave_filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    glove_model =  loadVgObjGloveFeat()
    labels_to_names = vgObjectsToNames()
    annotations = load_trainvg_ann()
    vg_generator = vgDataGenerator(annotations,glove_model, labels_to_names, batch_size=1, n_classes=100, shuffle=True)
    generator = vg_data_gen(True, glove_model, labels_to_names, 1,100)
    model.fit_generator(vg_generator, steps_per_epoch=73794, epochs=20, verbose=1, callbacks=[checkpointer], workers=5, use_multiprocessing=True)
    return



'''
testing the feature model for visual genome
'''
def test_model_vg():
    load_session()
    print('Loading Model')
    feature_model = load_featmodel('feat_models/weights-vg40-1.hdf5')
    feature_model.summary()
    print('Loaded Model')
    glove_model =  loadVgObjGloveFeat()
    labels_to_names = vgObjectsToNames()
    annotations = load_testvg_ann()
    vg_generator = vgDataGenerator(annotations,glove_model, labels_to_names, batch_size=1, n_classes=100, shuffle=True)
    generator = vg_data_gen(True, glove_model, labels_to_names, 1,100)
    scores =feature_model.evaluate_generator(vg_generator, steps=25858, max_queue_size=10, workers=1, use_multiprocessing=True)
    print(scores)
    return




'''
testing thr feature model zeroshot
'''
def test_model_vgzeroshot():
    load_session()
    print('Loading Model')
    feature_model = load_featmodel('feat_models/weights-vg-06.hdf5')
    feature_model.summary()
    print('Loaded Model')
    print("Loading Feature Extraction Module")
    sem_feat,appr_feat,pred = train_vg(False,True, 40,1)
    print("Loaded Features :)")
    pred_encoded = to_categorical(pred, num_classes=100)
    scores = feature_model.evaluate([appr_feat,sem_feat],pred_encoded)
    print(scores)


##################################################################################################################################################################################






def main():
	#resnet50_coco_best_v2.0.1.h5
	#model = load_model('resnet50_csv_07.h5')
	#bb_score = image_pred(model,'objdet/test_images/image3.jpg')
	#visualize_pred('objdet/test_images/image3.jpg',bb_score)
	#feat= feature_extraction('1602315_961e6acf72_b.jpg')
    #resume_model()
    # save_featuresVG(True,False)
    # load_features_tot()
    # x =fetch_batch_vg(32,glove_model,labels_to_names):
    # train_model_vg()
    # a = vgg_model()
    # train_model(True)
    # feature_extraction_vgtot1(True,False,True)
    x =10



if __name__ == "__main__":
    main()


