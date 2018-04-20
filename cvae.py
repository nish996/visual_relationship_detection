from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from keras.layers import Input, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
import numpy as np
import keras
# import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras.utils import to_categorical
import cv2
import glob, os
import random
import keras.backend.tensorflow_backend as KTF
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn import svm

import scipy.spatial.distance as spat_dist


from utils.csv_gen import train_annotations,test_annotations,labelsToNames, imageToLabels, predicatesToNames
from utils.zeroSplit import zeroTestImageToLabels
from feature_predicate import feature_extraction_tot,feature_model,load_features_tot,load_featmodel,train_vg
from utils.sem_utils import loadGloveModel, saveVgObjGloveFeat,loadVgObjGloveFeat,GLOVE_FILE_PATH,loadObjGloveFeat
from utils.vgdata_gen import vgDataGenerator,vg_preprocess,fetch_batch_vg,vg_data_gen
from utils.vg_load import load_testvg_ann,load_trainvg_ann,load_zeroshot,vgPredicatesToNames,vgObjectsToNames,vgImageToLabels
from utils.appr_utils import getUnionBBox,getSimpleSpatialFeatures,getAppr
#===================================================================#
# Some Constants

m = 32
n_x = 100
n_y = 256
n_z = 50
interNo = 1024
n_epoch = 100
FEAT_MODEL_FILE = 'feat_models/weights-30.hdf5'
FEAT_MODEL_VG = 'feat_models/weights-vg-06.hdf5'
nSamples = 5
cpu_id= "0"
num_pred =10

# ================== LAB RESOURCES ARE LIMITED=================== #



def get_session(gpu_fraction=0.8):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def load_session():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=cpu_id
    KTF.set_session(get_session())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

#===================================================================================#


def allpred2vec(glove_model):
    pred_to_names = predicatesToNames()
    pred = []
    for i in pred_to_names:
        pred_list = i
        pred_vec = []
        for j in pred_list:
            pred_vec_temp = glove_model[j]
            pred_vec.append(pred_vec_temp)
        pred_vec_temp = np.array(pred_vec)
        pred_v = np.average(pred_vec_temp,axis=0)
        #print(pred_vec_temp.shape,pred_v.shape)
        pred.append(pred_v)
    return np.array(pred)


def pred2vec(pred):
    glove_model = loadGloveModel(GLOVE_FILE_PATH)
    pred_out  = []
    pred_to_names = predicatesToNames()
    for i in pred:
        pred_list = pred_to_names[i]
        pred_vec = []
        for j in pred_list:
            pred_vec_temp = glove_model[j]
            pred_vec.append(pred_vec_temp)
        pred_vec_temp = np.array(pred_vec)
        pred_v = np.average(pred_vec_temp,axis=0)
        #print(pred_vec_temp.shape,pred_v)
        pred_out.append(pred_v)
    return np.array(pred_out)

def pred2onehot(pred,num_classes):
    return to_categorical(pred, num_classes)



def save_allpred():
    glove_model = loadGloveModel(GLOVE_FILE_PATH)
    pred_out  = allpred2vec(glove_model)
    np.save(open('snapshots/allpred_w2vec','wb'),pred_out)

def load_allpred():
    pred = np.load('snapshots/allpred_w2vec')
    return pred



def cond_pred_features_train(load_flag,num_classes):
    if not(load_flag):
        print("Loading Feature Extraction Module")
        sem_feat,appr_feat,pred = feature_extraction_tot(True,False)
        print("Loaded Features :)")
    else:
        print("Loading Features ...")
        sem_feat,appr_feat,pred = load_features_tot(True, False)
        print("Loaded Features")
    feat_model = load_featmodel(FEAT_MODEL_FILE)
    feat_model.summary()
    model =   Model(feat_model.input, feat_model.get_layer('fcfeatures').output, name='feat_model')
    feat_out = model.predict([appr_feat,sem_feat],batch_size=1)
    # pred_out = pred2vec(pred)
    pred_out = pred2onehot(pred, num_classes)
    return feat_out,pred_out

def cond_pred_features_train1(load_flag):
    if not(load_flag):
        print("Loading Feature Extraction Module")
        sem_feat,appr_feat,pred = feature_extraction_tot(True,False)
        print("Loaded Features :)")
    else:
        print("Loading Features ...")
        sem_feat,appr_feat,pred = load_features_tot(True,False)
        print("Loaded Features")
    feat_model = load_featmodel(FEAT_MODEL_FILE)
    feat_model.summary()
    model =   Model(feat_model.input, feat_model.get_layer('appr_feat').output, name='feat_model')
    appr_out = model.predict([appr_feat,sem_feat],batch_size=1)
    feat_out  = np.concatenate([appr_out,sem_feat],axis=-1)
    pred_out = pred2vec(pred)
    return feat_out,pred_out


def cond_pred_features_test(load_flag,zero_shot_flag):
    if not(load_flag):
        print("Loading Feature Extraction Module")
        sem_feat,appr_feat,pred = feature_extraction_tot(False,False)
        print("Loaded Features :)")
    else:
        print("Loading Features ...")
        sem_feat,appr_feat,pred = load_features_tot(False,zero_shot_flag)
        print("Loaded Features")
    feat_model = load_featmodel(FEAT_MODEL_FILE)
    model =   Model(feat_model.input, feat_model.get_layer('fcfeatures').output, name='feat_model')
    feat_out = model.predict([appr_feat,sem_feat],batch_size=1)
   # pred_out = pred2vec(pred)
    return feat_out,pred

def cond_pred_features_test1(load_flag,zero_shot_flag):
    if not(load_flag):
        print("Loading Feature Extraction Module")
        sem_feat,appr_feat,pred = feature_extraction_tot(False,False)
        print("Loaded Features :)")
    else:
        print("Loading Features ...")
        sem_feat,appr_feat,pred = load_features_tot(False,zero_shot_flag)
        print("Loaded Features")
    feat_model = load_featmodel(FEAT_MODEL_FILE)
    model =   Model(feat_model.input, feat_model.get_layer('appr_features').output, name='feat_model')
    feat_temp = model.predict([appr_feat,sem_feat],batch_size=1)
    feat_out = np.concatenate([feat_temp,sem_feat],axis=-1)
   # pred_out = pred2vec(pred)
    return feat_out,pred


def save_features(cond_in, pred_in,train_flag,zero_shot_flag):
    if(train_flag):
        np.save(open('snapshots/cond_feat','wb'),cond_in)
        np.save(open('snapshots/pred_feat_onehot','wb'),pred_in)
    else:
        if not(zero_shot_flag):
            np.save(open('snapshots/cond_feat_test','wb'),cond_in)
            np.save(open('snapshots/pred_feat_test','wb'),pred_in)
        else:
            np.save(open('snapshots/cond_feat_zero','wb'),cond_in)
            np.save(open('snapshots/pred_feat_zero','wb'),pred_in)

def load_features(train_flag,zero_shot_flag):
    if(train_flag):
        cond_in = np.load('snapshots/cond_feat')
        pred_in = np.load('snapshots/pred_feat_onehot')
    else:
        if not(zero_shot_flag):
            cond_in = np.load('snapshots/cond_feat_test')
            pred_in = np.load('snapshots/pred_feat_test')
        else:
            cond_in = np.load('snapshots/cond_feat_zero')
            pred_in = np.load('snapshots/pred_feat_zero')
    return cond_in, pred_in



#================================================================#

#VAE model

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=[n_z], mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps



def vae_model():
    input_ic = Input(shape=[n_x+n_y], name = 'pred' )
    cond  = Input(shape=[n_y] , name='feat_vector')
    h_q = Dense(interNo, activation='relu',name= 'enc_h1')(input_ic)
    # h_q_zd = Dropout(rate=0.7)(temp_h_q)
    h_q_out = Dense(interNo, activation='relu')(h_q)
    h_q_bn = BatchNormalization()(h_q_out)
    mu = Dense(n_z, activation='linear',name='mu')(h_q_bn)
    log_sigma = Dense(n_z, activation='linear',name='log_sig')(h_q_bn)

    z = Lambda(sample_z,name='z')([mu, log_sigma])
    # z_cond = merge([z, cond] , mode='concat', concat_axis=1)

    z_cond = concatenate([z, cond] , axis=1,name='z_cond')

    decoder_hidden = Dense(1024, activation='relu',name = 'd_h')
    decoder_hidden1= Dense(1024,activation='relu',name='d_h1')
    decoder_bn = BatchNormalization(name='d_bn')
    decoder_out = Dense(n_x, activation='linear',name = 'd_out')
    h_p = decoder_hidden(z_cond)
    h_p1 = decoder_hidden1(h_p)
    h_p2 = decoder_bn(h_p1)
    reconstr = decoder_out(h_p2)
    vae = Model(inputs=[input_ic , cond], outputs=[reconstr])
    encoder = Model(inputs=[input_ic , cond], outputs=[mu])


    d_in = Input(shape=[n_z+n_y],name='d_in')
    d_h = decoder_hidden(d_in)
    d_h1 = decoder_hidden1(d_h)
    d_h2 = decoder_bn(d_h1)
    d_out = decoder_out(d_h2)
    decoder = Model(d_in, d_out)

    # encoder.summary()
    # decoder.summary()
    vae.summary()

    return vae

def recon_loss(y_true, y_pred,log_sigma,mu):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.mean(K.square(y_pred - y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    #print 'kl : ' + str(kl)
    return recon + kl

def vae_loss(log_sigma,mu):
    def recon(y_true,y_pred):
        return recon_loss(y_true, y_pred, log_sigma, mu)
    return recon


def vae_train(load_flag):
    load_session()
    vae = vae_model()
    vae.compile(optimizer='adam', loss=vae_loss(vae.get_layer('log_sig').output,vae.get_layer('mu').output))
    if not(load_flag):
        cond_in, pred_in = cond_pred_features_train(not(load_flag),70)
    else:
        cond_in, pred_in = load_features(True,False)
    total_vec = np.concatenate((pred_in,cond_in), axis=1)
    modelsave_filepath="snapshots/vae_models/vae-weightsonehot-{epoch:02d}.hdf5"
    checkpointer = ModelCheckpoint(modelsave_filepath, verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=5)
    vae.fit({'pred':total_vec,'feat_vector':cond_in}, pred_in, batch_size=m, epochs=n_epoch,callbacks=[checkpointer])
    #vae.fit({'pred':pred_in,'feat_vector':cond_in}, pred_in, batch_size=m, epochs=n_epoch,callbacks=[checkpointer])
    return vae

def vae_test(load_flag,zero_shot_flag,svm_flag):
    if not(load_flag):
        cond_in, pred_tr = cond_pred_features_test(load_flag,zero_shot_flag)
    else:
        cond_in, pred_tr = load_features(False,zero_shot_flag)
    print(cond_in.shape,pred_tr.shape)
    load_session()
    vae=vae_model()
    vae.load_weights('snapshots/vae_models/vae-weightsonehot-05.hdf5', by_name=True)

    decoder = Model(vae.get_layer('d_h').get_input_at(1), vae.get_layer('d_out').get_output_at(1), name='decoder')

    decoder.summary()
    test_size = pred_tr.shape[0]
    total_ex = test_size*nSamples
    noise_gen = np.random.normal(size=(total_ex,n_z))
    new_cond_in = []
    new_pred_tr=[]
    print("Generating Samples")
    for i in range(test_size):
        for j in range(0,nSamples):
            new_cond_in.append(cond_in[i])
            new_pred_tr.append(pred_tr[i])
    sample_cond_in = np.array(new_cond_in)
    sample_pred_tr = np.array(new_pred_tr)
    print(noise_gen.shape,cond_in.shape)
    print("Generated Samples")
    total_vec = np.concatenate((noise_gen,sample_cond_in), axis=1)
    print("Generating Predicates")
    pred_gen = decoder.predict(total_vec)

    # Normalize for SVM accuracy...
    if(svm_flag):
        pseudoTrainData = normalize(pred_gen , axis=1)
        testData = normalize(to_categorical(pred_tr,70) , axis=1)

        print('Training SVM-100')
        clf5 = svm.SVC(verbose=True)
        clf5.fit(pseudoTrainData, sample_pred_tr)
        print ('Predicting...')
        pred = clf5.predict(testData)
    print(sample_pred_tr.shape)
    print(accuracy_score(pred, pred_tr))

    return pred_gen, sample_pred_tr


def cos_list(pred,all_pred):
    out= np.empty(num_pred)
    for i in range(num_pred):
        out[i] = 1-spat_dist.cosine(pred,all_pred[i])
    return np.argmax(out)

def vae_accuracy(pred_gen, pred_tr):
    all_pred = load_allpred()
    count = 0
    for i in range(pred_gen.shape[0]):
        out = cos_list(pred_gen[i],all_pred)
        if(out == pred_tr[i]):
            count = count +1
    return count/pred_gen.shape[0]






########################################################################################################
#vg preprocessing


def train_vae_vg(train_flag,batch_size,epoch):
    load_session()
    feat_model =  load_featmodel(FEAT_MODEL_VG)
    model =  Model(feat_model.input, feat_model.get_layer('fcfeatures').output, name='feat_model')
    vae = vae_model()
    vae.compile(optimizer='adam', loss=vae_loss(vae.get_layer('log_sig').output,vae.get_layer('mu').output))
    modelsave_filepath="snapshots/vae_models/vae-weightvg-{0:02d}.hdf5" 
    glove_model =  loadVgObjGloveFeat()
    labels_to_names = vgObjectsToNames()
    if(train_flag):
        annotations = load_trainvg_ann()
        steps = 73794
    else:
        annotations =  load_testvg_ann()
        steps  = 25858
    tot_features = []
    tot_appr_features =[]
    tot_pred = []
    filepath = 'datasets/vg/VG_100K/'
    # vg_generator = vgDataGenerator(annotations,glove_model, labels_to_names, batch_size=1, n_classes=100, shuffle=True)
    # feat_out =  model.predict_generator(vg_generator, steps= steps, max_queue_size=10, workers=10, use_multiprocessing=True, verbose=1)
    # print(feat_out.shape)
    # return feat_out
    for j in range(epoch):
        print("Epoch ",j)
        random.shuffle(annotations, random=random.random)
        sample_count  = 0
        batch_count = 0
        t =0
        cum_x = 0
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
                    cond_in = model.predict_on_batch([X,X1])
                    total_vec = np.concatenate((Y,cond_in), axis=1)
                    x = vae.train_on_batch({'pred':total_vec,'feat_vector':cond_in}, Y)
                    cum_x = cum_x+x
                    tot_features = []
                    tot_appr_features =[]
                    tot_pred = []
                    avg = cum_x/batch_count
                    print(x,avg)
            print(sample_count)
        vae.save(modelsave_filepath.format(j))
    return

def train_vae_vggenerator():
    load_session()
    feat_model =  load_featmodel(FEAT_MODEL_VG)
    model =  Model(feat_model.input, feat_model.get_layer('fcfeatures').output, name='feat_model')
    vae = vae_model()
    vae.compile(optimizer='adam', loss=vae_loss(vae.get_layer('log_sig').output,vae.get_layer('mu').output))
    modelsave_filepath="snapshots/vae_models/vae-weightvg-{epoch:02d}.hdf5" 
    checkpointer = ModelCheckpoint(modelsave_filepath, verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    glove_model =  loadVgObjGloveFeat()
    labels_to_names = vgObjectsToNames()
    annotations = load_trainvg_ann()
    vg_generator = vgDataGenerator(annotations,glove_model, labels_to_names, batch_size=1, n_classes=100, shuffle=True)
    generator = vg_data_gen(True, glove_model, labels_to_names, 1,100)
    model.predict_generator(vg_generator, steps_per_epoch=73794, epochs=20, verbose=1,workers=5, use_multiprocessing=True)
    return

def cond_pred_vgfeatures_zero():
    load_session()
    print("Loading Feature Extraction Module")
    sem_feat,appr_feat,pred = train_vg(False,True,1,1)
    print("Loaded Features :)")
    feat_model = load_featmodel(FEAT_MODEL_VG)
    model =   Model(feat_model.input, feat_model.get_layer('fcfeatures').output, name='feat_model')
    feat_out = model.predict([appr_feat,sem_feat],batch_size=1)
   # pred_out = pred2vec(pred)
    return feat_out,pred



def vae_test_zerovg(svm_flag):
    cond_in, pred_tr = cond_pred_vgfeatures_zero()
    print(cond_in.shape,pred_tr.shape)
    load_session()
    vae=vae_model()
    vae.load_weights('snapshots/vae_models/vae-weightvg-03.hdf5', by_name=True)

    decoder = Model(vae.get_layer('d_h').get_input_at(1), vae.get_layer('d_out').get_output_at(1), name='decoder')

    decoder.summary()
    test_size = pred_tr.shape[0]
    total_ex = test_size*nSamples
    noise_gen = np.random.normal(size=(total_ex,n_z))
    new_cond_in = []
    new_pred_tr=[]
    print("Generating Samples")
    for i in range(test_size):
        for j in range(0,nSamples):
            new_cond_in.append(cond_in[i])
            new_pred_tr.append(pred_tr[i])
    sample_cond_in = np.array(new_cond_in)
    sample_pred_tr = np.array(new_pred_tr)
    print(noise_gen.shape,cond_in.shape)
    print("Generated Samples")
    total_vec = np.concatenate((noise_gen,sample_cond_in), axis=1)
    print("Generating Predicates")
    pred_gen = decoder.predict(total_vec)

    # Normalize for SVM accuracy...
    if(svm_flag):
        pseudoTrainData = normalize(pred_gen , axis=1)
        testData = normalize(to_categorical(pred_tr,100) , axis=1)

        print('Training SVM-100')
        clf5 = svm.SVC(verbose=True)
        clf5.fit(pseudoTrainData, sample_pred_tr)
        print ('Predicting...')
        pred = clf5.predict(testData)
        print(sample_pred_tr.shape)
        print(accuracy_score(pred, pred_tr))
    else:
        print(accuracy_score(np.argmax(pred_gen,axis=1),sample_pred_tr))

    return pred_gen, sample_pred_tr



########################################################################################

def main():
    # vae = vae_train(True)
    # print("Loading Test Model")
    # pred_gen, pred_tr = vae_test(True,True)
    # print("Predicates Generated")
    # print("Calculating Accuracy")
    # score = vae_accuracy(pred_gen, pred_tr)
    # print(score)
    # cond_in, pred_in = cond_pred_features_test(True, True)
    # save_features(cond_in, pred_in, False,True)
    # cond_in, pred_in = load_features(False,True)
    # print(cond_in.shape,pred_in)
    # vae_test(True, True, False)
    x = 10

if __name__ == '__main__':
        main()
