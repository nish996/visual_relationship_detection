import numpy as np
import cv2

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
def apprSubnet(bbox,model):
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
#   model.add(Conv2D(96,(5,5),strides=(2,2) ,padding='same',activation='relu'),input_shape=)
#   model.add(Conv2D(128,(5,5),strides=(2,2) ,padding='same',activation='relu'))
#   model.add(Conv2D(64,(8,8),padding='same',activation='relu'))
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


def main():
    x  =10 #do nothing


if __name__ == '__main__':
    main()

