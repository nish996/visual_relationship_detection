import pickle as pk
import numpy as np

from utils.vg_load import vgObjectsToNames
from utils.csv_gen import labelsToNames
'''
sematic features for a word using Glove
'''
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(os.path.join(gloveFile),'r')
    model = {}
    count = 0;
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model



'''
save vrd object glove features
'''
def saveObjGloveFeat(glove_model):
    obj = labelsToNames()
    out = {}
    for i in obj:
        out[i] = glove_model[i]
    dict_out = open('/media/data/nishanth/snapshots/obj_glove.pkl','wb')
    pk.dump(out,dict_out)
    dict_out.close()



'''
load vrd object glove features
'''
def loadObjGloveFeat():
    pickle_in = open('/media/data/nishanth/snapshots/obj_glove.pkl', 'rb')
    return pk.load(pickle_in)




'''
save vg object glove features
'''
def saveVgObjGloveFeat(glove_model):
    # glove_model = loadGloveModel(GLOVE_FILE_PATH)
    obj = vgObjectsToNames()
    out = {}
    for i in obj:
        if(i == '__background__'):
            continue
        out[i] = glove_model[i]
    dict_out = open('/media/data/nishanth/snapshots/obj_vgglove.pkl','wb')
    pk.dump(out,dict_out)
    dict_out.close()


def loadVgObjGloveFeat():
    pickle_in = open("/media/data/nishanth/snapshots/obj_vgglove.pkl","rb")
    out = pk.load(pickle_in)
    return out




def main():
    x = 10



if __name__ == '__main__':
    main()