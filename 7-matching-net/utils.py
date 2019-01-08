from keras.models import Model
from keras.layers import Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.applications.inception_v3 import preprocess_input
from keras import backend as K

from config import *
from matching_metrics import MatchCosine

"""
:param samples_per_class: samples generated per class
:param classes_per_set: classes per whole set
"""
def get_matching_network(samples_per_class, classes_per_set):
    # Image embedding using Deep Convolutional Network
    conv1 = Conv2D(64,(3,3),padding='same',activation='relu')
    bnorm1 = BatchNormalization()
    mpool1 = MaxPooling2D((2,2),padding='same')
    conv2 = Conv2D(64,(3,3),padding='same',activation='relu')
    bnorm2 = BatchNormalization()
    mpool2 = MaxPooling2D((2,2),padding='same')
    conv3 = Conv2D(64,(3,3),padding='same',activation='relu')
    bnorm3 = BatchNormalization()
    mpool3 = MaxPooling2D((2,2),padding='same')
    conv4 = Conv2D(64,(3,3),padding='same',activation='relu')
    bnorm4 = BatchNormalization()
    mpool4 = MaxPooling2D((2,2),padding='same')
    fltn = Flatten()

    # Function that generarates Deep CNN embedding given the input image x
    def convembedding(x):
        x = conv1(x)
        x = bnorm1(x)
        x = mpool1(x)
        x = conv2(x)
        x = bnorm2(x)
        x = mpool2(x)
        x = conv3(x)
        x = bnorm3(x)
        x = mpool3(x)
        x = conv4(x)
        x = bnorm4(x)
        x = mpool4(x)
        x = fltn(x)
        
        return x

    numsupportset = samples_per_class*classes_per_set
    input1 = Input((numsupportset+1,28,28,1))

    modelinputs = []
    for lidx in range(numsupportset):
        modelinputs.append(convembedding(Lambda(lambda x: x[:,lidx,:,:,:])(input1)))

    targetembedding = convembedding(Lambda(lambda x: x[:,-1,:,:,:])(input1))
    modelinputs.append(targetembedding)
    supportlabels = Input((numsupportset,classes_per_set))
    modelinputs.append(supportlabels)

    knnsimilarity = MatchCosine(nway=classes_per_set)(modelinputs)

    model = Model(inputs=[input1,supportlabels],outputs=knnsimilarity)

    return model

def get_callbacks():
    # checkpoint
    checkpoint = ModelCheckpoint(WEIGHT_FILE_PATH, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    # tensorboard
    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    # early stopping
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    return [checkpoint, tbCallBack, earlyStopping]
