"""
Implementation of "Matching network for one short learning" in Keras
__author__ = Chetan Nichkawde
"""

from keras.models import Model
from keras.layers import Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from matching_metrics import MatchCosine
from data_generator import *
from utils import *

data = OmniglotNShotDataset(batch_size=BSIZE,
                classes_per_set=CLASSES_PER_SET,
                samples_per_class=SAMPLES_PER_SET,
                trainsize=TRAIN_SIZE,
                valsize=VAL_SIZE)

model = get_matching_network(SAMPLES_PER_SET, CLASSES_PER_SET)
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("0", data.datasets_cache["train"][0].shape)
print("1", data.datasets_cache["train"][1].shape)
print("2", data.datasets_cache["train"][2].shape)
model.fit([data.datasets_cache["train"][0],data.datasets_cache["train"][1]],data.datasets_cache["train"][2],
          validation_data=[[data.datasets_cache["val"][0],data.datasets_cache["val"][1]],data.datasets_cache["val"][2]],
          epochs=EPOCHS,
          batch_size=BSIZE,
          verbose=1,
          callbacks = get_callbacks())