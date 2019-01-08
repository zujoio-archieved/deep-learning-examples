WEIGHT_FILE_PATH="weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
BSIZE = 32 # batch size
CLASSES_PER_SET = 5 # classes per set or 5-way
SAMPLES_PER_SET = 1 # samples per class 1-short
TRAIN_SIZE = 64000
VAL_SIZE = 20000
EPOCHS = 10