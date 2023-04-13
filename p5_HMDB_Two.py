import tensorflow as tf
import os
import json
import p4
import p5_HMDB_Fra
import numpy as np
import p5_HMDB_Opt
from keras import layers
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Generate two-stream sequence to feed into generators
class TwoStreamSequence(Sequence):
    def __init__(self, fra, opt, labels):
        self.fra = fra
        self.opt = opt
        self.labels = labels

    def __len__(self):
        return len(self.fra)

    def __getitem__(self, idx):
        indexes = np.arange(len(self.fra))
        X_fra = self.fra[indexes[idx]]
        X_opt = self.opt[indexes[idx]]
        # Combine the two input streams
        X = [X_fra, X_opt]
        y = self.labels[indexes[idx]]
        return X, y

# Generate two-stream sequence from generators to train
class TwoStreamSequenceGen(Sequence):
    def __init__(self, generator_fra, generator_opt, shuffle=True):
        self.generator_fra = generator_fra
        self.generator_opt = generator_opt
        self.shuffle = shuffle

    def __len__(self):
        return len(self.generator_fra)

    def __getitem__(self, idx):
        if self.shuffle:
            indexes = np.random.permutation(len(self.generator_fra))     # Shuffle the indexes
        else:
            indexes = np.arange(len(self.generator_fra))
        X_fra, y_fra = self.generator_fra[indexes[idx]]
        X_opt, y_opt = self.generator_opt[indexes[idx]]
        # Combine the two input streams
        # print(y_fra[0])
        # print(y_opt[0])
        X = [X_fra, X_opt]
        y = y_fra
        return X, y

# Build and train the model
def HMDB_Two(train_generator_fra, val_generator_fra, train_generator_opt, val_generator_opt,
             len_train, len_val):

    frame_model = tf.keras.models.load_model('./DATA/HMDB_final.h5')
    opticalflow_model = tf.keras.models.load_model('./DATA/HMDB_Opt_final.h5')

    # Create a two-stream CNN
    frame_input = layers.Input(shape=(224, 224, 3))
    opticalflow_input = layers.Input(shape=(16, 224, 224, 2))

    frame_output = frame_model(frame_input)
    opticalflow_output = opticalflow_model(opticalflow_input)

    merged = layers.concatenate([frame_output, opticalflow_output])
    merged = layers.Dense(256, activation='relu', name='two_dense')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(12, activation='softmax', name='two_dense_1')(merged)

    two_stream_model = tf.keras.models.Model(inputs=[frame_input, opticalflow_input], outputs=output)
    two_stream_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    two_stream_model.summary()
    # for i, w in enumerate(two_stream_model.weights):
    #     print(i, w.name)
    train_two_stream = TwoStreamSequenceGen(train_generator_fra, train_generator_opt, True)
    test_two_stream = TwoStreamSequenceGen(val_generator_fra, val_generator_opt, False)

    two_stream = two_stream_model.fit(
        train_two_stream,
        steps_per_epoch=len_train // 32,
        epochs=13,
        validation_data=test_two_stream,
        validation_steps=len_val // 32
    )

    #print(two_stream_model.weights)

    if not os.path.exists('./DATA/'):
        os.makedirs('./DATA/')

    two_stream_model.save('./DATA/HMDB_Two_final.h5')
    with open('./DATA/HMDB_Two_final.json', 'w') as f:
        json.dump(two_stream.history, f)

# Generate two-stream sequence to feed into generators
def data_generation(all_files_fra, all_labels_fra, all_files_opt, all_labels_opt):
    # print(all_labels_fra[100])
    # print(all_labels_opt[100])
    two_stream = TwoStreamSequence(all_files_fra, all_files_opt, all_labels_fra)

    all_files = []
    all_labels = []
    for i in range(len(two_stream)):
        all_files.append(two_stream[i][0])
        all_labels.append(two_stream[i][1])

    train, test, trainLabels, testLabels = train_test_split(all_files, all_labels, test_size=0.1,
                                                            stratify=all_labels, random_state=0)

    train_exp, val_exp, trainLabels_exp, valLabels_exp = train_test_split(train, trainLabels,
                                                                          test_size=0.1, stratify=trainLabels,
                                                                          random_state=0)
    train_fra = []
    train_opt = []
    trainLabel_two = []
    for i in range(len(train)):
        train_fra.append(train[i][0])
        train_opt.append(train[i][1])
        trainLabel_two.append(trainLabels[i])
    # for i in range(len(train_exp)):             # validation
    #     train_fra.append(train_exp[i][0])
    #     train_opt.append(train_exp[i][1])
    #     trainLabel_two.append(trainLabels_exp[i])

    test_fra = []
    test_opt = []
    testLabels_two = []
    for i in range(len(test)):
        test_fra.append(test[i][0])
        test_opt.append(test[i][1])
        testLabels_two.append(testLabels[i])
    # for i in range(len(val_exp)):           # validation
    #     test_fra.append(val_exp[i][0])
    #     test_opt.append(val_exp[i][1])
    #     testLabels_two.append(valLabels_exp[i])

    train_generator_fra, val_generator_fra = p5_HMDB_Fra.load_Fra(train_fra, trainLabel_two,
                                                      test_fra, testLabels_two)
    train_generator_opt, val_generator_opt = p5_HMDB_Opt.load_Opt(train_opt, trainLabel_two,
                                                      test_opt, testLabels_two)
    return train_generator_fra, val_generator_fra, train_generator_opt, val_generator_opt


if __name__ == '__main__':
    # train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
    # all_files_fra, all_labels_fra = p5_HMDB_Fra.load_Fra_Two(train_files, train_labels,
    #                                                          test_files, test_labels)
    # all_files_opt, all_labels_opt = p5_HMDB_Opt.load_Opt_Two(train_files, train_labels,
    #                                                          test_files, test_labels)
    # train_generator_fra, val_generator_fra, train_generator_opt, val_generator_opt = \
    #     data_generation(all_files_fra, all_labels_fra, all_files_opt, all_labels_opt)
    #
    # HMDB_Two(train_generator_fra, val_generator_fra, train_generator_opt, val_generator_opt,
    #          (len(train_files) + len(test_files)) * 0.9, (len(train_files) + len(test_files)) * 0.1)
    # validation
    # HMDB_Two(train_generator_fra, val_generator_fra, train_generator_opt, val_generator_opt,
    #          (len(train_files) + len(test_files))*0.9*0.9, (len(train_files) + len(test_files))*0.9*0.1)


    filename_final = './DATA/HMDB_Two_final.json'
    filename_validation = './DATA/HMDB_Two_final_validation.json'
    # p4.plotting(filename_final)
    # # p4.weights('./DATA/')
    # # model = tf.keras.models.load_model('./DATA/HMDB_Two_final.h5')
    # # for i, w in enumerate(model.weights):
    # #     print(i, w.name)
    #
    p4.topAcc([filename_final, filename_validation])
    #
    # model = tf.keras.models.load_model('./DATA/HMDB_Opt_final_1.h5')
    # model.summary()