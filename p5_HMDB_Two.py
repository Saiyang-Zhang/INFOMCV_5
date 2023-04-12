import tensorflow as tf
import os
import json
import p4
import p5_HMDB_Fra
import numpy as np
import p5_HMDB_Opt
from keras import layers
from keras.utils import Sequence

class TwoStreamSequence(Sequence):
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
        # print("y_fra", y_fra)
        # print("y_opt", y_opt)
        # Combine the two input streams
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
    train_two_stream = TwoStreamSequence(train_generator_fra, train_generator_opt, True)
    test_two_stream = TwoStreamSequence(val_generator_fra, val_generator_opt, False)

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


if __name__ == '__main__':
    # train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
    # train_generator_fra, val_generator_fra = p5_HMDB_Fra.load_Fra(train_files, train_labels,
    #                                                               test_files, test_labels)
    # train_generator_opt, val_generator_opt = p5_HMDB_Opt.load_Opt(train_files, train_labels,
    #                                                               test_files, test_labels)
    #
    # HMDB_Two(train_generator_fra, val_generator_fra, train_generator_opt, val_generator_opt,
    #          len(train_files), len(test_files))
    #
    filename_final = './DATA/HMDB_Two_final.json'
    # p4.plotting(filename_final)
    # # p4.weights('./DATA/')
    # # model = tf.keras.models.load_model('./DATA/HMDB_Two_final.h5')
    # # for i, w in enumerate(model.weights):
    # #     print(i, w.name)
    #
    p4.topAcc([filename_final])
    #
    # model = tf.keras.models.load_model('./DATA/HMDB_Opt_final_1.h5')
    # model.summary()