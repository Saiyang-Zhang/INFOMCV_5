import tensorflow as tf
import os
import json
import p4
import p5_HMDB_Fra
import p5_HMDB_Opt
from keras import layers

# Build and train the model
def HMDB_Two(x_train_fra, y_train, x_test_fra, y_test, x_train_opt, x_test_opt):

    frame_model = tf.keras.models.load_model('./DATA/HMDB_final.h5')
    opticalflow_model = tf.keras.models.load_model('./DATA/HMDB_Opt_final.h5')

    # Create a two-stream CNN
    frame_input = layers.Input(shape=(224, 224, 3))
    opticalflow_input = layers.Input(shape=(16, 112, 112, 2))

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

    two_stream = two_stream_model.fit(
        x=[x_train_fra, x_train_opt],
        y=y_train,
        batch_size=32,
        epochs=15,
        validation_data=([x_test_fra, x_test_opt], y_test)
    )

    #print(two_stream_model.weights)

    if not os.path.exists('./DATA/'):
        os.makedirs('./DATA/')

    two_stream_model.save('./DATA/HMDB_Two_final.h5')
    with open('./DATA/HMDB_Two_final.json', 'w') as f:
        json.dump(two_stream.history, f)


if __name__ == '__main__':
    # train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
    # x_train_fra, y_train, x_test_fra, y_test = p5_HMDB_Fra.load_Fra(train_files, train_labels,
    #                                                                 test_files, test_labels)
    # x_train_opt, _, x_test_opt, _ = p5_HMDB_Opt.load_Opt(train_files, train_labels,
    #                                                      test_files, test_labels)
    # HMDB_Two(x_train_fra, y_train, x_test_fra, y_test, x_train_opt, x_test_opt)

    filename_final = './DATA/HMDB_Two_final.json'
    # p4.plotting(filename_final)
    # p4.weights('./DATA/')
    # model = tf.keras.models.load_model('./DATA/HMDB_Two_final.h5')
    # for i, w in enumerate(model.weights):
    #     print(i, w.name)

    p4.topAcc([filename_final])