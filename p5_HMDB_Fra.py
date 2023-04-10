import tensorflow as tf
import cv2
import glob
import os
import json
import numpy as np
import p4
from keras.optimizers import Adam
from keras import layers

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
            "run", "shoot_bow", "smoke", "throw", "wave"]

# Load data from HMDB51 (skeleton code)
def load_data():
    TRAIN_TAG, TEST_TAG = 1, 2
    train_files, test_files = [], []
    train_labels, test_labels = [], []
    split_pattern_name = f"*test_split1.txt"
    split_pattern_path = os.path.join('test_train_splits', split_pattern_name)
    annotation_paths = glob.glob(split_pattern_path)
    annotation_paths = [s.replace('\\', '/') for s in annotation_paths]

    for filepath in annotation_paths:
        class_name = '_'.join(filepath.split('/')[-1].split('_')[:-2])
        if class_name not in keep_hmdb51:
            continue  # skipping the classes that we won't use.
        with open(filepath) as fid:
            lines = fid.readlines()
        for line in lines:
            video_filename, tag_string = line.split()
            tag = int(tag_string)
            if tag == TRAIN_TAG:
                train_files.append(video_filename)
                train_labels.append(class_name)
            elif tag == TEST_TAG:
                test_files.append(video_filename)
                test_labels.append(class_name)

    print(f'Train files ({len(train_files)})')      #:\n\t{train_files}')
    print(f'Train labels ({len(train_labels)})')    #:\n\t{train_labels}\n')
    print(f'Test files ({len(test_files)})')        #:\n\t{test_files}')
    print(f'Test labels ({len(test_labels)})')      #:\n\t{test_labels}\n')
    return train_files, train_labels, test_files, test_labels

# Extract middle frame and save it
def extract_frame(train_files, train_labels, test_files, test_labels):
    video_data_path = './video_data/'
    middle_frame_path = './train_middle_frames/'

    # Extract middle frames from train files
    if not os.path.exists(middle_frame_path):
        os.makedirs(middle_frame_path)
    lb_num = 0
    for file in train_files:
        video_path = os.path.join(video_data_path,train_labels[lb_num], file)
        print(video_path)
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = vidcap.read()
        frame_path = os.path.join(middle_frame_path, file.split('.')[0] + '.jpg')
        cv2.imwrite(frame_path, frame)
        lb_num += 1

    middle_frame_path = './test_middle_frames/'
    # Extract middle frames from test files
    if not os.path.exists(middle_frame_path):
        os.makedirs(middle_frame_path)
    lb_num = 0
    for file in test_files:
        video_path = os.path.join(video_data_path, test_labels[lb_num], file)
        print(video_path)
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = vidcap.read()
        frame_path = os.path.join(middle_frame_path, file.split('.')[0] + '.jpg')
        cv2.imwrite(frame_path, frame)
        lb_num += 1

# Load and create training data
def load_Fra(train_files, train_labels, test_files, test_labels):
    x_train = []
    y_train = []
    for i, file in enumerate(train_files):
        frame_path = os.path.join('./train_middle_frames/', file.split('.')[0] + '.jpg')
        # print(frame_path)
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (224, 224))
        x_train.append(img)

        # Create one-hot encoded label vector for each file based on its class
        label = train_labels[i]
        label_idx = keep_hmdb51.index(label)  # get the index of the class in the class list
        label_vec = np.zeros(len(keep_hmdb51))
        label_vec[label_idx] = 1
        y_train.append(label_vec)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # print(x_train[0], len(x_train))
    # print(y_train[0], len(y_train))

    x_test = []
    y_test = []
    for i, file in enumerate(test_files):
        frame_path = os.path.join('./test_middle_frames/', file.split('.')[0] + '.jpg')
        # print(frame_path)
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (224, 224))
        x_test.append(img)

        # Create one-hot encoded label vector for each file based on its class
        label = test_labels[i]
        label_idx = keep_hmdb51.index(label)  # get the index of the class in the class list
        label_vec = np.zeros(len(keep_hmdb51))
        label_vec[label_idx] = 1
        y_test.append(label_vec)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

# Build and train the model
def HMDB_Fra(x_train, y_train, x_test, y_test):
    model = tf.keras.models.load_model('./DATA/Stanford40_final.h5')
    # transfer learning: freeze the weights of layers except for the last two layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # transfer learning: freeze all layers
    # model.trainable = False

    new_output = layers.Dense(12, activation='softmax', name='fra_dense')(model.layers[-2].output)
    fine_tuned_model = tf.keras.models.Model(inputs=model.inputs, outputs=new_output)
    fine_tuned_model.summary()

    fine_tuned_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    #HMDB = fine_tuned_model.fit(x_train, y_train, epochs=15, validation_split=0.1)
    HMDB = fine_tuned_model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

    fine_tuned_model.save('./DATA/HMDB_final.h5')
    with open('./DATA/HMDB_final.json', 'w') as f:
        json.dump(HMDB.history, f)


if __name__ == '__main__':
    # train_files, train_labels, test_files, test_labels = load_data()
    # extract_frame(train_files, train_labels, test_files, test_labels)
    # x_train, y_train, x_test, y_test = load_Fra(train_files, train_labels, test_files, test_labels)
    # HMDB_Fra(x_train, y_train, x_test, y_test)
    filename_final = './DATA/HMDB_final.json'
    # p4.plotting(filename_final)

    p4.topAcc([filename_final])