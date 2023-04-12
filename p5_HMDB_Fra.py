import tensorflow as tf
import cv2
import glob
import os
import json
import numpy as np
import p4
import pandas as pd
from keras.optimizers import Adam
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

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

    # Extract middle frames from train files
    middle_frame_path = './train_middle_frames/'
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

    # Extract middle frames from test files
    middle_frame_path = './test_middle_frames/'
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
    train_files_img = []
    test_files_img = []
    for i, file in enumerate(train_files):
         train_files_img.append(os.path.join(file.split('.')[0] + '.jpg'))
    for i, file in enumerate(test_files):
         test_files_img.append(os.path.join(file.split('.')[0] + '.jpg'))

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_files_img, 'label': train_labels}),
        directory='./train_middle_frames/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': test_files_img, 'label': test_labels}),
        directory='./test_middle_frames/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    # X_fra, y_fra = train_generator[0]
    # print(len(y_fra))
    return train_generator, val_generator

# Build and train the model
def HMDB_Fra(train_generator, val_generator, len_train, len_val):
    model = tf.keras.models.load_model('./DATA/Stanford40_final.h5')
    # transfer learning: freeze the weights of layers except for the last two layers
    # for layer in model.layers[:-2]:
    #     layer.trainable = False

    new_output = layers.Dense(12, activation='softmax', name='fra_dense')(model.layers[-2].output)
    fine_tuned_model = tf.keras.models.Model(inputs=model.inputs, outputs=new_output)
    fine_tuned_model.summary()

    fine_tuned_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    HMDB = model.fit(
            train_generator,
            steps_per_epoch=len_train // 32,
            epochs=13,
            validation_data=val_generator,
            validation_steps=len_val // 32,
            callbacks=[lr_scheduler])

    # Convert the learning rate to a Python float
    lr_value = float(np.array(HMDB.history['lr'])[0])
    HMDB_new = HMDB.history.copy()
    HMDB_new['lr'] = lr_value

    fine_tuned_model.save('./DATA/HMDB_final.h5')
    with open('./DATA/HMDB_final.json', 'w') as f:
        # json.dump(HMDB.history, f)
        json.dump(HMDB_new, f)


if __name__ == '__main__':
    # train_files, train_labels, test_files, test_labels = load_data()
    # # # extract_frame(train_files, train_labels, test_files, test_labels)
    # train_generator, val_generator = load_Fra(train_files, train_labels, test_files, test_labels)
    # HMDB_Fra(train_generator, val_generator, len(train_files), len(test_files))

    filename_final = './DATA/HMDB_final.json'
    # p4.plotting(filename_final)

    p4.topAcc([filename_final])

    # p4.comparison(filename, filename_final)

    # model = tf.keras.models.load_model('./DATA/HMDB_final.h5')
    # model.summary()