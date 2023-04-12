import json
import p4
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
from keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", "riding_a_horse",
        "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]

#Build and train the model
def Stanford40():
    # Split data (skeleton code)
    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                       '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                      '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

    # Test train (skeleton code)
    train_exp, val_exp, trainLabels_exp, valLabels_exp = train_test_split(train_files, train_labels, test_size=0.1,
                                                                          stratify=train_labels, random_state=0)
    # Final train (skeleton code)
    all_files = train_files + test_files
    all_labels = train_labels + test_labels
    train, test, trainLabels, testLabels = train_test_split(all_files, all_labels, test_size=0.1,
                                                            stratify=all_labels, random_state=0)

    # Create the training and validation data
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        # dataframe=pd.DataFrame({'filename': train_exp, 'label': trainLabels_exp}),
        dataframe=pd.DataFrame({'filename': train, 'label': trainLabels}),
        directory='./Stanford40/JPEGImages/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        # dataframe=pd.DataFrame({'filename': val_exp, 'label': valLabels_exp}),
        dataframe=pd.DataFrame({'filename': test, 'label': testLabels}),
        directory='./Stanford40/JPEGImages/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = tf.keras.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (5,5), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (5,5), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(256, activation='relu', name='sta_dense'),
        layers.Dropout(0.5),
        layers.Dense(12, activation='softmax')
    ])
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    Stanford = model.fit(
            train_generator,
            steps_per_epoch=len(train) // 32,
            epochs=13,
            validation_data=val_generator,
            validation_steps=len(test) // 32)

    model.save('./DATA/Stanford40_final.h5')
    with open('./DATA/Stanford40_final.json', 'w') as f:
        json.dump(Stanford.history, f)

if __name__ == '__main__':
    Stanford40()
    filename_final = './DATA/Stanford40_final.json'

    p4.plotting(filename_final)
    # p4.comparison(filename, filename_final)

    p4.topAcc([filename_final])

    # p4.testing('./DATA/Stanford40.h5')
    # p4.weights('./DATA')
