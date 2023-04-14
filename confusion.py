import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
from keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tensorflow import keras

train_files, train_labels, test_files, test_labels = load_data()
# extract_opt_save(train_files, train_labels, './optical_flow/')
# extract_opt_save(test_files, test_labels, './optical_flow/')
train_generator, val_generator = load_Opt(train_files, train_labels, test_files, test_labels)

model = tf.keras.models.load_model('./HMDB_Opt_final.h5')

# Get the number of batches in the val_generator
num_batches = len(val_generator)

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Iterate through val_generator to get test data
for batch_idx in range(num_batches):
    # Get the test data and labels for the current batch
    X_test_batch, y_test_batch = val_generator[batch_idx]

    # Predict the labels for the current batch
    y_pred_batch_probs = model.predict(X_test_batch)
    y_pred_batch = np.argmax(y_pred_batch_probs, axis=1)

    # Collect true labels and predicted labels
    true_labels.extend(np.argmax(y_test_batch, axis=1))
    predicted_labels.extend(y_pred_batch)

    # Iterate through val_generator to get test data
for batch_idx in range(len(train_generator)):
    # Get the test data and labels for the current batch
    X_test_batch, y_test_batch = train_generator[batch_idx]

    # Predict the labels for the current batch
    y_pred_batch_probs = model.predict(X_test_batch)
    y_pred_batch = np.argmax(y_pred_batch_probs, axis=1)

    # Collect true labels and predicted labels
    true_labels.extend(np.argmax(y_test_batch, axis=1))
    predicted_labels.extend(y_pred_batch)
    
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8,8), tight_layout=True)
sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, square=True, fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.savefig("b.pdf")

def preprocess_images(image_paths, target_size=(224, 224)):
    image_array_list = []

    for img_path in image_paths:
        img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        image_array_list.append(img_array)

    return np.vstack(image_array_list)

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

label_to_index = train_generator.class_indices
index_to_label = {v: k for k, v in label_to_index.items()}