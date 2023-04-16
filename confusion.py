import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import p5_HMDB_Fra
import p5_HMDB_Opt
import p5_Stanford
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Confusion_matrix for HMDB51 optical flow
train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
train_generator, val_generator = p5_HMDB_Opt.load_Opt(train_files, train_labels, test_files, test_labels)

model = tf.keras.models.load_model('./DATA/HMDB_Opt_final.h5')

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
plt.figure(figsize=(8, 8), tight_layout=True)
sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, square=True, fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.savefig("HMDB51.pdf")


# Confusion_matrix for Stanford40 frames
with open('Stanford40/ImageSplits/train.txt', 'r') as f:
    train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                   '_'.join(file_name.split('_')[:-1]) in p5_Stanford.keep_stanford40]
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

with open('Stanford40/ImageSplits/test.txt', 'r') as f:
    test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if
                  '_'.join(file_name.split('_')[:-1]) in p5_Stanford.keep_stanford40]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
all_files = train_files + test_files
all_labels = train_labels + test_labels

train, test, trainLabels, testLabels = train_test_split(all_files, all_labels, test_size=0.1,
                                                        stratify=all_labels, random_state=0)

model = tf.keras.models.load_model('./DATA/Stanford40_final.h5')
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


def preprocess_images(image_paths, target_size=(224, 224)):
    image_array_list = []

    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        image_array_list.append(img_array)

    return np.vstack(image_array_list)

def confusion(model, images, labels, save_path):
    pred = np.argmax(model.predict(images), axis=1)
    cm = confusion_matrix(labels, pred)
    plt.figure(figsize=(8,8), tight_layout=True)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, square=True, fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.savefig(save_path)

files = ['Stanford40/JPEGImages/' + file for file in all_files]
imgs = preprocess_images(files)


label_to_index = train_generator.class_indices
index_to_label = {v: k for k, v in label_to_index.items()}
labels = np.array([label_to_index[label] for label in all_labels])

confusion(model, imgs, labels, "Stanford40.pdf")