from keras import layers
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import fashion_mnist
from PIL import Image
from sklearn.metrics import confusion_matrix

import os
import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

def baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    baseline = model.fit(trainSet_images, trainSet_labels, 
                         batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/baseline_model.h5')
    with open('./data/baseline_history.json', 'w') as f:
        json.dump(baseline.history, f)


# increase number of kernels
def variant1(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # Train the model
    # variant1 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
    #                      validation_data=(validSet_images, validSet_labels))
    variant1 = model.fit(trainSet_images, trainSet_labels, 
                         batch_size=128, epochs=15,
                         validation_split=0.2)

    # Evaluate both models on the test set
    model.evaluate(validSet_images, validSet_labels)

    # Save the model and training history
    # model.save('./data/variant1_model.h5')
    # with open('./data/variant1_history.json', 'w') as f:
    #     json.dump(variant1.history, f)
    model.save('./data/best1_model.h5')
    with open('./data/best1_history.json', 'w') as f:
        json.dump(variant1.history, f)


# change activation
def variant2(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='valid', activation='tanh'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # Train the model
    variant2 = model.fit(trainSet_images, trainSet_labels, 
                         batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant2_model.h5')
    with open('./data/variant2_history.json', 'w') as f:
        json.dump(variant2.history, f)


# add Batch Normalization
def variant3(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])


    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    variant3 = model.fit(trainSet_images, trainSet_labels, 
                         batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant3_model.h5')
    with open('./data/variant3_history.json', 'w') as f:
        json.dump(variant3.history, f)


# change padding
def variant4(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    # variant4 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
    #                      validation_data=(validSet_images, validSet_labels))
    variant4 = model.fit(trainSet_images, trainSet_labels, 
                         batch_size=128, epochs=15,
                         validation_split=0.2)

    # Evaluate both models on the test set
    model.evaluate(validSet_images, validSet_labels)

    # Save the model and training history
    # model.save('./data/variant4_model.h5')
    # with open('./data/variant4_history.json', 'w') as f:
    #     json.dump(variant4.history, f)
    model.save('./data/best2_model.h5')
    with open('./data/best2_history.json', 'w') as f:
        json.dump(variant4.history, f)


# plotting for a single model
def plotting(filename):
    # Load the history
    with open(filename, 'r') as f:
        history = json.load(f)

    # Plot the training/validation loss per epoch
    plt.plot(history['loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename[:-5]+'_loss.png')
    plt.show()
    plt.close()

    # Plot the training/validation accuracy per epoch
    plt.plot(history['accuracy'], label='Train accuracy')
    plt.plot(history['val_accuracy'], label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename[:-5] + '_acc.png')
    plt.show()
    plt.close()


# plotting for pairwise comparison
def comparison(filename1, filename2):
    # Load the history
    with open(filename1, 'r') as f:
        history1 = json.load(f)
    with open(filename2, 'r') as f:
        history2 = json.load(f)

    # Plot the training/validation loss per epoch
    plt.plot(history1['loss'], label='Training loss 1', color='C0')
    plt.plot(history1['val_loss'], label='Validation loss 1', color='C1')
    plt.plot(history2['loss'], label='Training loss 2',linestyle='--', color='C0')
    plt.plot(history2['val_loss'], label='Validation loss 2',linestyle='--', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename1[:-5] + '_' + filename2[7:-5] + '_loss.png')
    plt.show()
    plt.close()

    # Plot the training/validation accuracy per epoch
    plt.plot(history1['accuracy'], label='Training accuracy 1', color='C0')
    plt.plot(history1['val_accuracy'], label='Validation accuracy 1', color='C1')
    plt.plot(history2['accuracy'], label='Training accuracy 2',linestyle='--', color='C0')
    plt.plot(history2['val_accuracy'], label='Validation accuracy 2', linestyle='--', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename1[:-5] + '_' + filename2[7:-5] + '_acc.png')
    plt.show()
    plt.close()

# compare to choose the best model
def best_comparison(filename):
    # Load the history
    history = []
    for fn in filename:
        with open(fn, 'r') as f:
            history.append(json.load(f))

    color = ['C0', 'C1', 'C2', 'C3', 'C4']
    # Plot the training/validation loss per epoch
    for i in range(len(history)):
        plt.plot(history[i]['val_loss'], label= f'Validation loss {i}', color=color[i])
        avg_val_loss = np.mean(history[i]['val_loss'])
        min_val_loss = np.min(history[i]['val_loss'])
        print(f'Average val_loss {i}: {avg_val_loss:.4f}')
        print(f'Min val_loss {i}: {min_val_loss:.4f}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('./data/best_val_loss.png')
    plt.show()
    plt.close()

    for i in range(len(history)):
        plt.plot(history[i]['val_accuracy'], label= f'Validation accuracy {i}', color=color[i])
        avg_val_acc = np.mean(history[i]['val_accuracy'])
        max_val_acc = np.max(history[i]['val_accuracy'])
        print(f'Average val_accuracy {i}: {avg_val_acc:.4f}')
        print(f'Max val_accuracy {i}: {max_val_acc:.4f}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig('./data/best_val_accuracy.png')
    plt.show()
    plt.close()

    # for i in range(len(history)):
    #     plt.plot(history[i]['loss'], label=f'Training loss {i}', color=color[i])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('./data/best_training_loss.png')
    # plt.show()
    # plt.close()

    # for i in range(len(history)):
    #     plt.plot(history[i]['accuracy'], label=f'Training accuracy {i}', color=color[i])
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig('./data/best_training_accuracy.png')
    # plt.show()
    # plt.close()


# prediction testing
def testing(filename):
    model = tf.keras.models.load_model(filename)

    # Preprocess the image
    image = Image.open("./images.jpeg") #.convert('L')
    image = image.resize((224, 224))    # Resize the image to (28, 28)
    image = np.array(image)
    img_arr = image[np.newaxis, ...] / 255.0
    # image = image.reshape(1, 224, 224, 3)  # Reshape the image to (1, 28, 28, 1) to match the model's input shape

    # Make a prediction
    prediction = model.predict(img_arr)
    predicted_class = np.argmax(prediction[0])
    plt.imshow(img_arr[0])
    plt.show()
    print("Predicted class:", predicted_class)


# choice task 1: confusion matrix
def confusion(model, images, labels, save_path):
    pred = np.argmax(model.predict(images), axis=1)
    cm = confusion_matrix(labels, pred)
    plt.figure(figsize=(8,8), tight_layout=True)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, square=True, fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.savefig(save_path)

# choice task 2: decrease the learning rate
def learningRateDec(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    learningRateDec = model.fit(trainSet_images, trainSet_labels,
                         batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels),
                         callbacks=[lr_scheduler])

    # Convert the learning rate to a Python float
    lr_value = float(np.array(learningRateDec.history['lr'])[0])
    learningRateDec_new = learningRateDec.history.copy()
    learningRateDec_new['lr'] = lr_value

    # Save the model and training history
    model.save('./data/learningRateDec_model.h5')
    with open('./data/learningRateDec_history.json', 'w') as f:
        json.dump(learningRateDec_new, f)


# choice task 4: create output layers
def outputs(model, images, savepath):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])

    features = extractor(images[:1])

    fig = plt.figure(figsize=(50, 20), tight_layout=True)
    n = len(features)
    p = 1
    for i in range(n):
        if len(features[i].shape) == 4:
            p += 1
    for i, feature in enumerate(features):
        if len(feature.shape) == 4:
            plt.subplot(1, p, i + 1)
            plt.imshow(feature[0, :, :, 0], cmap='viridis')
        if i == n - 1:
            plt.subplot(1, p, p)
            plt.imshow(feature, cmap='viridis')
        plt.axis('off')
        plt.title(model.layers[i].name, fontsize=36)
    plt.savefig(savepath)

# display weights
def weights(filename):
    models = []
    models_name = []

    for fn in os.listdir(filename):
        if fn.endswith(".h5"):
            model_path = os.path.join(filename, fn)
            model = tf.keras.models.load_model(model_path)
            print("\n", fn[:-3], ":")
            model.summary()
            models.append(model)
            models_name.append(fn[:-3])

    # Print the weights of each model
    # for i in range(len(models)):
    #     print(models_name[i], ":")
    #     for layer in models[i].layers:
    #         print(layer.name, ":", layer.get_weights())

# get top-1 accuracy
def topAcc(filename):
    # Load the history
    history = []
    models_name = []
    for fn in filename:
        with open(fn, 'r') as f:
            history.append(json.load(f))
            models_name.append(fn[7:-5])

    for i in range(len(history)):
        topAcc_tra = max(history[i]['accuracy'])
        topAcc_val = max(history[i]['val_accuracy'])
        print(models_name[i], ':\nTop-1 training accuracy:', topAcc_tra*100,
              ', Top-1 validation accuracy:', topAcc_val*100)
        topLoss_tra = min(history[i]['loss'])
        topLoss_val = min(history[i]['val_loss'])
        print(models_name[i], ':\nTop-1 training loss:', topLoss_tra,
              ', Top-1 validation loss:', topLoss_val)

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    """Get the best models"""
    # create new sets 0.8/0.2
    train_num = int(0.8 * len(train_images))
    trainSet_images = train_images[:train_num]
    trainSet_labels = train_labels[:train_num]

    validSet_images = train_images[train_num:]
    validSet_labels = train_labels[train_num:]

    # Normalize pixel values (0 and 1)
    trainSet_images = trainSet_images.astype('float32') / 255
    validSet_images = validSet_images.astype('float32') / 255
    # Reshape the images to 28x28x1
    trainSet_images = np.expand_dims(trainSet_images, axis=-1)
    validSet_images = np.expand_dims(validSet_images, axis=-1)
    # Convert labels
    trainSet_labels = keras.utils.to_categorical(trainSet_labels, 10)
    validSet_labels = keras.utils.to_categorical(validSet_labels, 10)

    # baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename0 = './data/baseline_history.json'
    # plotting(filename0)

    # variant1(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename1 = './data/variant1_history.json'
    # plotting(filename1)
    # comparison(filename0, filename1)

    # variant2(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename2 = './data/variant2_history.json'
    # plotting(filename2)
    # comparison(filename0, filename2)

    # variant3(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename3 = './data/variant3_history.json'
    # plotting(filename3)
    # comparison(filename0, filename3)

    # variant4(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename4= './data/variant4_history.json'
    # plotting(filename4)
    # comparison(filename0, filename4)

    # choose the best performaing model (result: variant1 and variant4)
    # best_comparison([filename0, filename1, filename2, filename3, filename4])

    """Train on the best models"""
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Combine the training and validation sets
    train_images_all = tf.concat([train_images, test_images], axis=0)
    train_labels_all = tf.concat([train_labels, test_labels], axis=0)

    # Convert labels
    train_labels_all = keras.utils.to_categorical(train_labels_all, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    # variant1(train_images_all, train_labels_all, test_images, test_labels)
    filename_b1 = './data/best1_history.json'
    # plotting(filename_b1)
    # variant4(train_images_all, train_labels_all, test_images, test_labels)
    filename_b2 = './data/best2_history.json'
    # plotting(filename_b2)
    # comparison(filename_b1, filename_b2)


    # testing('./data/best1_model.h5')

    # learningRateDec(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename_lr = './data/learningRateDec_history.json'
    # plotting(filename_lr)
    # comparison(filename0, filename_lr)

    weights('./weights')
    # topAcc([filename0, filename1, filename2, filename3, filename4])

    """delete later"""
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()'./data/baseline_model.h5')