import tensorflow as tf
import cv2
import os
import json
import numpy as np
import p4
import p5_HMDB_Fra
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras import layers, regularizers
from tqdm import tqdm

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
            "run", "shoot_bow", "smoke", "throw", "wave"]

# Extract optical flow and save it for future use
def extract_opt_save(train_files, train_labels, test_files, test_labels):
    # Parameters for optical flow calculation
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2

    # Extract optical flow from train files
    video_data_path = './video_data/'
    optical_flow_path = './train_optical_flow/'
    if not os.path.exists(optical_flow_path):
        os.makedirs(optical_flow_path)

    for i, file in tqdm(enumerate(train_files), total=len(train_files), bar_format="{percentage:3.0f}%"):
        video_path = os.path.join(video_data_path, train_labels[i], file)
        cap = cv2.VideoCapture(video_path)
        ret, frame1 = cap.read()    # the first frame
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        flow_frames = []
        for j in range(16):
            ret, frame2 = cap.read()
            if not ret:
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n,
                                                poly_sigma, 0)

            resized_flow = np.zeros((112, 112, 2))
            for n in range(2):
                resized_flow[..., n] = cv2.resize(flow[..., n], (112, 112))
            # print(resized_flow.shape)
            flow_frames.append(resized_flow)
            prvs = next

            # visualization
            # magnitude, angle = cv2.cartToPolar(resized_flow[..., 0], resized_flow[..., 1])
            # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # flow_color = cv2.applyColorMap(magnitude.astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imshow('Optical Flow', flow_color)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        stack = np.stack(flow_frames, axis=0)
        flow_path = os.path.join(optical_flow_path, file.split('.')[0] + '.npy')
        np.save(flow_path, stack)   # Save flow to a .npy file
    cap.release()

    # Extract optical flow from test files
    optical_flow_path = './test_optical_flow/'
    if not os.path.exists(optical_flow_path):
        os.makedirs(optical_flow_path)
    for i, file in tqdm(enumerate(test_files), total=len(test_files), bar_format="{percentage:3.0f}%"):
        video_path = os.path.join(video_data_path, test_labels[i], file)
        cap = cv2.VideoCapture(video_path)
        ret, frame1 = cap.read()    # the first frame
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        flow_frames = []
        for j in range(16):
            ret, frame2 = cap.read()
            if not ret:
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n,
                                                poly_sigma, 0)

            resized_flow = np.zeros((112, 112, 2))
            for n in range(2):
                resized_flow[..., n] = cv2.resize(flow[..., n], (112, 112))
            flow_frames.append(resized_flow)
            prvs = next

        stack = np.stack(flow_frames, axis=0)
        flow_path = os.path.join(optical_flow_path, file.split('.')[0] + '.npy')
        np.save(flow_path, stack)   # Save flow to a .npy file
    cap.release()

# Extract optical flow without saving it (not using)
def extract_opt(files, labels):
    video_data_path = './video_data/'
    optical_flow_path = './train_optical_flow/'

    # Parameters for optical flow calculation
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2

    # Extract optical flow from train files
    if not os.path.exists(optical_flow_path):
        os.makedirs(optical_flow_path)

    video_path = os.path.join(video_data_path, labels, files)
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()    # the first frame
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    flow_frames = []
    for j in range(16):
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n,
                                            poly_sigma, 0)

        resized_flow = np.zeros((112, 112, 2))
        for n in range(2):
            resized_flow[..., n] = cv2.resize(flow[..., n], (112, 112))
        #print(resized_flow.shape)
        flow_frames.append(resized_flow)
        prvs = next

    stack = np.stack(flow_frames, axis=0)
    cap.release()
    return stack

# Load and create training data
def load_Opt(train_files, train_labels, test_files, test_labels):
    x_train = np.empty((0, 16, 112, 112, 2))
    y_train = []

    print("Loading train files:")
    for i, file in tqdm(enumerate(train_files), total=len(train_files), bar_format="{percentage:3.0f}%"):
        flow_path = os.path.join('./train_optical_flow/', file.split('.')[0] + '.npy')
        flow = np.load(flow_path)
        # print(f"Shape of flow {i}: {flow.shape}")
        # flow = extract_opt(file, train_labels[i])
        x_train = np.concatenate((x_train, np.expand_dims(flow, axis=0)), axis=0)

        # Create one-hot encoded label vector for each file based on its class
        label = train_labels[i]
        label_idx = keep_hmdb51.index(label)  # get the index of the class in the class list
        label_vec = np.zeros(len(keep_hmdb51))
        label_vec[label_idx] = 1
        y_train.append(label_vec)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 2)).reshape(x_train.shape)
    y_train = np.array(y_train)
    # print(x_train, len(x_train))
    # print(y_train, len(y_train))

    print("Loading test files:")
    x_test = np.empty((0, 16, 112, 112, 2))
    y_test = []
    for i, file in tqdm(enumerate(test_files), total=len(test_files), bar_format="{percentage:3.0f}%"):
        flow_path = os.path.join('./test_optical_flow/', file.split('.')[0] + '.npy')
        flow = np.load(flow_path)
        # flow = extract_opt(file, test_labels[i])
        x_test = np.concatenate((x_test, np.expand_dims(flow, axis=0)), axis=0)
        # Create one-hot encoded label vector for each file based on its class
        label = test_labels[i]
        label_idx = keep_hmdb51.index(label)  # get the index of the class in the class list
        label_vec = np.zeros(len(keep_hmdb51))
        label_vec[label_idx] = 1
        y_test.append(label_vec)

    x_test = scaler.transform(x_test.reshape(-1, 2)).reshape(x_test.shape)
    y_test = np.array(y_test)
    # print(x_test, len(x_test))
    # print(y_test, len(y_test))
    return x_train, y_train, x_test, y_test

# Build and train the model
def HMDB_Opt(x_train, y_train, x_test, y_test):
    model = tf.keras.Sequential([
        layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', input_shape=(16, 112, 112, 2),
                      kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Conv3D(32, kernel_size=(2, 2, 2), strides=(1, 1, 1), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Conv3D(64, kernel_size=(2, 2, 2), strides=(1, 1, 1), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01)),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     bias_regularizer=regularizers.l1(0.01)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     bias_regularizer=regularizers.l1(0.01)),
        layers.Dropout(0.5),
        layers.Dense(12, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),
                             metrics=['accuracy'])
    # HMDB_Opt = model.fit(x_train, y_train, epochs=15, validation_split=0.1)
    HMDB_Opt = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

    if not os.path.exists('./DATA/'):
        os.makedirs('./DATA/')
    model.save('./DATA/HMDB_Opt_final.h5')
    with open('./DATA/HMDB_Opt_final.json', 'w') as f:
        json.dump(HMDB_Opt.history, f)


if __name__ == '__main__':
    # train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
    # extract_opt_save(train_files, train_labels, test_files, test_labels)
    # x_train, y_train, x_test, y_test = load_Opt(train_files, train_labels, test_files, test_labels)
    # HMDB_Opt(x_train, y_train, x_test, y_test)
    filename_final = './DATA/HMDB_Opt_final.json'
    # p4.plotting(filename_final)
    # p4.comparison(filename, filename_final)

    # model = tf.keras.models.load_model('./DATA/HMDB_Opt_final.h5')
    # for i, w in enumerate(model.weights):
    #     print(i, w.name)

    p4.topAcc([filename_final])
