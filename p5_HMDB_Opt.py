import tensorflow as tf
import cv2
import os
import json
import numpy as np
import p4
import p5_HMDB_Fra
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers, regularizers
from tqdm import tqdm
from keras.utils import to_categorical
from keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Generate batches of data
class OpticalFlowSequence(Sequence):
    def __init__(self, stack_dir, file_names, labels, batch_size, shuffle=True):
        self.stack_dir = stack_dir
        self.file_names = file_names
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.labels))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i in batch_indexes:
            # load the optical flow stack for sample i from disk
            file_name = self.file_names[i]
            stack = np.load(os.path.join(self.stack_dir, file_name))
            batch_x.append(stack)
            batch_y.append(self.labels[i])
        if self.shuffle:
            # shuffle the batch data
            zipped = list(zip(batch_x, batch_y))
            np.random.shuffle(zipped)
            batch_x, batch_y = zip(*zipped)
        return np.array(batch_x), np.array(batch_y)

    def __str__(self):
        num_samples = len(self.file_names)
        num_batches = len(self)
        return f"OpticalFlowSequence with {num_samples} samples in {num_batches} batches of size {self.batch_size}"


keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
            "run", "shoot_bow", "smoke", "throw", "wave"]

# Extract optical flow and save it for future use
def extract_opt_save(files, labels, optical_flow_path):
    # Parameters for optical flow calculation
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2

    video_data_path = './video_data/'
    if not os.path.exists(optical_flow_path):
        os.makedirs(optical_flow_path)

    # Extract optical flow from files
    for i, file in tqdm(enumerate(files), total=len(files),
                        bar_format="{percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt}"):
        video_path = os.path.join(video_data_path, labels[i], file)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = frame_count // 2
        start_index = middle_frame_index - 8     # Set the starting frame around the middle
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)

        ret, frame1 = cap.read()
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
            resized_flow = np.zeros((224, 224, 2))
            for n in range(2):
                resized_flow[..., n] = cv2.resize(flow[..., n], (224, 224))
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
        np.save(flow_path, stack)   # Save flow stack to a .npy file
    cap.release()


# Load and create training data
def load_Opt(train_files, train_labels, test_files, test_labels):
    train_files_opt = []
    test_files_opt = []
    for i, file in enumerate(train_files):
         train_files_opt.append(os.path.join(file.split('.')[0] + '.npy'))
    for i, file in enumerate(test_files):
         test_files_opt.append(os.path.join(file.split('.')[0] + '.npy'))

    label_dict = {label: index for index, label in enumerate(keep_hmdb51)}
    train_label_indices = [label_dict[label] for label in train_labels]
    train_onehot_labels = to_categorical(train_label_indices)
    test_label_indices = [label_dict[label] for label in test_labels]
    test_onehot_labels = to_categorical(test_label_indices)

    train_exp, val_exp, trainLabels_exp, valLabels_exp = train_test_split(train_files_opt, train_onehot_labels,
                                                                          test_size=0.1,stratify=train_labels,
                                                                          random_state=0)

    all_files = train_files_opt + test_files_opt
    all_labels = np.concatenate((train_onehot_labels, test_onehot_labels), axis=0)
    # for HMDB_Opt train-test
    # train, test, trainLabels, testLabels = train_test_split(all_files, all_labels, test_size=0.1,
    #                                                         stratify=all_labels, random_state=0)
    # for two-stream only
    train, test, trainLabels, testLabels = train_test_split(all_files, all_labels, test_size=0.1, shuffle=False)

    data_dir = './optical_flow/'
    batch_size = 32
    # validation
    # train_flow_sequence = OpticalFlowSequence(data_dir, train_exp, trainLabels_exp,
    #                                           batch_size, True)
    # print(train_flow_sequence)
    # test_flow_sequence = OpticalFlowSequence(data_dir, val_exp, valLabels_exp,
    #                                           batch_size, False)
    # print(test_flow_sequence)

    # for HMDB_Opt train-test
    # train_flow_sequence = OpticalFlowSequence(data_dir, train, trainLabels,
    #                                           batch_size, True)
    # for two-stream only
    train_flow_sequence = OpticalFlowSequence(data_dir, train, trainLabels,
                                              batch_size, False)
    print(train_flow_sequence)
    test_flow_sequence = OpticalFlowSequence(data_dir, test, testLabels,
                                              batch_size, False)
    print(test_flow_sequence)

    # x_test, y_test = test_flow_sequence[0]
    # print(x_test.shape)
    return train_flow_sequence, test_flow_sequence


def load_Opt_Two(train_files, train_labels, test_files, test_labels):
    train_files_opt = []
    test_files_opt = []
    for i, file in enumerate(train_files):
         train_files_opt.append(os.path.join(file.split('.')[0] + '.npy'))
    for i, file in enumerate(test_files):
         test_files_opt.append(os.path.join(file.split('.')[0] + '.npy'))

    label_dict = {label: index for index, label in enumerate(keep_hmdb51)}
    train_label_indices = [label_dict[label] for label in train_labels]
    train_onehot_labels = to_categorical(train_label_indices)
    test_label_indices = [label_dict[label] for label in test_labels]
    test_onehot_labels = to_categorical(test_label_indices)

    all_files = train_files_opt + test_files_opt
    all_labels = np.concatenate((train_onehot_labels, test_onehot_labels), axis=0)

    return all_files, all_labels


# Build and train the model
def HMDB_Opt(train_generator, val_generator, len_train, len_val):
    model = tf.keras.Sequential([
        layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', input_shape=(16, 224, 224, 2)),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Conv3D(32, kernel_size=(2, 2, 2), strides=(1, 1, 1), activation='relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Conv3D(64, kernel_size=(2, 2, 2), strides=(1, 1, 1), activation='relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(12, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    HMDB_Opt = model.fit(
            train_generator,
            # steps_per_epoch=(len_train * 0.9) // 32,        # validation
            steps_per_epoch=(len_train+len_val) * 0.9 // 32,
            epochs=13,
            validation_data=val_generator,
            # validation_steps=(len_train * 0.1) // 32)       # validation
            validation_steps=(len_train+len_val) * 0.1 // 32)

    if not os.path.exists('./DATA/'):
        os.makedirs('./DATA/')
    model.save('./DATA/HMDB_Opt_final.h5')
    with open('./DATA/HMDB_Opt_final.json', 'w') as f:
        json.dump(HMDB_Opt.history, f)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
    # extract_opt_save(train_files, train_labels, './optical_flow/')
    # extract_opt_save(test_files, test_labels, './optical_flow/')
    train_generator, val_generator = load_Opt(train_files, train_labels, test_files, test_labels)
    HMDB_Opt(train_generator, val_generator, len(train_files), len(test_files))

    filename_final = './DATA/HMDB_Opt_final.json'
    filename_validation = './DATA/HMDB_Opt_final_validation.json'
    # p4.plotting(filename_final)
    # p4.comparison(filename, filename_final)
    #
    model = tf.keras.models.load_model('./DATA/HMDB_Opt_final.h5')
    model.summary()
    # # for i, w in enumerate(model.weights):
    # #     print(i, w.name)
    #
    p4.topAcc([filename_final, filename_validation])
