import tensorflow as tf
import cv2
import os
import json
import numpy as np
import p4
import p5_HMDB_Fra
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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


train_files, train_labels, test_files, test_labels = p5_HMDB_Fra.load_data()
extract_opt_save(train_files, train_labels, './optical_flow/')
extract_opt_save(test_files, test_labels, './optical_flow/')

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
