"""
take cropped video frames and subtitles, generate:
    training image: 384*384 images each consists of n frames of tile of 16*16 video frames
    training label: transcript of the video within those frames
video frames
"""
# load video frames into numpy array
import cv2
import numpy as np
from glob import glob
from os.path import join,basename
import json
from pprint import pprint
from data_process_ops import *
import os

data_root = "/data/datasets_v2/"
video_paths = glob(join(data_root,"mouth_crops/*.mp4"))
sample_video = video_paths[0]
label_path = join(data_root,"labels_v3",basename(sample_video).replace(".mp4",".json"))
with open(label_path) as f:
    label = json.load(f)
text = label["text_postprocessed"]
timestamps = label["time_stamp"]
frames = video_to_3d_np_array(sample_video)
cap = cv2.VideoCapture(sample_video)
fps = cap.get(cv2.CAP_PROP_FPS)

time_slide_window_step = 1 # second
time_slide_window_size = 384*384/16/16 # frames

# generate training data
for start_time_stamp in range(0,len(frames),time_slide_window_step*fps):
    end_time_stamp = start_time_stamp + time_slide_window_size
    if end_time_stamp > len(frames):
        break
    # get frames
    training_frames = frames[start_time_stamp:end_time_stamp]
    # get text
    training_text = ""
    for i in range(len(timestamps)):
        if timestamps[i][0] >= start_time_stamp/fps and timestamps[i][1] <= end_time_stamp/fps:
            training_text += text[i]
    print(training_text)
    print(len(training_frames))
    # save frames and text
    # np.save(join(data_root,"training_frames",basename(sample_video).replace(".mp4","_%d_%d.npy"%(start_time_stamp,end_time_stamp))),training_frames)
    # with open(join(data_root,"training_text",basename(sample_video).replace(".mp4","_%d_%d.txt"%(start_time_stamp,end_time_stamp))),"w") as f:
    #     f.write(training_text)


