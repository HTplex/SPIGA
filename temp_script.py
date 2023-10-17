import cv2
import numpy as np
import ht2
from glob import glob
from os.path import join,basename
import json
from pprint import pprint
from data_process_ops import *
import os
from tqdm import tqdm

data_root = "/data/datasets_v2/"
os.makedirs(join(data_root,"mouth_crops"),exist_ok=True)
video_paths = glob(join(data_root,"clip_videos_v3/*.mp4"))
for video_path in tqdm(video_paths):
    try:
        save_path = join(data_root,"mouth_crops",basename(video_path))
        fc = FaceCropper(data_root,video_path)
        process_video(video_path,save_path,fc.process_face_video)
    except:
        pass