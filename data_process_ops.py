import cv2
import numpy as np
import ht2
import json
from os.path import join,basename

def crop_face(image,face_landmarks,mode='face',output_size=(256,256),draw_points=False):
        
    if len(face_landmarks) == 0:
        return np.zeros((256,256,3), dtype=np.uint8)
    if mode == 'mouth':
        landmarks = np.array(face_landmarks[0]['landmarks'][76:-2])
        mouth_width = landmarks[6][0] - landmarks[0][0]
        face_bbox = [landmarks[0][0], landmarks[0][1]-mouth_width//2,
                    landmarks[6][0], landmarks[6][1]+mouth_width//2]
    elif mode == 'face':
        landmarks = np.array(face_landmarks[0]['landmarks'])
        face_bbox = [np.min(landmarks[:,0]), np.min(landmarks[:,1]),
                    np.max(landmarks[:,0]), np.max(landmarks[:,1])]
    face_bbox = [int(i) for i in face_bbox]
    face_image = image[face_bbox[1]:face_bbox[3],
                       face_bbox[0]:face_bbox[2]]
    # resize
    face_image = cv2.resize(face_image, (256,256))

    # add points
    transformed_landmarks = [[
        (points[0]-face_bbox[0])*(256/(face_bbox[2]-face_bbox[0])),
        (points[1]-face_bbox[1])*(256/(face_bbox[3]-face_bbox[1]))]
        for points in landmarks]
    # draw dots for mouth
    if draw_points:
        for idx,points in enumerate(transformed_landmarks):
            cv2.circle(face_image, (int(points[0]),int(points[1])), 2, (0,idx*16,0), -1)
    face_image = cv2.resize(face_image, output_size)
    return face_image

def process_video(input_path,output_path,process_func):
    """
    load video and process each frame using process_func,
    save the video after that.
    """
    # Load the video
    cap = cv2.VideoCapture(input_path)
    
    # Check if video loaded successfully
    if not cap.isOpened():
        raise ValueError("invalid video: can not process by cv2.VideoCapture")
    
    # Get video dimensions
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # get output size by process the first frame
    output_h,output_w = input_h,input_w



    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            break
    
        # Crop the frame (example: get the central half of the frame)
        processed_frame = process_func(frame,0)
        output_h,output_w = processed_frame.shape[:2]

        break
    # reset capture object
    cap.release()
    cap = cv2.VideoCapture(input_path)

        

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or you can use 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))


    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # process frame
        processed_frame = process_func(frame,frame_count)
        # Save the processed frame
        out.write(processed_frame)
        frame_count += 1

    
    cap.release()
    out.release()

class FaceCropper:
    def __init__(self,data_root,video_path):
        self.json_path = join(data_root,"faces_v3",basename(video_path)[:-4]+".json")
        self.data = []
        with open(self.json_path, 'r') as fp:
            self.data = json.load(fp)
    
        
    def process_face_video(self,frame,frame_no):
        frame = crop_face(frame,self.data[frame_no],mode='mouth',output_size=(128,128))
        return frame


def video_to_3d_np_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.array(frames)

def frames_to_square_tiles(frames):
    """
    input: frames: n*256*256*3
    output: tiles: (sqrt(n)*256)*(sqrt(n)*256)*3
    """
    n_frames = frames.shape[0]
    h,w = frames.shape[1:3]
    # check if n is a square number
    sqrt_n = int(np.sqrt(n_frames))
    assert sqrt_n*sqrt_n == n_frames, "number of frames must be a square number"
    
    canvas = np.zeros((sqrt_n*h,sqrt_n*w,3),dtype=np.uint8)
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            canvas[i*h:(i+1)*h,j*w:(j+1)*w,:] = frames[i*sqrt_n+j]
    return canvas


# scrabble

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
import ht2

data_root = "/data/datasets_v2/"
video_paths = glob(join(data_root,"mouth_crops/*.mp4"))
sample_video = video_paths[0]
print(sample_video)
label_path = join(data_root,"labels_v3",basename(sample_video).replace(".mp4",".json"))
with open(label_path) as f:
    label = json.load(f)
text = label["text_postprocessed"]
timestamps = label["time_stamp"]
frames = video_to_3d_np_array(sample_video)
cap = cv2.VideoCapture(sample_video)
fps = cap.get(cv2.CAP_PROP_FPS)

time_slide_window_step = 1 # second
time_slide_window_size = 384*384//(32*32) # frames

# generate training data
for start_frame_no in range(0,len(frames),int(time_slide_window_step*fps)):
    end_frame_no = start_frame_no + time_slide_window_size
    if end_frame_no > len(frames):
        break
    # get frames
    print(start_frame_no,end_frame_no)
    training_frames = frames[start_frame_no:end_frame_no]
    
    print(len(training_frames))
    tile_image = frames_to_square_tiles(training_frames)
    # tile_image = cv2.cvtColor(tile_image,cv2.COLOR_BGR2RGB)
    tile_image = cv2.resize(tile_image,(384,384))
    cv2.imwrite("test_img.jpg",tile_image)
    

    # get text
    training_text = ""
    for i in range(len(timestamps)):
        if timestamps[i][0] >= start_frame_no/fps*1000 and timestamps[i][1] <= end_frame_no/fps*1000:
            training_text += text[i]
    print(training_text)
    
    
    # save frames and text
    # np.save(join(data_root,"training_frames",basename(sample_video).replace(".mp4","_%d_%d.npy"%(start_time_stamp,end_time_stamp))),training_frames)
    # with open(join(data_root,"training_text",basename(sample_video).replace(".mp4","_%d_%d.txt"%(start_time_stamp,end_time_stamp))),"w") as f:
    #     f.write(training_text)


# clean up into functions

def generate_image_sequence_transcription_training_data(frames,text,timestamps,export_path):
    """
    input: frames: video represented by 3d numpy array
           text: text transcript
           timestamps: timestamps of each word in the transcript
    output:
        image file saved
        text file saved
    """
    text = label["text_postprocessed"]
    timestamps = label["time_stamp"]
    frames = video_to_3d_np_array(sample_video)
    cap = cv2.VideoCapture(sample_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    time_slide_window_step = 1 # second
    time_slide_window_size = 384*384//(32*32) # frames

    # generate training data
    for start_frame_no in range(0,len(frames),int(time_slide_window_step*fps)):
        end_frame_no = start_frame_no + time_slide_window_size
        if end_frame_no > len(frames):
            break
        # get frames
        print(start_frame_no,end_frame_no)
        training_frames = frames[start_frame_no:end_frame_no]
        
        print(len(training_frames))
        tile_image = frames_to_square_tiles(training_frames)
        # tile_image = cv2.cvtColor(tile_image,cv2.COLOR_BGR2RGB)
        tile_image = cv2.resize(tile_image,(384,384))
        cv2.imwrite("test_img.jpg",tile_image)
        

        # get text
        training_text = ""
        for i in range(len(timestamps)):
            if timestamps[i][0] >= start_frame_no/fps*1000 and timestamps[i][1] <= end_frame_no/fps*1000:
                training_text += text[i]
        print(training_text)



    