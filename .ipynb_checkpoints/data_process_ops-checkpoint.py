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