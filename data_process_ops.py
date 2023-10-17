import cv2
import numpy as np
import ht2

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
