import numpy as np

# Demo libs
from spiga.demo.analyze.features.basic import ObjectAnalyzed


class Face(ObjectAnalyzed):

    def __init__(self):
        super().__init__()
        self.bbox = np.zeros(5)
        self.key_landmarks = - np.ones((5, 2))
        self.landmarks = None
        self.face_id = -1
        self.past_states = []
        self.num_past_states = 5

    def get_attributes(self):
        # 2d array to list
        bbox = self.bbox.tolist()
        key_landmarks = self.key_landmarks.tolist()

        return {
            "bbox":bbox,
            "key_landmarks":key_landmarks,
            "landmarks":self.landmarks,
            "face_id":self.face_id,
            "past_states":self.past_states,
            "num_past_states":self.num_past_states
        }




