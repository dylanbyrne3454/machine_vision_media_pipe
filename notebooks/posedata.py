import numpy as np
import pandas as pd
from utilities import utils


class PoseData:
    LANDMARK_NAMES = {  # Using MediaPipe's landmark names
        "NOSE": 0,
        "LEFT_EYE_INNER": 1,
        "LEFT_EYE": 2,
        "LEFT_EYE_OUTER": 3,
        "RIGHT_EYE_INNER": 4,
        "RIGHT_EYE": 5,
        "RIGHT_EYE_OUTER": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "MOUTH_LEFT": 9,
        "MOUTH_RIGHT": 10,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_PINKY": 17,
        "RIGHT_PINKY": 18,
        "LEFT_INDEX": 19,
        "RIGHT_INDEX": 20,
        "LEFT_THUMB": 21,
        "RIGHT_THUMB": 22,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32,
    }

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self.load_keypoints_to_dataframe()

    def load_keypoints_to_dataframe(self):
        """Loads keypoints from the text file into a normalized Pandas DataFrame."""
        landmarks = []
        frame = []
        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line == "-":
                    flat = np.array(frame).flatten()
                    landmarks.append(flat)
                    frame = []
                else:
                    x, y, z = map(float, line.split(","))
                    frame.append([x, y, z])

        columns = [
            f"{landmark_name}_{coord}"
            for landmark_name in self.LANDMARK_NAMES
            for coord in ["x", "y", "z"]
        ]
        df = pd.DataFrame(landmarks, columns=columns)
        return df