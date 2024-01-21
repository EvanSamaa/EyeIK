import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/Gaze_project')
sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/EvansToolBox/Utils")
sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/Gaze_project")
sys.path.insert(0, "C:/Users/evan1/Documents/GitHub/EvansToolBox/Utils")
sys.path.insert(0, "C:/Users/evan1/Documents/GitHub/Gaze_project")

from Geometry_Util import rotation_angles_frome_positions, rotation_axis_angle_from_vector, rotation_matrix_from_axis_angle, rotation_matrix_from_vectors, directions_from_rotation_angles
from Signal_processing_utils import dx_dt
from InputStructures import InputStructure

class Ground_truth_SacccadeGenerator:
    # helper functions
    def __init__(self, input_json):
        print("yeet")
        self.input_json = input_json
    def compute(self):
        eye_pos = self.input_json["eye_trajectory"]
        head_pos = self.input_json["head_trajectory"]
        eye_pos_pre_transpose = np.array([eye_pos])
        head_pos_pre_transpose = np.array([head_pos])
        # switch [:, :, 1] and [:, :, 2]
        eye_pos = np.concatenate([eye_pos_pre_transpose[:, :, 0:1], eye_pos_pre_transpose[:, :, 2:3], eye_pos_pre_transpose[:, :, 1:2], eye_pos_pre_transpose[:, :, 1:2]], axis=2)
        head_pos = np.concatenate([head_pos_pre_transpose[:, :, 0:1], head_pos_pre_transpose[:, :, 2:3], head_pos_pre_transpose[:, :, 1:2]], axis=2)
        eye_pos[0, :, 0] = eye_pos[0, :, 0] - eye_pos[0, 0, 0]
        eye_pos[0, :, 1:3] = eye_pos[0, :, 1:3] + head_pos[0, :, 1:] 
        head_pos[0, :, 0] = head_pos[0, :, 0] - head_pos[0, 0, 0]
        
        
        eye_pos[0, :, 1:] = directions_from_rotation_angles(eye_pos[0, :, 1:], 100).tolist()

        head_pos = head_pos[:, :, 0:3].tolist()
        
        # use float() on every single element in eye_pos and head_pos, so that they are native python variable instead of numpy variable
        out_eye = []
        for i in range(0, len(eye_pos[0])):
            out_eye.append([float(eye_pos[0][i][0]), float(eye_pos[0][i][1]), float(eye_pos[0][i][2]), float(eye_pos[0][i][3])])
        out_head = []
        for i in range(0, len(head_pos[0])):
            out_head.append([float(head_pos[0][i][0]), float(head_pos[0][i][1]), float(head_pos[0][i][2])])
        out_eye = [out_eye]
        out_head = [out_head]
        return out_eye, out_head, [[]]