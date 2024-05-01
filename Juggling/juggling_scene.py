import json
from typing import Dict, List
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, 'C://Users//evansamaa//Documents//GitHub//EvansToolBox//Utils')

from Geometry_Util import rotation_angles_frome_positions, directions_from_rotation_angles
from Signal_processing_utils import intensity_from_signal
from scipy.interpolate import interp1d
from scipy.special import softmax
import librosa

class JugglingScene:
    # image based variables
    def __init__(self, scene_data_path, motion_data_path, wonder = True):
        self.wonder = wonder
        with open(scene_data_path) as f:
            scene_data = json.load(f)
        # print(scene_data.keys())
        self_info = scene_data["self_pos"]
        # get camera position
        try:
            self.camera_pos = np.array(scene_data["cam_pos"])
        except:
            self.camera_pos = np.array(self_info["calibration_dir_world"])
        # the position of the speaker, in world coordinate (constant in this version)
        self.self_position_world = np.array(self_info["pos"])
        self.speaker_frame_pos = np.array(self_info["pos"])
        # get info used to compute transformation matrix from world coordinate to face coordinate
        self.speaker_face_direction_local = np.array(self_info["calibration_dir_local"])
        v_ref_world = np.array(self_info["calibration_dir_world"])
        v_ref_local = np.array(self_info["calibration_dir_local"])
        self.local_to_world = self.rotation_matrix_from_vectors(v_ref_local, v_ref_world - self.self_position_world)
        self.world_to_local = np.linalg.inv(self.local_to_world)
        # get info regarding other items in the scene
        temp_object_pos = scene_data["object_pos"]
        temp_scene_object_ids = list(temp_object_pos.keys())
        # sort the object id
        temp_scene_object_ids = sorted(temp_scene_object_ids)
        # name of the passive objects of the scene
        self.object_pos = []
        self.object_id = []
        for i in range(0, len(temp_scene_object_ids)):
            self.object_pos.append(temp_object_pos[temp_scene_object_ids[i]])
            self.object_id.append(temp_scene_object_ids[i])
            
        self.object_pos = np.array(self.object_pos)
        self.scene_data = scene_data
        
        self.motion_json = json.load(open(motion_data_path, "r"))
        self.look_at_id = np.array(self.motion_json["look_at_indices"], dtype=np.int)
        
    def get_gaze_trajectory(self):
        output_traj = np.zeros(self.object_pos[0].shape)
        print(output_traj.shape)
        for i in range(0, output_traj.shape[0]):
            output_traj[i] = self.transform_world_to_local(self.object_pos[self.look_at_id[i], i])
            # output_traj[i] = self.object_pos[self.look_at_id[i], i]
        return output_traj
    
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1z, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c == 1:
            return np.eye(3)
        elif c == -1:
            return -np.eye(3)

        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    def transform_world_to_local(self, pos_world):
        p = pos_world - self.self_position_world
        return self.world_to_local @ p
    def transform_local_to_world(self, pos_local):
        p = self.local_to_world @ pos_local + self.self_position_world
        return p
    def get_object_positions(self, id=-1, coordinate_space="local"):
        if id < 0:
            if coordinate_space == "global":
                return self.object_pos
            elif coordinate_space == "local":
                out = np.zeros(self.object_pos.shape)
                for i in range(0, self.object_pos.shape[0]):
                    out[i] = self.transform_world_to_local(self.object_pos[i])
                return out
        elif id >= 0:
            if coordinate_space == "global":
                return self.object_pos[id]
            elif coordinate_space == "local":
                out = self.transform_world_to_local(self.object_pos[id])
                return out
    def get_active_object_position(self, coordinate_space="local"):
        id = self.active_object_id
        return self.get_object_positions(id, coordinate_space=coordinate_space)
    def get_all_positions(self, coordinate_space="local"):
        objs = self.get_object_positions(coordinate_space=coordinate_space)
        active_objs = self.get_active_object_position()
        wp = self.get_wondering_points(coordinate_space=coordinate_space, neutral_gaze_spot_local=active_objs)
        if self.wonder:
            possss = np.concatenate([objs, wp], axis=0)
        else:
            possss = np.concatenate([objs], axis=0)
        return possss
    def get_camera_pos(self, coordinate_space="local"):
        if coordinate_space == "local":
            return self.transform_world_to_local(self.camera_pos)
        else:
            return self.camera_pos
