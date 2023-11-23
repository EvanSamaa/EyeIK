import pickle as pkl
from sklearn.mixture import GaussianMixture
import json
import os
import numpy as np
from EyeCatch_implementation import *
# from Oyekoya_implementation import *



class Jin_SacccadeGenerator:
    def __init__(self, target_times, target_positions, target_index, internal_model):
        # gaze state variables:
        # video metadata
        self.target_times = target_times
        self.target_positions = target_positions
        self.target_index = target_index
        self.movement_threshold = 2000 # use to detect intervals with no gaze shift. In which micro saccade are generated
        self.internal_model = internal_model
        self.eye_catch = EyeCatch_SacccadeGenerator(target_times, target_positions, target_index, internal_model)
        self.simulation_dt = self.eye_catch.simulation_dt
        self.decomp_azimuth = GMM_Decomposition.fromfile("jin2019_related_files/model/head_eye_decomposition_azimuth_60_clusters_fixation/")
        self.decomp_elevaiton = GMM_Decomposition.fromfile("jin2019_related_files/model/head_eye_decomposition_elevation_60_clusters_fixation/")
        self.gaze_positions = []
        self.head_positions = []
        self.micro_saccade_kf = []
    def compute(self):
        ek, hk, saccade = self.eye_catch.compute()
        gaze_positions = self.eye_catch.gaze_positions
        gaze_angles = gaze_vector_to_angle(gaze_positions)
        head_angles_azi = self.decomp_azimuth.decompose_sequence(gaze_angles[:, 0])[1]
        head_angles_ele = self.decomp_elevaiton.decompose_sequence(gaze_angles[:, 1])[1]
        head_angles = np.zeros((len(head_angles_azi), 2))
        head_angles[:, 0] = head_angles_azi
        head_angles[:, 1] = head_angles_ele
        self.head_positions = head_angles
        self.gaze_positions = self.eye_catch.gaze_positions
        self.micro_saccade_kf = self.eye_catch.micro_saccade_kf
        return self.prepare_output()
    def prepare_output(self):
        eye_kf = []
        head_kf = []
        ts = np.arange(0, self.target_times[-1] + 10.0, self.simulation_dt)
        # insert the key frames for gaze into the output array
        for i in range(0, ts.shape[0]):
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]), float(self.gaze_positions[i][2])])

        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(self.head_positions[i][0]), float(self.head_positions[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf
    
def gaze_vector_to_angle(arr):
    # F: arr (N, 3) -> arr (N, 2)
    # in the output is in the convention of (azimuth, elevation)
    # azimuth: +right,-left
    # elevation: +up,-down
    mag = np.sqrt(np.sum(arr * arr, axis=1, keepdims=True))

    out = arr / mag
    out[:, 0] = np.arcsin(out[:, 0])
    out[:, 1] = np.arcsin(out[:, 1])

    return out[:, 0:2] * 180 / np.pi

class GMM_Decomposition:
    """
    The main thing here to use are decompose and decompose_sequence
    decompose:              breaks down a world angle into an Eye-In-Head angle and head angle
    decompose_sequence:     breaks down a sequence of world angle into a sequence of Eye-In-Head angle and head angle

    To get the model from saved file, call:
    asimuth_decomp = GMM_Decomposition.fromfile("prototypes/Jin2019/model/head_eye_decomposition_azimuth_60_clusters_fixation/")
    elevation_decomp = GMM_Decomposition.fromfile("prototypes/Jin2019/model/head_eye_decomposition_elevation_60_clusters_fixation/")
    """
    def __init__(self, gmm_dict: dict):
        self.gmm_dict: dict = gmm_dict

    @classmethod
    def fromfile(cls, model_path: str):
        temp_gmm_dict: dict = {}
        metadata: dict = json.load(open(model_path + "/metadata.json"))
        for key in metadata.keys():
            filepath = metadata[key]
            try:
                temp_gmm_dict[int(float(key))] = pkl.load(open(filepath, "rb"))
            except:
                alt_filepath = "C:/Users/evansamaa/Documents/Github/Gaze_project/prototypes/Jin2019/model/" + filepath[54:] 
                try:               
                    temp_gmm_dict[int(float(key))] = pkl.load(open(alt_filepath, "rb"))
                except:
                    alt_filepath = "/Users/evanpan/Documents/GitHub/Gaze_project/prototypes/Jin2019/model/" + filepath[54:]
                    temp_gmm_dict[int(float(key))] = pkl.load(open(alt_filepath, "rb"))
                    
        return cls(temp_gmm_dict)
    def save_model(self, model_path: str):
        try:
            os.mkdir(model_path)
        except:
            print("model already exist")

        metadata = {}
        for key in self.gmm_dict.keys():
            if key >= 0:
                file_prefix = str(int(key))
            else:
                file_prefix = "n" + str(abs(int(key)))
            full_file_path = model_path + "/" + file_prefix + "model_.sav"
            pkl.dump(self.gmm_dict[key], open(full_file_path, 'wb'))
            metadata[key] = full_file_path
        json.dump(metadata, open(model_path + "/metadata.json", "w"))
    def decompose(self, alpha, prev=None):
        # alpha is a single float point value, either the degree of angle at the elevation or azimuth plane
        # prev is a np array with shape (2, ), it is the previous point
        bin_num = np.round(alpha)
        gmm: GaussianMixture = self.gmm_dict[bin_num]
        means = gmm.means_
        prob = gmm.weights_
        next_val = np.zeros((2,))
        if prev is None:
            next_val = means[np.argmax(prob)]
        else:
            selection_criteria = means - np.expand_dims(prev, axis=0)
            selection_criteria = np.sum(selection_criteria * selection_criteria, axis=1)
            selection_criteria = selection_criteria / prob
            next_val = means[np.argmin(selection_criteria)]
        return next_val
    def decompose_sequence(self, alpha_list):
        EIH_arr = []
        head_arr = []
        prev_break_down = None
        for i in range(0, len(alpha_list)):
            alpha = alpha_list[i]
            current_break_down = self.decompose(alpha, prev_break_down)
            EIH_arr.append(current_break_down[0])
            head_arr.append(current_break_down[1])
            prev_break_down = current_break_down
        return EIH_arr, head_arr