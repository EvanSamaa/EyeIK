import pickle as pkl
from sklearn.mixture import GaussianMixture
import json
import os
import numpy as np


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
class Heuristic_decomposition_azimuth:
    # this is within version 2p2
    def __init__(self, assym=0):
        # assym = 0: symmetrical,
        # assym = 1: more neck contribution at positive angle (right)
        # assym = 2: more neck contribution at negative angle (left)
        self.assym = assym
        # here the percentage the is the percentage of neck contribution
        self.sym_pts_low = [[-90, 0.3], [-50, 0.1], [-25, 0.05], [-10, 0.05], [0, 0], [10, 0.05],
        [25, 0.05], [50, 0.1], [90, 0.3]]
        self.sym_pts_mid = [[-90, 0.6], [-50, 1], [-25, 0.75], [-10, 0.4], [0, 0], [10, 0.4],
                            [25, 0.75], [50, 1], [90, 0.6]]
        self.sym_pts_high = [[-90, 0.6], [-50, 1], [-25, 0.9], [-10, 0.5], [0, 0], [10, 0.5],
        [25, 0.9], [50, 1], [90, 0.6]]

    def get_y(self, p1, p2, x):
        return (x - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]

    def decompose(self, alpha, neck_enthusiasm):
        # alpha is a single float point value, should be the degree of angle at the azimuth plane

        # if alpha is an extreme angle, then we just fix neck contribution at 60%
        if abs(alpha) >= 90:
            neck_alpha = 0.6 * alpha
            eye_alpha = alpha - neck_alpha
            return eye_alpha, neck_alpha

        # otherwise, we find the right bin and use the corresponding line
        for i in range(1, len(self.sym_pts_low)):
            if self.sym_pts_low[i][0] >= alpha:
                bin0 = i - 1
                bin1 = i
                break
        if neck_enthusiasm >= 0.5:
            factor = 2.0 * (neck_enthusiasm - 0.5)
            pt0 = np.array([self.sym_pts_low[bin0][0],
                            self.sym_pts_high[bin0][1] * factor + (1.0 - factor) * self.sym_pts_mid[bin0][1]])
            pt1 = np.array([self.sym_pts_low[bin1][0],
                            self.sym_pts_high[bin1][1] * factor + (1.0 - factor) * self.sym_pts_mid[bin1][1]])
        else:
            factor = 2.0 * neck_enthusiasm
            pt0 = np.array([self.sym_pts_low[bin0][0],
                            self.sym_pts_mid[bin0][1] * factor + (1.0 - factor) *  self.sym_pts_low[bin0][1]])
            pt1 = np.array([self.sym_pts_low[bin1][0],
                            self.sym_pts_mid[bin1][1] * factor + (1.0 - factor) *  self.sym_pts_low[bin1][1]])
        ratio = self.get_y(pt0, pt1, alpha)
        neck_alpha = alpha * ratio
        eye_alpha = alpha - neck_alpha
        return eye_alpha, neck_alpha

    def decompose_sequence(self, alpha_list):
        EIH_arr = []
        head_arr = []
        for i in range(0, len(alpha_list)):
            alpha = alpha_list[i]
            current_break_down = self.decompose(alpha)
            EIH_arr.append(current_break_down[0])
            head_arr.append(current_break_down[1])
        return EIH_arr, head_arr
class Heuristic_decomposition_elevation:
    def __init__(self, p1=None, p2=None, p3=None, p4=None):
        # here the percentage the is the percentage of neck contribution
        self.p0 = [90, 0.5]
        self.p1 = [0, 0.1]
    def get_y(self, p1, p2, x):
        return (x - p1[0])/(p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    def decompose(self, alpha, neck_contribution):
        # alpha is a single float point value, should be the degree of angle at the azimuth plane
        ab_alpha = abs(alpha)
        neck_alpha = 0.25 * ab_alpha * neck_contribution
        eye_alpha = ab_alpha - neck_alpha
        if alpha >= 0:
            neck_alpha = 0.25 * ab_alpha * neck_contribution
            eye_alpha = ab_alpha - neck_alpha
            return eye_alpha, neck_alpha
        else:
            neck_alpha = 0.6 * ab_alpha * neck_contribution
            eye_alpha = ab_alpha - neck_alpha
            return -eye_alpha, -neck_alpha
    def decompose_sequence(self, alpha_list):
        EIH_arr = []
        head_arr = []
        for i in range(0, len(alpha_list)):
            alpha = alpha_list[i]
            current_break_down = self.decompose(alpha)
            EIH_arr.append(current_break_down[0])
            head_arr.append(current_break_down[1])
        return EIH_arr, head_arr
class Heuristic_eye_velocity:
    def __init__(self, p1=None, p2=None, p3=None, p4=None):
        # here the percentage the is the percentage of neck contribution
        self.p0 = [0, 220]
        self.p1 = [20, 400]
        self.p2 = [40, 200]

    def get_y(self, p1, p2, x):
        return (x - p1[0])/(p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    def decompose(self, angle_displacement, prev=None):
        # alpha is a single float point value, should be the degree of angle at the azimuth plane
        if angle_displacement < self.p1[0]:
            return self.get_y(self.p0, self.p1, angle_displacement)
        elif angle_displacement < self.p2[0]:
            return self.get_y(self.p1, self.p2, angle_displacement)
        else:
            return 200
class Heuristic_eye_velocity_simple:
    def __init__(self):
        # here the percentage the is the percentage of neck contribution
        # by Coordination of the Eyes and Head during Visual Orienting
        self.travel_time_per_deg = 1.4 / 1000
    def decompose(self, angle_displacement, prev=None):
        # alpha is a single float point value, should be the degree of angle at the azimuth plane
        return 1 / self.travel_time_per_deg