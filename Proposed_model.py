import sys
sys.path.insert(0, '/Users/evansamaa/Documents/GitHub/Gaze_project')
sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/EvansToolBox/Utils")
import numpy as np
from matplotlib import pyplot as plt
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info, MultiPartyConversationalSceneInfo, AgentInfo, AgentInfo_final
from Geometry_Util import rotation_angles_frome_positions, rotation_axis_angle_from_vector, directions_from_rotation_angles, rotation_matrix_from_axis_angle, rotation_matrix_from_vectors
from Signal_processing_utils import dx_dt, laplacian_smoothing, sparse_key_smoothing
from InputStructures import InputStructure
import cvxpy as cp
import networkx as nx
import time

from prototypes.Optimization_based_head_eye_seperator.Baseline_optimization import optimize_for_head_gaze_breakdown, optimize_for_head_gaze_breakdown_dynamic_scene, optimize_for_head_gaze_breakdown_three_party_scene
class InternalModelCenterBias:
    def __init__(self, scene: AgentInfo):
        self.scene = scene
    def estimate_target_pose(self, index, previous_pos=None):
        if previous_pos is None:
            previous_pos = self.get_base_pose()
        personal_center_pos = self.get_base_pose() # this is in local space
        target_mean_pos = self.scene.get_all_positions(coordinate_space="local", index=index)

        # accuracy drops with distance (in degrees) and speed (speed seems to correlate with distance )
        target_rot = rotation_angles_frome_positions(target_mean_pos) # get the target pos in degrees
        prev_rot = rotation_angles_frome_positions(previous_pos) # get the prev pos in degrees
        distance = np.linalg.norm(target_rot - prev_rot) # get distance in degrees\

        # an exit condition, if the delta target is small, then there is no variability (as per
        # Speed and accuracy of saccadic eye movements: Characteristics of impulse variability in the oculomotor system)
        if distance <= 1:
            return target_mean_pos
        # estimate actual target to attend to be slightly off from the real position
        slightly_wrong_target_center = previous_pos + 0.8 * (target_mean_pos-previous_pos)
        # the variation of the target
        target_variation = np.abs(target_mean_pos-slightly_wrong_target_center)
        output_position = np.random.normal(slightly_wrong_target_center, target_variation)
        return output_position
    def get_base_pose(self):
        return self.scene.speaker_face_direction_local
    def estimate_listener_pose(self):
        return self.scene.get_active_object_position()[0]
        # for i in range(0, len(self.scene.object_type)):
        #     if self.scene.object_type[i] == 5:
        #         return self.scene.transform_world_to_local(self.scene.positions_world[i])
class InternalModelCenterBias_dynamic_scene:
    def __init__(self, scene: AgentInfo_final):
        self.scene = scene
    def estimate_target_pose(self, index, previous_pos=None):
        if previous_pos is None:
            previous_pos = self.get_base_pose()
        personal_center_pos = self.get_base_pose() # this is in local space
        target_mean_pos = self.scene.get_all_positions(coordinate_space="local")[index]

        # accuracy drops with distance (in degrees) and speed (speed seems to correlate with distance )
        target_rot = rotation_angles_frome_positions(target_mean_pos) # get the target pos in degrees
        prev_rot = rotation_angles_frome_positions(previous_pos) # get the prev pos in degrees
        distance = np.linalg.norm(target_rot - prev_rot) # get distance in degrees\

        # an exit condition, if the delta target is small, then there is no variability (as per
        # Speed and accuracy of saccadic eye movements: Characteristics of impulse variability in the oculomotor system)
        if distance <= 1:
            return target_mean_pos
        # estimate actual target to attend to be slightly off from the real position
        slightly_wrong_target_center = previous_pos + 0.8 * (target_mean_pos-previous_pos)
        # the variation of the target
        target_variation = np.abs(target_mean_pos-slightly_wrong_target_center)
        output_position = np.random.normal(slightly_wrong_target_center, target_variation)
        return output_position
    def get_base_pose(self):
        return self.scene.speaker_face_direction_local
    def new_listener(self, id):
        self.scene.active_object_id = id
    def estimate_listener_pose(self):
        # return self.estimate_target_pose(self.scene.active_object_id)
        return self.scene.get_active_object_position()
class InternalModelExact:
    def __init__(self, scene: InputStructure):
        self.scene = scene
    def estimate_target_pose(self, index):
        return self.scene.object_pos[index]
    def get_base_pose(self):
        return self.scene.speaker_face_direction_local

def display_targetss(gaze_points, label=None):
    # gaze_points = gaze_points / gaze_points[:, 2:3] * 100
    if label is None:
        plt.scatter(gaze_points[:, 0], gaze_points[:, 1])
    else:
        plt.scatter(gaze_points[:, 0], gaze_points[:, 1], label=label)
class Proposed_saccade_generator_vanilla:
    def interpolate_gaze_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_positions[i]
        print("Error")
    def interpolate_gaze_goal_index(self, t):
        if t < self.target_times[0]:
            return self.target_index[0]
        elif t >= self.target_times[-1]:
            return self.target_index[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_index[i]
        print("Error")
    def interpolate_head_goal(self, t):
        if t < self.target_gaze_intervals_time[0][0]:
            return self.target_positions_head[0]
        elif t >= self.target_gaze_intervals_time[-1][0]:
            return self.target_positions_head[-1]
        else:
            for i in range(0, len(self.target_gaze_intervals_time) - 1):
                if self.target_gaze_intervals_time[i][0] <= t and self.target_gaze_intervals_time[i + 1][0] >= t:
                    return self.target_positions_head[i]
        print("Error")
    def get_gaze_intervals(self, time_arr, pos_arr):
        """
        Given two arrays, one a sequence of times, and the other a sequence of positions.
        This function segment the sequence into a sequence of intervals with trackable
        Start and end time.
        :param time_arr: an array of times shape is (N, )
        :param pos_arr: an array of positions shape is  (N, K)
        :return:
        gaze_intervals_time: a list of [start, end] times
        gaze_intervals_pos: a list of positions
        """
        gaze_intervals_time = []
        gaze_intervals_pos = []
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            # in the case that we are at the final time, and the current differs from previous
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) < 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                if i > 0:
                    gaze_intervals_time.append([start, time_arr[i]])
                    gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) == 0:
                pass
            else:
                print("I wanna eat korean fried chicken")
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    def __init__(self, target_times, target_positions, target_index, internal_model, dt=0.02):
        # gaze state variables:
        self.gaze_current_goal_position = internal_model.get_base_pose()
        self.head_current_goal_position = internal_model.get_base_pose()
        self.gaze_most_recent_index = 0
        self.head_most_recent_index = 0

        # meta parameters:
        self.simulation_dt = dt
        self.submovement_dt = 0.200 
        self.movement_threshold = 200  # use to detect intervals with no gaze shift. In which micro saccade are generated

        # Internal Model of the scene (will introduce bias, errors and everything)
        self.internal_model = internal_model

        # simulation state variables:
        self.target_times = target_times
        self.target_positions = np.array(target_positions)

        self.target_index = target_index
        self.target_positions_head = None
        self.target_gaze_intervals_time, self.target_gaze_intervals_pos = self.get_gaze_intervals(self.target_times, self.target_positions)
    
        self.t = 0

        # Initialize aray to store the state history (positions) i.e. the animation curve
        end_t = self.target_times[-1] + 10.0
        end_t = int(np.ceil(end_t / self.simulation_dt))
        # the memory will be allocated ahead of time since we know
        self.gaze_positions = internal_model.get_base_pose().astype(np.float32)
        self.gaze_positions = np.expand_dims(self.gaze_positions, axis=0)
        self.gaze_positions = np.tile(self.gaze_positions, [end_t, 1])
        self.head_positions = internal_model.get_base_pose().astype(np.float32)
        self.head_positions = np.expand_dims(self.head_positions, axis=0)
        self.head_positions = np.tile(self.head_positions, [end_t, 1])
        
        # set aside array to store the saccade keyframes
        self.micro_saccade_kf = []
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_frome_positions(pos0)
        rot1 = rotation_angles_frome_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        return duration / 1000
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(round(t0 / dt))
        tf = int(round(tf / dt))
        t = np.arange(t0, tf, 1)
        if t.shape[0] != 1:
            v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        else:
            v = np.array([1])
        v = v / v.sum()

        return v
    def add_gaze_submovement(self, t0, t1, p0, p1):
        # if there is nothing to do, do nothing
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # get the velocity profile of the submovement
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        # get the direction of the submovement
        submovement_direction = p1 - p0
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction
        # get the starting and ending frame of the submovmeent
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # update current gaze_goal position
        # print(p1 - p0, submovement.sum(axis=0), t0, t1)
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        p0 = p0_not_normalized / np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized / np.linalg.norm(p1_not_normalized) * 100
        # if the distance very close I don't make any submovements
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement
        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
            # threshold the rotation speed
            rot_angle = np.maximum(0.0, rot_angle * 0.9)
            # compute the actual rotation matrix
            rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
            # find the displacement using the rotation
            submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)
        # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        else:
            submovement_direction = (p1_not_normalized - p0_not_normalized)
            submovement_magnitude = np.linalg.norm(submovement_direction)
            reduced_submovement_magnitude = submovement_magnitude * 0.9
            submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        self.head_current_goal_position = p0_not_normalized + submovement_direction
        # find the speed
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05,
                            avg_saccade_interval=0.5):
        output_list = []
        rig_factor = 10  # I believe it should be rig_factor[130] and rig_factor[131],
        max_saccade_x = 2 * saccade_factor
        max_saccade_y = 2 * saccade_factor
        prev_saccade_frame_counter = prev_saccade_frame
        if prev_saccade_frame <= start_frame:
            saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
            saccade_duration = 1.0 / 24.0
            output_list.append([start_frame, 0, 0])
            prev_saccade_frame_counter = start_frame + saccade_interval
            output_list.append([prev_saccade_frame_counter, 0, 0])
            prev_saccade_frame_counter += saccade_duration
        while prev_saccade_frame_counter < end_frame:
            # compute offset
            offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
            offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
            saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval / 10.0
            saccade_duration = 1.0 / 24.0
            if prev_saccade_frame_counter + saccade_duration + saccade_interval >= end_frame:
                offset_x = 0
                offset_y = 0
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_interval
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_duration
        return output_list, prev_saccade_frame_counter
    def compute(self):
        # first compute the head/eye contribution of the gaze:
        # self.target_positions_head = optimize_for_head_gaze_breakdown_dynamic_scene(self.target_gaze_intervals_time, self.target_gaze_intervals_pos, self.internal_model, self.gaze_target_over_time)        
        self.target_positions_head = self.target_positions
        self.gaze_current_goal_position = self.target_gaze_intervals_pos[0]
        self.head_current_goal_position = self.target_gaze_intervals_pos[0]
        for i in range(0, self.head_positions.shape[0]):
            self.head_positions[i] = self.head_current_goal_position
            self.gaze_positions[i] = self.gaze_current_goal_position
        
        # add an end time to the sequence
        end_t = self.target_times[-1] + 10.0
        # store the current list of submovments that gets updated
        gaze_submovements = []  # use to store a list of existing submovements
        gaze_submovements_indexes = []  # use to store the index of each submovement
        head_submovements = []  # use to store a list of existing submovements
        head_submovements_indexes = []  # use to store the index of each submovement
        while self.t < end_t - self.simulation_dt/2:
            t_index = int(np.round(self.t / self.simulation_dt))
            # use to store the gaze and head submovements that have expired
            expired_gaze = []
            expired_head = []
            # update the gaze positions
            self.gaze_positions[t_index] = self.gaze_positions[max(t_index - 1, 0)]
            for i in range(0, len(gaze_submovements)):
                if t_index < gaze_submovements_indexes[i][1] and t_index >= gaze_submovements_indexes[i][0]:
                    self.gaze_positions[t_index] += gaze_submovements[i][t_index - gaze_submovements_indexes[i][0]]
                elif t_index < gaze_submovements_indexes[i][0]:
                    pass
                else:
                    expired_gaze.append(i)
            # update the head positions
            self.head_positions[t_index] = self.head_positions[max(t_index - 1, 0)]
            for i in range(0, len(head_submovements)):
                if t_index < head_submovements_indexes[i][1] and t_index >= head_submovements_indexes[i][0]:
                    self.head_positions[t_index] += head_submovements[i][t_index - head_submovements_indexes[i][0]]
                elif t_index < head_submovements_indexes[i][0]:
                    pass
                else:
                    expired_head.append(i)
            # only generate saccade at fixed intervals i.e. if saccade_generation_test is an integer
            saccade_generation_test = self.t / self.submovement_dt
            if abs(saccade_generation_test - int(round(saccade_generation_test))) <= 0.001:
                # obtain gaze shift duration as per properties of main sequence
                gaze_movement_duration = self.get_saccade_duration(self.gaze_current_goal_position,
                                                                   self.interpolate_gaze_goal(self.t))
                # add the gaze submovement
                next_gaze_index = int(self.interpolate_gaze_goal_index(self.t))
                next_gaze_goal = self.internal_model.estimate_target_pose(next_gaze_index)
                gaze_submovement, gaze_submovement_range = self.add_gaze_submovement(self.t,
                                                                                        self.t + gaze_movement_duration,
                                                                                        self.gaze_current_goal_position,
                                                                                        next_gaze_goal)
                if not gaze_submovement is None:
                    gaze_submovements.append(gaze_submovement)
                    gaze_submovements_indexes.append(gaze_submovement_range)
                    self.gaze_positions[t_index] += gaze_submovement[0]
                # obtain head shift duration
                head_movement_duration = 0.6
                # get angle between two vectors and do something if the angle is small
                if np.linalg.norm(self.head_current_goal_position - self.interpolate_head_goal(self.t)) > 1E-3:
                    magnitude1 = np.linalg.norm(self.head_current_goal_position)
                    magnitude2 = np.linalg.norm(self.interpolate_head_goal(self.t))
                    dot_product = np.dot(self.head_current_goal_position, self.interpolate_head_goal(self.t))
                    # Calculate the cosine of the angle between the vectors
                    cosine_angle = dot_product / (magnitude1 * magnitude2)
                    # Use arccosine to find the angle in radians
                    angle_in_radians = np.arccos(cosine_angle)
                    # Convert the angle from radians to degrees
                    angle_in_degrees = np.degrees(angle_in_radians)
                    if angle_in_degrees <= 20:
                        # head_movement_duration = 0.3
                        # print(self.head_current_goal_position, self.interpolate_head_goal(self.t))                        
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        pass
                         
                # add the head movement
                head_submovement, head_submovement_range = self.add_head_submovement(self.t,
                                                                                     self.t + head_movement_duration,
                                                                                     self.head_current_goal_position,
                                                                                     self.interpolate_head_goal(self.t))
                if not head_submovement is None:
                    head_submovements.append(head_submovement)
                    head_submovements_indexes.append(head_submovement_range)
                    self.head_positions[t_index] += head_submovement[0]

            # remove the submovements that have been expired, doing this backwards to not mess with indexing order
            for i in range(len(expired_head) - 1, -1, -1):
                head_submovements.pop(expired_head[i])
                head_submovements_indexes.pop(expired_head[i])
            # remove the submovements that have been expired
            for i in range(len(expired_gaze) - 1, -1, -1):
                gaze_submovements.pop(expired_gaze[i])
                gaze_submovements_indexes.pop(expired_gaze[i])


            # accumulate time
            self.t += self.simulation_dt
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0  # track the start of an interval
        end = 0  # track the end of an interval
        plottt = []
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i - 30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            plottt.append(speed_arr[i - 30:i].sum())
            end += self.simulation_dt
        prev_saccade = 0
        for i in range(0, len(stable_windows)):
            start = stable_windows[i][0]
            end = stable_windows[i][1]
            micro_saccade_list, prev_saccade = self.handle_microsaccade(start, prev_saccade, end)
            self.micro_saccade_kf.append(micro_saccade_list)
        return self.prepare_output()

    def prepare_output(self):
        eye_kf = []
        head_kf = []
        ts = np.arange(0, self.target_times[-1] + 10.0, self.simulation_dt)
        # insert the key frames for gaze into the output array
        for i in range(0, ts.shape[0]):
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]),
                           float(self.gaze_positions[i][2])])
        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(head_rotations[i][0]), float(head_rotations[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf
class Proposed_saccade_generator:
    def optimize_for_neck_angle(self, gaze_intervals, list_of_gaze_positions):
        gaze_angles = rotation_angles_frome_positions(list_of_gaze_positions)
        prev_angle = gaze_angles[0]
        solved_angles = []
        for i in range(0, gaze_angles.shape[0]):
            gaze_time = gaze_intervals[i][1] - gaze_intervals[i][0]
            neck_angle_ele = cp.Variable(1)
            neck_angle_azi = cp.Variable(1)
            # next_angle = self.get_average_angle_at_times([gaze_intervals[i][0]-2, gaze_intervals[i][0] + 2])
            # optimize for neck angle azi
            # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2
            #                         + (neck_angle_azi - listener_angle[0])**2)
            gaze_target = gaze_angles[i]
            next_angle = self.input_gaze_pos_as_angles_per_frame_smoothed[int(gaze_intervals[i][0])]            
            objective = cp.Minimize(np.maximum(0.0001, 1.5-np.log(gaze_time + 1)) * (neck_angle_azi - next_angle[0]) ** 2 +
                            (np.log(gaze_time + 1)) * (neck_angle_azi - gaze_angles[i, 0]) ** 2)
            
            
            problem = cp.Problem(objective, [])
            opt = problem.solve()
            objective = cp.Minimize(np.maximum(0.0001, 1.5-np.log(gaze_time + 1))  * (neck_angle_ele - next_angle[1]) ** 2 +
                            np.log(gaze_time + 1) * (neck_angle_ele - gaze_angles[i, 1]) ** 2)
            problem = cp.Problem(objective, [])
            opt = problem.solve()
            prev_angle = [neck_angle_ele.value[0], neck_angle_azi.value[0]]
            if False:
            # if gaze_intervals[i][1] - gaze_intervals[i][0] <= 0.2 and neck_angle_azi.value[0] <= 20:
                solved_angles.append([[prev_angle[0], prev_angle[1]]])
            else:
                solved_angles.append(np.array([[neck_angle_azi.value[0], 
                                                neck_angle_ele.value[0]]]))
        solved_positions = np.concatenate(solved_angles, axis=0)
        solved_positions = directions_from_rotation_angles(solved_positions, 100)
        return solved_positions
    def interpolate_gaze_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_positions[i]
        print("Error")
    def interpolate_gaze_goal_index(self, t):
        if t < self.target_times[0]:
            return self.target_index[0]
        elif t >= self.target_times[-1]:
            return self.target_index[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_index[i]
        print("Error")
    def interpolate_head_goal(self, t):
        if t < self.target_gaze_intervals_time[0][0]:
            return self.target_positions_head[0]
        elif t >= self.target_gaze_intervals_time[-1][0]:
            return self.target_positions_head[-1]
        else:
            for i in range(0, len(self.target_gaze_intervals_time) - 1):
                if self.target_gaze_intervals_time[i][0] <= t and self.target_gaze_intervals_time[i + 1][0] >= t:
                    return self.target_positions_head[i]
        print("Error")
    def get_gaze_intervals(self, time_arr, pos_arr):
        """
        Given two arrays, one a sequence of times, and the other a sequence of positions.
        This function segment the sequence into a sequence of intervals with trackable
        Start and end time.
        :param time_arr: an array of times shape is (N, )
        :param pos_arr: an array of positions shape is  (N, K)
        :return:
        gaze_intervals_time: a list of [start, end] times
        gaze_intervals_pos: a list of positions
        """
        gaze_intervals_time = []
        gaze_intervals_pos = []
        for i in range(0, len(time_arr)-1):
            gaze_intervals_time.append([time_arr[i], time_arr[i+1]])
            gaze_intervals_pos.append(np.expand_dims(pos_arr[i], 0))
        gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
        gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
        
        
        
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            # in the case that we are at the final time, and the current differs from previous
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) < 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                if i > 0:
                    gaze_intervals_time.append([start, time_arr[i]])
                    gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) == 0:
                pass
            else:
                print("I wanna eat korean fried chicken")
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    def __init__(self, target_times, target_positions, target_index, internal_model, dt=0.025):
        # gaze state variables:
        self.gaze_current_goal_position = internal_model.get_base_pose()
        self.head_current_goal_position = internal_model.get_base_pose()
        self.gaze_most_recent_index = 0
        self.head_most_recent_index = 0
        self.dire_change_delay = 0.0
        
        # meta parameters:
        self.simulation_dt = dt
        self.submovement_dt = 0.200 
        self.movement_threshold = 200  # use to detect intervals with no gaze shift. In which micro saccade are generated
        self.head_movement_duration = 0.4
        
        # Internal Model of the scene (will introduce bias, errors and everything)
        self.internal_model = internal_model

        # simulation state variables:
        self.target_times = target_times
        self.target_positions = np.array(target_positions)

        self.target_index = target_index
        self.target_positions_head = None
        self.target_gaze_intervals_time, self.target_gaze_intervals_pos = self.get_gaze_intervals(self.target_times, self.target_positions)
    
        self.t = 0

        # Initialize aray to store the state history (positions) i.e. the animation curve
        end_t = self.target_times[-1] + 10.0
        end_t = int(np.ceil(end_t / self.simulation_dt))
        # the memory will be allocated ahead of time since we know
        self.gaze_positions = internal_model.get_base_pose().astype(np.float32)
        self.gaze_positions = np.expand_dims(self.gaze_positions, axis=0)
        self.gaze_positions = np.tile(self.gaze_positions, [end_t, 1])
        self.gaze_pos_as_angles = rotation_angles_frome_positions(self.target_positions)
        self.input_gaze_pos_as_angles_per_frame = np.zeros((end_t, 2)) 
        for i in range(0, end_t):
            self.input_gaze_pos_as_angles_per_frame[i] = self.gaze_pos_as_angles[int(self.interpolate_gaze_goal_index(i * self.simulation_dt))]
        self.input_gaze_pos_as_angles_per_frame_smoothed = np.array(laplacian_smoothing(self.input_gaze_pos_as_angles_per_frame, 50))   
        # plt.plot(self.input_gaze_pos_as_angles_per_frame[:, 0])
        # plt.plot(self.input_gaze_pos_as_angles_per_frame_smoothed[:, 0])
        # plt.show()
        self.head_positions = internal_model.get_base_pose().astype(np.float32)
        self.head_positions = np.expand_dims(self.head_positions, axis=0)
        self.head_positions = np.tile(self.head_positions, [end_t, 1])
        
        # set aside array to store the saccade keyframes
        self.micro_saccade_kf = []
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_frome_positions(pos0)
        rot1 = rotation_angles_frome_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        return duration / 1000
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(round(t0 / dt))
        tf = int(round(tf / dt))
        t = np.arange(t0, tf, 1)
        if t.shape[0] != 1:
            v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        else:
            v = np.array([1])
        v = v / v.sum()

        return v
    def add_gaze_submovement(self, t0, t1, p0, p1):
        # if there is nothing to do, do nothing
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # get the velocity profile of the submovement
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        # get the direction of the submovement
        submovement_direction = p1 - p0
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction
        # get the starting and ending frame of the submovmeent
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # update current gaze_goal position
        # print(p1 - p0, submovement.sum(axis=0), t0, t1)
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        p0 = p0_not_normalized / np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized / np.linalg.norm(p1_not_normalized) * 100
        # if the distance very close I don't make any submovements
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement
        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
            # threshold the rotation speed
            rot_angle = np.maximum(0.0, rot_angle * 0.7)
            # compute the actual rotation matrix
            rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
            # find the displacement using the rotation
            submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)
        # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        else:
            submovement_direction = (p1_not_normalized - p0_not_normalized)
            submovement_magnitude = np.linalg.norm(submovement_direction)
            reduced_submovement_magnitude = submovement_magnitude * 0.7
            submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        self.head_current_goal_position = p0_not_normalized + submovement_direction
        # find the speed
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05,
                            avg_saccade_interval=0.5):
        output_list = []
        rig_factor = 10  # I believe it should be rig_factor[130] and rig_factor[131],
        max_saccade_x = 2 * saccade_factor
        max_saccade_y = 2 * saccade_factor
        prev_saccade_frame_counter = prev_saccade_frame
        if prev_saccade_frame <= start_frame:
            saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
            saccade_duration = 1.0 / 24.0
            output_list.append([start_frame, 0, 0])
            prev_saccade_frame_counter = start_frame + saccade_interval
            output_list.append([prev_saccade_frame_counter, 0, 0])
            prev_saccade_frame_counter += saccade_duration
        while prev_saccade_frame_counter < end_frame:
            # compute offset
            offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
            offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
            saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval / 10.0
            saccade_duration = 1.0 / 24.0
            if prev_saccade_frame_counter + saccade_duration + saccade_interval >= end_frame:
                offset_x = 0
                offset_y = 0
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_interval
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_duration
        return output_list, prev_saccade_frame_counter
    def compute(self):
        # first compute the head/eye contribution of the gaze:
        self.target_positions_head = self.optimize_for_neck_angle(self.target_gaze_intervals_time, self.target_gaze_intervals_pos)        
        # self.target_positions_head = self.target_positions
        self.gaze_current_goal_position = self.target_gaze_intervals_pos[0]
        self.head_current_goal_position = self.target_gaze_intervals_pos[0]
        for i in range(0, self.head_positions.shape[0]):
            self.head_positions[i] = self.head_current_goal_position
            self.gaze_positions[i] = self.gaze_current_goal_position
        
        # add an end time to the sequence
        end_t = self.target_times[-1] + 10.0
        # store the current list of submovments that gets updated
        gaze_submovements = []  # use to store a list of existing submovements
        gaze_submovements_indexes = []  # use to store the index of each submovement
        head_submovements = []  # use to store a list of existing submovements
        head_submovements_indexes = []  # use to store the index of each submovement
        while self.t < end_t - self.simulation_dt/2:
            t_index = int(np.round(self.t / self.simulation_dt))
            # use to store the gaze and head submovements that have expired
            expired_gaze = []
            expired_head = []
            # update the gaze positions
            self.gaze_positions[t_index] = self.gaze_positions[max(t_index - 1, 0)]
            for i in range(0, len(gaze_submovements)):
                if t_index < gaze_submovements_indexes[i][1] and t_index >= gaze_submovements_indexes[i][0]:
                    self.gaze_positions[t_index] += gaze_submovements[i][t_index - gaze_submovements_indexes[i][0]]
                elif t_index < gaze_submovements_indexes[i][0]:
                    pass
                else:
                    expired_gaze.append(i)
            # update the head positions
            self.head_positions[t_index] = self.head_positions[max(t_index - 1, 0)]
            # look at all the submovements and find the most recent one:
            if len(head_submovements) > 0:
                most_recent_head_submovement = head_submovements[-1]
                most_recent_head_submovement_dir = np.sum(most_recent_head_submovement, axis=0)
                # see if any submovement goes in a different direction
                dir_change = False
                for i in range(len(head_submovements)):
                    submovement_dir = head_submovements[i].sum(axis=0)
                    if submovement_dir[1] * most_recent_head_submovement_dir[1] < 0:
                        dir_change = True
                # if we need to change direction
                if dir_change:   
                    # remove all head_submovements
                    head_submovements = []
                    head_submovements_indexes = []
                    # add a new head submovement from current position to the old goal position
                    head_submovement, head_submovement_range = self.add_head_submovement(self.t, self.t + self.head_movement_duration, self.head_positions[t_index], self.interpolate_head_goal(self.t))
                    if not head_submovement is None:
                        head_submovements.append(head_submovement)
                        head_submovements_indexes.append(head_submovement_range)
            
            for i in range(0, len(head_submovements)):
                if t_index < head_submovements_indexes[i][1] and t_index >= head_submovements_indexes[i][0]:
                    self.head_positions[t_index] += head_submovements[i][t_index - head_submovements_indexes[i][0]]
                elif t_index < head_submovements_indexes[i][0]:
                    pass
                else:
                    expired_head.append(i)
            # only generate saccade at fixed intervals i.e. if saccade_generation_test is an integer
            saccade_generation_test = self.t / self.submovement_dt
            if abs(saccade_generation_test - int(round(saccade_generation_test))) <= 0.001:
                # obtain gaze shift duration as per properties of main sequence
                gaze_movement_duration = self.get_saccade_duration(self.gaze_current_goal_position,
                                                                   self.interpolate_gaze_goal(self.t))
                # add the gaze submovement
                next_gaze_index = int(self.interpolate_gaze_goal_index(self.t))
                next_gaze_goal = self.internal_model.estimate_target_pose(next_gaze_index)
                gaze_submovement, gaze_submovement_range = self.add_gaze_submovement(self.t,
                                                                                        self.t + gaze_movement_duration,
                                                                                        self.gaze_current_goal_position,
                                                                                        next_gaze_goal)
                if not gaze_submovement is None:
                    gaze_submovements.append(gaze_submovement)
                    gaze_submovements_indexes.append(gaze_submovement_range)
                    self.gaze_positions[t_index] += gaze_submovement[0]
                # obtain head shift duration
                head_movement_duration = self.head_movement_duration
                # get angle between two vectors and do something if the angle is small
                if np.linalg.norm(self.head_current_goal_position - self.interpolate_head_goal(self.t)) > 1E-3:
                    magnitude1 = np.linalg.norm(self.head_current_goal_position)
                    magnitude2 = np.linalg.norm(self.interpolate_head_goal(self.t))
                    dot_product = np.dot(self.head_current_goal_position, self.interpolate_head_goal(self.t))
                    # Calculate the cosine of the angle between the vectors
                    cosine_angle = dot_product / (magnitude1 * magnitude2)
                    # Use arccosine to find the angle in radians
                    angle_in_radians = np.arccos(cosine_angle)
                    # Convert the angle from radians to degrees
                    angle_in_degrees = np.degrees(angle_in_radians)
                    if angle_in_degrees <= 20:
                        # head_movement_duration = 0.3
                        # print(self.head_current_goal_position, self.interpolate_head_goal(self.t))                        
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        pass
                         
                # add the head movement
                head_delay = 0.05
                if len(head_submovements) > 0:
                    if len(head_submovements) >= 1:
                        head_delay = 0
                head_submovement, head_submovement_range = self.add_head_submovement(self.t + head_delay,
                                                                                     self.t + head_movement_duration + head_delay,
                                                                                     self.head_current_goal_position,
                                                                                     self.interpolate_head_goal(self.t))
                if not head_submovement is None:
                    head_submovements.append(head_submovement)
                    head_submovements_indexes.append(head_submovement_range)
                    self.head_positions[t_index] += head_submovement[0]

            # remove the submovements that have been expired, doing this backwards to not mess with indexing order
            for i in range(len(expired_head) - 1, -1, -1):
                head_submovements.pop(expired_head[i])
                head_submovements_indexes.pop(expired_head[i])
            # remove the submovements that have been expired
            for i in range(len(expired_gaze) - 1, -1, -1):
                gaze_submovements.pop(expired_gaze[i])
                gaze_submovements_indexes.pop(expired_gaze[i])


            # accumulate time
            self.t += self.simulation_dt
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0  # track the start of an interval
        end = 0  # track the end of an interval
        plottt = []
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i - 30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            plottt.append(speed_arr[i - 30:i].sum())
            end += self.simulation_dt
        prev_saccade = 0
        for i in range(0, len(stable_windows)):
            start = stable_windows[i][0]
            end = stable_windows[i][1]
            micro_saccade_list, prev_saccade = self.handle_microsaccade(start, prev_saccade, end)
            self.micro_saccade_kf.append(micro_saccade_list)
        return self.prepare_output()

    def prepare_output(self):
        eye_kf = []
        head_kf = []
        ts = np.arange(0, self.target_times[-1] + 10.0, self.simulation_dt)
        # insert the key frames for gaze into the output array
        for i in range(0, ts.shape[0]):
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]),
                           float(self.gaze_positions[i][2])])
        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(head_rotations[i][0]), float(head_rotations[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf
class Proposed_saccade_generator_with_graph:
    def interpolate_gaze_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_positions[i]
        print("Error")
    def interpolate_gaze_goal_index(self, t):
        if t <= self.target_times[0]:
            return self.target_index[0]
        elif t >= self.target_times[-1]:
            return self.target_index[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_index[i]
        print("Error")
    def interpolate_head_goal(self, t):
        if t < self.target_gaze_intervals_time[0][0]:
            return self.target_positions_head[0]
        elif t >= self.target_gaze_intervals_time[-1][0]:
            return self.target_positions_head[-1]
        else:
            for i in range(0, len(self.target_gaze_intervals_time) - 1):
                if self.target_gaze_intervals_time[i][0] <= t and self.target_gaze_intervals_time[i + 1][0] >= t:
                    return self.target_positions_head[i]
        print("Error")
    def get_gaze_intervals(self, time_arr, pos_arr):
        """
        Given two arrays, one a sequence of times, and the other a sequence of positions.
        This function segment the sequence into a sequence of intervals with trackable
        Start and end time.
        :param time_arr: an array of times shape is (N, )
        :param pos_arr: an array of positions shape is  (N, K)
        :return:
        gaze_intervals_time: a list of [start, end] times
        gaze_intervals_pos: a list of positions
        """
        gaze_intervals_time = []
        gaze_intervals_pos = []
        for i in range(0, len(time_arr)-1):
            gaze_intervals_time.append([time_arr[i], time_arr[i+1]])
            gaze_intervals_pos.append(np.expand_dims(pos_arr[i], 0))
        gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
        gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    
        gaze_intervals_time = []
        gaze_intervals_pos = []
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            # in the case that we are at the final time, and the current differs from previous
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) < 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                if i > 0:
                    gaze_intervals_time.append([start, time_arr[i]])
                    gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) == 0:
                gaze_intervals_time.append([start, time_arr[i]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            else:
                print("I wanna eat korean fried chicken")
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    def __init__(self, target_times, target_positions, target_index, internal_model, dt=0.025, reach_target_at_times=False, head_delay = 0.05):
        # gaze state variables:
        self.gaze_current_goal_position = internal_model.get_base_pose()
        self.head_current_goal_position = internal_model.get_base_pose()
        self.gaze_most_recent_index = 0
        self.head_most_recent_index = 0
        self.dire_change_delay = 0.0
        
        # meta parameters:
        self.simulation_dt = dt
        self.submovement_dt = 0.100 
        self.movement_threshold = 200  # use to detect intervals with no gaze shift. In which micro saccade are generated
        self.head_movement_duration = 0.3
        self.reach_target_at_times = reach_target_at_times
        self.head_delay = head_delay
        # Internal Model of the scene (will introduce bias, errors and everything)
        self.internal_model = internal_model

        # simulation state variables:
        self.target_times = target_times
        self.target_positions = np.array(target_positions)

        self.target_index = target_index
        self.target_positions_head = None
        
        self.target_gaze_intervals_time, self.target_gaze_intervals_pos = self.get_gaze_intervals(self.target_times, self.target_positions)
    
        self.t = 0

        # Initialize aray to store the state history (positions) i.e. the animation curve
        end_t = self.target_times[-1] + 10.0
        end_t = int(np.ceil(end_t / self.simulation_dt))
        # the memory will be allocated ahead of time since we know
        self.gaze_positions = internal_model.get_base_pose().astype(np.float32)
        self.gaze_positions = np.expand_dims(self.gaze_positions, axis=0)
        self.gaze_positions = np.tile(self.gaze_positions, [end_t, 1])
        self.gaze_pos_as_angles = rotation_angles_frome_positions(self.target_positions)
        self.input_gaze_pos_as_angles_per_frame = np.zeros((end_t, 2)) 
        min_index = int(np.min(self.target_index))
        for i in range(0, end_t):
            self.input_gaze_pos_as_angles_per_frame[i] = self.gaze_pos_as_angles[int(self.interpolate_gaze_goal_index(i * self.simulation_dt)) - min_index]
        self.input_gaze_pos_as_angles_per_frame_smoothed = np.array(laplacian_smoothing(self.input_gaze_pos_as_angles_per_frame, 100))   
        # plt.plot(self.input_gaze_pos_as_angles_per_frame[:, 0])
        # plt.plot(self.input_gaze_pos_as_angles_per_frame_smoothed[:, 0])
        # plt.show()
        
        self.head_positions = internal_model.get_base_pose().astype(np.float32)
        self.head_positions = np.expand_dims(self.head_positions, axis=0)
        self.head_positions = np.tile(self.head_positions, [end_t, 1])
        
        # set aside array to store the saccade keyframes
        self.micro_saccade_kf = []
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_frome_positions(pos0)
        rot1 = rotation_angles_frome_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        return duration / 1000
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(round(t0 / dt))
        tf = int(round(tf / dt))
        t = np.arange(t0, tf, 1)
        if t.shape[0] != 1:
            v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        else:
            v = np.array([1])
        v = v / v.sum()

        return v
    def add_gaze_submovement(self, t0, t1, p0, p1):
        # if there is nothing to do, do nothing
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # get the velocity profile of the submovement
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        # get the direction of the submovement
        submovement_direction = p1 - p0
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction
        # get the starting and ending frame of the submovmeent
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # update current gaze_goal position
        # print(p1 - p0, submovement.sum(axis=0), t0, t1)
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        p0 = p0_not_normalized / np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized / np.linalg.norm(p1_not_normalized) * 100
        # if the distance very close I don't make any submovements
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement
        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
            # threshold the rotation speed
            rot_angle = np.maximum(0.0, rot_angle * 0.7)
            # compute the actual rotation matrix
            rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
            # find the displacement using the rotation
            submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)
        # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        else:
            submovement_direction = (p1_not_normalized - p0_not_normalized)
            submovement_magnitude = np.linalg.norm(submovement_direction)
            reduced_submovement_magnitude = submovement_magnitude * 0.7
            submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        self.head_current_goal_position = p0_not_normalized + submovement_direction
        # find the speed
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05,
                            avg_saccade_interval=0.5):
        output_list = []
        rig_factor = 10  # I believe it should be rig_factor[130] and rig_factor[131],
        max_saccade_x = 2 * saccade_factor
        max_saccade_y = 2 * saccade_factor
        prev_saccade_frame_counter = prev_saccade_frame
        if prev_saccade_frame <= start_frame:
            saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
            saccade_duration = 1.0 / 24.0
            output_list.append([start_frame, 0, 0])
            prev_saccade_frame_counter = start_frame + saccade_interval
            output_list.append([prev_saccade_frame_counter, 0, 0])
            prev_saccade_frame_counter += saccade_duration
        while prev_saccade_frame_counter < end_frame:
            # compute offset
            offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
            offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
            saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval / 10.0
            saccade_duration = 1.0 / 24.0
            if prev_saccade_frame_counter + saccade_duration + saccade_interval >= end_frame:
                offset_x = 0
                offset_y = 0
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_interval
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_duration
        return output_list, prev_saccade_frame_counter
    def compute(self, head_sequence):
        # first compute the head/eye contribution of the gaze:
        self.target_positions_head = head_sequence        
        # self.target_positions_head = self.target_positions
        self.gaze_current_goal_position = self.target_gaze_intervals_pos[0]
        self.head_current_goal_position = self.target_gaze_intervals_pos[0]
        # print(self.head_current_goal_position.shape)
        for i in range(0, self.head_positions.shape[0]):
            self.head_positions[i] = self.head_current_goal_position
            self.gaze_positions[i] = self.gaze_current_goal_position
        
        # add an end time to the sequence
        end_t = self.target_times[-1] + 10.0
        # store the current list of submovments that gets updated
        gaze_submovements = []  # use to store a list of existing submovements
        gaze_submovements_indexes = []  # use to store the index of each submovement
        head_submovements = []  # use to store a list of existing submovements
        head_submovements_indexes = []  # use to store the index of each submovement
        while self.t < end_t - self.simulation_dt/2:
            t_index = int(np.round(self.t / self.simulation_dt))
            # use to store the gaze and head submovements that have expired
            expired_gaze = []
            expired_head = []
            # update the gaze positions
            self.gaze_positions[t_index] = self.gaze_positions[max(t_index - 1, 0)]
            for i in range(0, len(gaze_submovements)):
                if t_index < gaze_submovements_indexes[i][1] and t_index >= gaze_submovements_indexes[i][0]:
                    self.gaze_positions[t_index] += gaze_submovements[i][t_index - gaze_submovements_indexes[i][0]]
                elif t_index < gaze_submovements_indexes[i][0]:
                    pass
                else:
                    expired_gaze.append(i)
            # update the head positions
            self.head_positions[t_index] = self.head_positions[max(t_index - 1, 0)]
            # look at all the submovements and find the most recent one:
            if len(head_submovements) > 0:
                most_recent_head_submovement = head_submovements[-1]
                most_recent_head_submovement_dir = np.sum(most_recent_head_submovement, axis=0)
                # see if any submovement goes in a different direction
                dir_change = False
                for i in range(len(head_submovements)):
                    submovement_dir = head_submovements[i].sum(axis=0)
                    if submovement_dir[0] * most_recent_head_submovement_dir[0] < 0:
                        dir_change = True
                # if we need to change direction
                if dir_change:   
                    # remove all head_submovements
                    head_submovements = []
                    head_submovements_indexes = []
                    # add a new head submovement from current position to the old goal position
                    head_submovement, head_submovement_range = self.add_head_submovement(self.t, self.t + self.head_movement_duration, self.head_positions[t_index], self.interpolate_head_goal(self.t))
                    if not head_submovement is None:
                        head_submovements.append(head_submovement)
                        head_submovements_indexes.append(head_submovement_range)
            
            for i in range(0, len(head_submovements)):
                if t_index < head_submovements_indexes[i][1] and t_index >= head_submovements_indexes[i][0]:
                    self.head_positions[t_index] += head_submovements[i][t_index - head_submovements_indexes[i][0]]
                elif t_index < head_submovements_indexes[i][0]:
                    pass
                else:
                    expired_head.append(i)
            # only generate saccade at fixed intervals i.e. if saccade_generation_test is an integer
            saccade_generation_test = self.t / self.submovement_dt
            ahead = 0.0
            if abs(saccade_generation_test - int(round(saccade_generation_test))) <= 0.001:
                # obtain gaze shift duration as per properties of main sequence
                gaze_movement_duration = self.get_saccade_duration(self.gaze_current_goal_position,
                                                                   self.interpolate_gaze_goal(self.t))
                # add the gaze submovement
                next_gaze_index = int(self.interpolate_gaze_goal_index(self.t))
                next_gaze_goal = self.internal_model.estimate_target_pose(next_gaze_index)
                if self.reach_target_at_times:
                    ahead = gaze_movement_duration
                gaze_submovement, gaze_submovement_range = self.add_gaze_submovement(self.t - ahead,
                                                                                        self.t + gaze_movement_duration - ahead,
                                                                                        self.gaze_current_goal_position,
                                                                                        next_gaze_goal)
                if not gaze_submovement is None:
                    gaze_submovements.append(gaze_submovement)
                    gaze_submovements_indexes.append(gaze_submovement_range)
                    self.gaze_positions[t_index] += gaze_submovement[0]
                # obtain head shift duration
                head_movement_duration = self.head_movement_duration
                # get angle between two vectors and do something if the angle is small
                if np.linalg.norm(self.head_current_goal_position - self.interpolate_head_goal(self.t)) > 1E-3:
                    magnitude1 = np.linalg.norm(self.head_current_goal_position)
                    magnitude2 = np.linalg.norm(self.interpolate_head_goal(self.t))
                    dot_product = np.dot(self.head_current_goal_position, self.interpolate_head_goal(self.t))
                    # Calculate the cosine of the angle between the vectors
                    cosine_angle = dot_product / (magnitude1 * magnitude2)
                    # Use arccosine to find the angle in radians
                    angle_in_radians = np.arccos(cosine_angle)
                    # Convert the angle from radians to degrees
                    angle_in_degrees = np.degrees(angle_in_radians)
                    if angle_in_degrees <= 20:
                        # head_movement_duration = 0.3
                        # print(self.head_current_goal_position, self.interpolate_head_goal(self.t))                        
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        pass
                         
                # add the head movement
                if len(head_submovements) > 0:
                    if len(head_submovements) >= 1:
                        head_delay = 0
                head_submovement, head_submovement_range = self.add_head_submovement(self.t + self.head_delay - ahead,
                                                                                     self.t + head_movement_duration + self.head_delay - ahead,
                                                                                     self.head_current_goal_position,
                                                                                     self.interpolate_head_goal(self.t))
                if not head_submovement is None:
                    head_submovements.append(head_submovement)
                    head_submovements_indexes.append(head_submovement_range)
                    self.head_positions[t_index] += head_submovement[0]

            # remove the submovements that have been expired, doing this backwards to not mess with indexing order
            for i in range(len(expired_head) - 1, -1, -1):
                head_submovements.pop(expired_head[i])
                head_submovements_indexes.pop(expired_head[i])
            # remove the submovements that have been expired
            for i in range(len(expired_gaze) - 1, -1, -1):
                gaze_submovements.pop(expired_gaze[i])
                gaze_submovements_indexes.pop(expired_gaze[i])


            # accumulate time
            self.t += self.simulation_dt
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0  # track the start of an interval
        end = 0  # track the end of an interval
        plottt = []
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i - 30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            plottt.append(speed_arr[i - 30:i].sum())
            end += self.simulation_dt
        prev_saccade = 0
        for i in range(0, len(stable_windows)):
            start = stable_windows[i][0]
            end = stable_windows[i][1]
            micro_saccade_list, prev_saccade = self.handle_microsaccade(start, prev_saccade, end)
            self.micro_saccade_kf.append(micro_saccade_list)
        return self.prepare_output()
    def prepare_output(self):
        eye_kf = []
        head_kf = []
        ts = np.arange(0, self.target_times[-1] + 10.0, self.simulation_dt)
        # insert the key frames for gaze into the output array
        for i in range(0, ts.shape[0]):
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]),
                           float(self.gaze_positions[i][2])])
        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(head_rotations[i][0]), float(head_rotations[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf

class Proposed_saccade_generator_with_graph_smooth_pursuit:
    def interpolate_gaze_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_positions[i]
        print("Error")
    def interpolate_gaze_goal_index(self, t):
        if t <= self.target_times[0]:
            return self.target_index[0]
        elif t >= self.target_times[-1]:
            return self.target_index[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_index[i]
        print("Error")
    def interpolate_head_goal(self, t):
        if t < self.target_gaze_intervals_time[0][0]:
            return self.target_positions_head[0]
        elif t >= self.target_gaze_intervals_time[-1][0]:
            return self.target_positions_head[-1]
        else:
            for i in range(0, len(self.target_gaze_intervals_time) - 1):
                if self.target_gaze_intervals_time[i][0] <= t and self.target_gaze_intervals_time[i + 1][0] >= t:
                    return self.target_positions_head[i]
        print("Error")
    def get_gaze_intervals(self, time_arr, pos_arr):
        """
        Given two arrays, one a sequence of times, and the other a sequence of positions.
        This function segment the sequence into a sequence of intervals with trackable
        Start and end time.
        :param time_arr: an array of times shape is (N, )
        :param pos_arr: an array of positions shape is  (N, K)
        :return:
        gaze_intervals_time: a list of [start, end] times
        gaze_intervals_pos: a list of positions
        """
        gaze_intervals_time = []
        gaze_intervals_pos = []
        for i in range(0, len(time_arr)-1):
            gaze_intervals_time.append([time_arr[i], time_arr[i+1]])
            gaze_intervals_pos.append(np.expand_dims(pos_arr[i], 0))
        gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
        gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    
        gaze_intervals_time = []
        gaze_intervals_pos = []
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            # in the case that we are at the final time, and the current differs from previous
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) < 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                if i > 0:
                    gaze_intervals_time.append([start, time_arr[i]])
                    gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) == 0:
                gaze_intervals_time.append([start, time_arr[i]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            else:
                print("I wanna eat korean fried chicken")
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    def __init__(self, target_times, target_positions, target_index, internal_model, dt=0.025, reach_target_at_times=False, head_delay = 0.05):
        # gaze state variables:
        self.gaze_current_goal_position = internal_model.get_base_pose()
        self.head_current_goal_position = internal_model.get_base_pose()
        self.gaze_most_recent_index = 0
        self.head_most_recent_index = 0
        self.dire_change_delay = 0.0
        
        # meta parameters:
        self.simulation_dt = dt
        self.submovement_dt = 0.200 
        self.movement_threshold = 200  # use to detect intervals with no gaze shift. In which micro saccade are generated
        self.head_movement_duration = 0.3
        self.reach_target_at_times = reach_target_at_times
        self.head_delay = head_delay
        # Internal Model of the scene (will introduce bias, errors and everything)
        self.internal_model = internal_model

        # simulation state variables:
        self.target_times = target_times
        self.target_positions = np.array(target_positions)

        self.target_index = target_index
        self.target_positions_head = None
        
        self.target_gaze_intervals_time, self.target_gaze_intervals_pos = self.get_gaze_intervals(self.target_times, self.target_positions)
    
        self.t = 0

        # Initialize aray to store the state history (positions) i.e. the animation curve
        end_t = self.target_times[-1] + 0.1
        end_t = int(np.ceil(end_t / self.simulation_dt))
        # the memory will be allocated ahead of time since we know
        self.gaze_positions = internal_model.get_base_pose().astype(np.float32)
        self.gaze_positions = np.expand_dims(self.gaze_positions, axis=0)
        self.gaze_positions = np.tile(self.gaze_positions, [end_t, 1])
        self.gaze_pos_as_angles = rotation_angles_frome_positions(self.target_positions)
        self.input_gaze_pos_as_angles_per_frame = np.zeros((end_t, 2)) 
        min_index = int(np.min(self.target_index))
        for i in range(0, end_t):
            self.input_gaze_pos_as_angles_per_frame[i] = self.gaze_pos_as_angles[int(self.interpolate_gaze_goal_index(i * self.simulation_dt)) - min_index]
        self.input_gaze_pos_as_angles_per_frame_smoothed = np.array(laplacian_smoothing(self.input_gaze_pos_as_angles_per_frame, 100))   
        # plt.plot(self.input_gaze_pos_as_angles_per_frame[:, 0])
        # plt.plot(self.input_gaze_pos_as_angles_per_frame_smoothed[:, 0])
        # plt.show()
        
        self.head_positions = internal_model.get_base_pose().astype(np.float32)
        self.head_positions = np.expand_dims(self.head_positions, axis=0)
        self.head_positions = np.tile(self.head_positions, [end_t, 1])
        
        # set aside array to store the saccade keyframes
        self.micro_saccade_kf = []
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_frome_positions(pos0)
        rot1 = rotation_angles_frome_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        return duration / 1000
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(round(t0 / dt))
        tf = int(round(tf / dt))
        t = np.arange(t0, tf, 1)
        if t.shape[0] != 1:
            v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        else:
            v = np.array([1])
        v = v / v.sum()

        return v
    def add_gaze_submovement(self, t0, t1, p0, p1):
        # if there is nothing to do, do nothing
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # get the velocity profile of the submovement
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        # get the direction of the submovement
        submovement_direction = p1 - p0
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction
        # get the starting and ending frame of the submovmeent
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # update current gaze_goal position
        # print(p1 - p0, submovement.sum(axis=0), t0, t1)
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        p0 = p0_not_normalized / np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized / np.linalg.norm(p1_not_normalized) * 100
        # if the distance very close I don't make any submovements
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement
        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
            # threshold the rotation speed
            rot_angle = np.maximum(0.0, rot_angle * 0.7)
            # compute the actual rotation matrix
            rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
            # find the displacement using the rotation
            submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)
        # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        else:
            submovement_direction = (p1_not_normalized - p0_not_normalized)
            submovement_magnitude = np.linalg.norm(submovement_direction)
            reduced_submovement_magnitude = submovement_magnitude * 0.7
            submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        self.head_current_goal_position = p0_not_normalized + submovement_direction
        # find the speed
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05,
                            avg_saccade_interval=0.5):
        output_list = []
        rig_factor = 10  # I believe it should be rig_factor[130] and rig_factor[131],
        max_saccade_x = 2 * saccade_factor
        max_saccade_y = 2 * saccade_factor
        prev_saccade_frame_counter = prev_saccade_frame
        if prev_saccade_frame <= start_frame:
            saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
            saccade_duration = 1.0 / 24.0
            output_list.append([start_frame, 0, 0])
            prev_saccade_frame_counter = start_frame + saccade_interval
            output_list.append([prev_saccade_frame_counter, 0, 0])
            prev_saccade_frame_counter += saccade_duration
        while prev_saccade_frame_counter < end_frame:
            # compute offset
            offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
            offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
            saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval / 10.0
            saccade_duration = 1.0 / 24.0
            if prev_saccade_frame_counter + saccade_duration + saccade_interval >= end_frame:
                offset_x = 0
                offset_y = 0
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_interval
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_duration
        return output_list, prev_saccade_frame_counter
    def compute(self, head_sequence):
        # first compute the head/eye contribution of the gaze:
        self.target_positions_head = head_sequence        
        # self.target_positions_head = self.target_positions
        self.gaze_current_goal_position = self.target_gaze_intervals_pos[0]
        self.head_current_goal_position = self.target_gaze_intervals_pos[0]
        # use a for loop to initialize the gaze and head positions
        for i in range(0, self.head_positions.shape[0]):
            self.head_positions[i] = self.head_current_goal_position
            self.gaze_positions[i] = self.gaze_current_goal_position
        
        # add an end time to the sequence
        end_t = self.target_times[-1] + 0.1
        # store the current list of submovments that gets updated
        gaze_submovements = []  # use to store a list of existing submovements
        gaze_submovements_indexes = []  # use to store the index of each submovement
        head_submovements = []  # use to store a list of existing submovements
        head_submovements_indexes = []  # use to store the index of each submovement
        while self.t < end_t - self.simulation_dt/2:
            # get the current time index (i.e. index that corresponds to the current time)
            t_index = int(np.round(self.t / self.simulation_dt))
            # use to store the gaze and head submovements that have expired
            expired_gaze = []
            expired_head = []
            # update the gaze positions
            self.gaze_positions[t_index] = self.gaze_positions[max(t_index - 1, 0)] # initialized as the previous position
            for i in range(0, len(gaze_submovements)):
                if t_index < gaze_submovements_indexes[i][1] and t_index >= gaze_submovements_indexes[i][0]:
                    self.gaze_positions[t_index] += gaze_submovements[i][t_index - gaze_submovements_indexes[i][0]]
                elif t_index < gaze_submovements_indexes[i][0]:
                    pass
                else:
                    expired_gaze.append(i)    
            
            # update the head positions
            self.head_positions[t_index] = self.head_positions[max(t_index - 1, 0)]
            # look at all the submovements and find the most recent one:
            if len(head_submovements) > 0:
                most_recent_head_submovement = head_submovements[-1]
                most_recent_head_submovement_dir = np.sum(most_recent_head_submovement, axis=0)
                # see if any submovement goes in a different direction
                dir_change = False
                for i in range(len(head_submovements)):
                    submovement_dir = head_submovements[i].sum(axis=0)
                    if submovement_dir[0] * most_recent_head_submovement_dir[0] < 0:
                        dir_change = True
                # if we need to change direction
                if dir_change:   
                    # remove all head_submovements
                    head_submovements = []
                    head_submovements_indexes = []
                    # add a new head submovement from current position to the old goal position
                    head_submovement, head_submovement_range = self.add_head_submovement(self.t, self.t + self.head_movement_duration, self.head_positions[t_index], self.interpolate_head_goal(self.t))
                    if not head_submovement is None:
                        head_submovements.append(head_submovement)
                        head_submovements_indexes.append(head_submovement_range)
            
            for i in range(0, len(head_submovements)):
                if t_index < head_submovements_indexes[i][1] and t_index >= head_submovements_indexes[i][0]:
                    self.head_positions[t_index] += head_submovements[i][t_index - head_submovements_indexes[i][0]]
                elif t_index < head_submovements_indexes[i][0]:
                    pass
                else:
                    expired_head.append(i)

            # generate smooth pursuit behviour for eye:
            eye_goal_position = self.internal_model.estimate_target_pose(self.t)
            eye_current_position = self.gaze_positions[t_index]
            if np.linalg.norm(eye_goal_position - eye_current_position) >= 1E-2:
                eye_rot_axis, eye_rot_angle = rotation_axis_angle_from_vector(eye_current_position, eye_goal_position)
                eye_rot_angle_speed = eye_rot_angle / self.simulation_dt
                # normalize it so that eye_rot_angle_speed is <= 100
                eye_rot_angle_speed = min(eye_rot_angle_speed, 100/180 * np.pi)
                eye_rot_matrix = rotation_matrix_from_axis_angle(eye_rot_axis, eye_rot_angle_speed * self.simulation_dt)
                eye_movement_displacement = (eye_rot_matrix - np.eye(3)) @ (eye_current_position)
                # add it to movement
                self.gaze_positions[t_index] = eye_current_position + eye_movement_displacement
                self.gaze_current_goal_position += eye_movement_displacement
            
            # generate smooth pursuit behaviour for head
            head_goal_position = self.internal_model.estimate_target_pose(self.t)
            head_current_position = self.head_positions[t_index]
            if np.linalg.norm(head_goal_position - head_current_position) >= 1E-2:
                head_rot_axis, head_rot_angle = rotation_axis_angle_from_vector(head_current_position, head_goal_position)
                head_rot_angle_speed = head_rot_angle / self.simulation_dt
                # normalize it so that head_rot_angle_speed is <= 100
                head_rot_angle_speed = min(head_rot_angle_speed, 20/180 * np.pi)
                head_rot_matrix = rotation_matrix_from_axis_angle(head_rot_axis, head_rot_angle_speed * self.simulation_dt)
                head_movement_displacement = (head_rot_matrix - np.eye(3)) @ (head_current_position)
                # add it to movement
                self.head_positions[t_index] = head_current_position + head_movement_displacement
                self.head_current_goal_position += head_movement_displacement
            # only generate saccade at fixed intervals i.e. if saccade_generation_test is an integer
            saccade_generation_test = self.t / self.submovement_dt
            ahead = 0.0
            if abs(saccade_generation_test - int(round(saccade_generation_test))) <= 0.001:
                # obtain gaze shift duration as per properties of main sequence
                gaze_movement_duration = self.get_saccade_duration(self.gaze_current_goal_position,
                                                                   self.interpolate_gaze_goal(self.t))
                # add the gaze submovement
                next_gaze_index = int(self.interpolate_gaze_goal_index(self.t))
                # next_gaze_goal = self.internal_model.estimate_target_pose(next_gaze_index)
                next_gaze_goal = self.internal_model.estimate_target_pose(self.t)
                
                if self.reach_target_at_times:
                    ahead = gaze_movement_duration
                gaze_submovement, gaze_submovement_range = self.add_gaze_submovement(self.t - ahead,
                                                                                        self.t + gaze_movement_duration - ahead,
                                                                                        eye_current_position,
                                                                                        next_gaze_goal)
                if not gaze_submovement is None:
                    gaze_submovements.append(gaze_submovement)
                    gaze_submovements_indexes.append(gaze_submovement_range)
                    self.gaze_positions[t_index] += gaze_submovement[0]
                # obtain head shift duration
                head_movement_duration = self.head_movement_duration
                # get angle between two vectors and do something if the angle is small
                if np.linalg.norm(self.head_current_goal_position - self.interpolate_head_goal(self.t)) > 1E-3:
                    magnitude1 = np.linalg.norm(self.head_current_goal_position)
                    magnitude2 = np.linalg.norm(self.interpolate_head_goal(self.t))
                    dot_product = np.dot(self.head_current_goal_position, self.interpolate_head_goal(self.t))
                    # Calculate the cosine of the angle between the vectors
                    cosine_angle = dot_product / (magnitude1 * magnitude2)
                    # Use arccosine to find the angle in radians
                    angle_in_radians = np.arccos(cosine_angle)
                    # Convert the angle from radians to degrees
                    angle_in_degrees = np.degrees(angle_in_radians)
                    if angle_in_degrees <= 20:
                        # head_movement_duration = 0.3
                        # print(self.head_current_goal_position, self.interpolate_head_goal(self.t))                        
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        pass
                         
                # add the head movement
                if len(head_submovements) > 0:
                    if len(head_submovements) >= 1:
                        head_delay = 0
                head_submovement, head_submovement_range = self.add_head_submovement(self.t + self.head_delay - ahead,
                                                                                     self.t + head_movement_duration + self.head_delay - ahead,
                                                                                     head_current_position,
                                                                                     self.interpolate_head_goal(self.t))
                if not head_submovement is None:
                    head_submovements.append(head_submovement)
                    head_submovements_indexes.append(head_submovement_range)
                    self.head_positions[t_index] += head_submovement[0]

            # remove the submovements that have been expired, doing this backwards to not mess with indexing order
            for i in range(len(expired_head) - 1, -1, -1):
                head_submovements.pop(expired_head[i])
                head_submovements_indexes.pop(expired_head[i])
            # remove the submovements that have been expired
            for i in range(len(expired_gaze) - 1, -1, -1):
                gaze_submovements.pop(expired_gaze[i])
                gaze_submovements_indexes.pop(expired_gaze[i])


            # accumulate time
            self.t += self.simulation_dt
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0  # track the start of an interval
        end = 0  # track the end of an interval
        plottt = []
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i - 30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            plottt.append(speed_arr[i - 30:i].sum())
            end += self.simulation_dt
        prev_saccade = 0
        for i in range(0, len(stable_windows)):
            start = stable_windows[i][0]
            end = stable_windows[i][1]
            micro_saccade_list, prev_saccade = self.handle_microsaccade(start, prev_saccade, end)
            self.micro_saccade_kf.append(micro_saccade_list)
        plt.plot(self.gaze_positions[:, 0])
        plt.plot(self.gaze_positions[:, 1])
        plt.plot(self.gaze_positions[:, 2])
        plt.show()
        plt.plot(self.head_positions[:, 0])
        plt.plot(self.head_positions[:, 1])
        plt.plot(self.head_positions[:, 2])
        plt.show()
        return self.prepare_output()
    def prepare_output(self):
        eye_kf = []
        head_kf = []
        ts = np.arange(0, self.target_times[-1] + 0.1, self.simulation_dt)
        # insert the key frames for gaze into the output array
        for i in range(0, ts.shape[0]):
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]),
                           float(self.gaze_positions[i][2])])
        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(head_rotations[i][0]), float(head_rotations[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf




if __name__ == "__main__":
    
    np.random.seed(0)
    # inputs
    scene_data_path = "C:/Users/evansamaa/Documents/GitHub/Gaze_project/data/look_at_points/simplest_scene2_less_items.json"
    agentScene1 = AgentInfo(scene_data_path)
    target_times = [0.0, 0.08, 0.27, 0.42, 0.55, 0.85, 0.91, 1.06, 1.31, 1.65, 1.68, 2.0, 2.36, 2.62, 2.87, 2.97, 3.41,
                    3.61, 3.86, 4.16, 4.31, 4.65, 4.72, 5.13, 5.69, 5.72, 6.08, 6.81, 7.39, 7.61, 7.78, 7.97, 8.19,
                    8.28, 8.52, 8.83, 8.93, 9.07, 9.21, 9.46, 9.58, 9.89, 9.93, 10.39, 10.71, 11.07, 11.15, 11.44, 11.5,
                    11.63, 11.93, 11.99, 12.31, 12.43, 12.79, 13.16, 13.32, 13.57, 13.88, 14.16, 14.25, 14.52, 14.6,
                    14.94]
    target_location = [[18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [77.6309886957713, -95.75313544830048, 66.38282084600831],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [149.60678466738037, -52.506574840153036, 125.18509402113789],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [99.19101647520384, -131.92320385473886, 110.48347469966897],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [-392.0731411936236, -63.456958472377835, 90.06554001765248],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [-76.27964963815825, -162.53685153909328, 126.67698267508115],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665],
                       [18.606098592763445, 0.9596711559607343, 161.84341837309665]]
    target_location = np.array(target_location)
    internal_model = InternalModelCenterBias(agentScene1)
    from numpy import array
    target_times = [0, 0, 0.4, 1.4000000000000001, 4.6000000000000005, 6.4, 7.6000000000000005, 7.7, 9.0, 10.9, 11.700000000000001, 11.8, 14.700000000000001, 15.8, 15.9, 17.400000000000002, 18.8, 19.900000000000002, 20.6, 21.3, 22.8, 25.3, 25.400000000000002, 25.5, 27.8, 28.900000000000002, 30.400000000000002, 31.5, 34.6, 35.0, 37.2, 38.5, 38.6, 42.300000000000004, 42.5, 42.800000000000004, 43.5, 44.6, 47.0, 49.300000000000004, 49.5, 50.7, 52.7, 54.1, 54.2, 56.7, 57.800000000000004, 58.4, 59.800000000000004, 59.9, 69.2, 70.60000000000001, 72.60000000000001, 72.7, 73.9, 76.0, 76.5, 76.60000000000001, 78.60000000000001, 78.7, 80.60000000000001, 82.7, 82.9, 84.5, 86.10000000000001, 86.9, 87.7, 91.7, 93.4, 93.5, 93.7, 94.9, 97.9, 98.0, 100.2, 101.0, 101.10000000000001, 102.8, 104.3, 106.5, 107.5, 109.7, 110.7, 111.7, 111.8, 113.2, 115.2, 115.3, 115.7, 116.0, 117.9, 118.7, 118.8]
    target_location = [array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([  99.19217314, -134.72294727,   98.83931537]), array([  99.19217314, -134.72294727,   98.83931537]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([  99.19217314, -134.72294727,   98.83931537]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ -76.25453769, -165.51713441,  114.94985496]), array([  99.19217314, -134.72294727,   98.83931537]), array([-392.1011579 ,  -66.03683609,   79.91651373]), array([ -76.25453769, -165.51713441,  114.94985496]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ -76.25453769, -165.51713441,  114.94985496]), array([  0.        , -34.20201433,  93.96926208]), array([  99.19217314, -134.72294727,   98.83931537]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ -76.25453769, -165.51713441,  114.94985496]), array([-135.88727278,  -14.58413319,   75.10063517]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([-135.88727278,  -14.58413319,   75.10063517]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([  99.19217314, -134.72294727,   98.83931537]), array([149.6303476 , -55.47518358, 114.35288247]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([149.6303476 , -55.47518358, 114.35288247]), array([-135.88727278,  -14.58413319,   75.10063517]), array([-392.1011579 ,  -66.03683609,   79.91651373]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([ -76.25453769, -165.51713441,  114.94985496]), array([149.6303476 , -55.47518358, 114.35288247]), array([  0.        , -34.20201433,  93.96926208]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([-135.88727278,  -14.58413319,   75.10063517]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([149.6303476 , -55.47518358, 114.35288247]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ -76.25453769, -165.51713441,  114.94985496]), array([  99.19217314, -134.72294727,   98.83931537]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-135.88727278,  -14.58413319,   75.10063517]), array([-135.88727278,  -14.58413319,   75.10063517]), array([  99.19217314, -134.72294727,   98.83931537]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([-392.1011579 ,  -66.03683609,   79.91651373]), array([-135.88727278,  -14.58413319,   75.10063517]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([  99.19217314, -134.72294727,   98.83931537]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-392.1011579 ,  -66.03683609,   79.91651373]), array([-135.88727278,  -14.58413319,   75.10063517]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([-135.88727278,  -14.58413319,   75.10063517]), array([ 77.5670647 , -98.062397  ,  55.17771834]), array([ -76.25453769, -165.51713441,  114.94985496]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([149.6303476 , -55.47518358, 114.35288247]), array([-86.57015776,  19.54535685, 130.97888255]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([  0.        , -34.20201433,  93.96926208]), array([-392.1011579 ,  -66.03683609,   79.91651373]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([  99.19217314, -134.72294727,   98.83931537]), array([  0.        , -34.20201433,  93.96926208]), array([-135.88727278,  -14.58413319,   75.10063517]), array([-135.88727278,  -14.58413319,   75.10063517]), array([ 18.60609859, 163.4965227 , 152.23135916]), array([ 77.5670647 , -98.062397  ,  55.17771834])] 
    target_index = [7, 7, 2, 4, 7, 2, 7, 2, 6, 6, 7, 4, 6, 7, 1, 6, 5, 1, 7, 1, 8, 6, 7, 1, 0, 7, 2, 4, 7, 4, 0, 7, 4, 2, 7, 6, 3, 4, 7, 7, 3, 0, 5, 2, 7, 2, 1, 3, 8, 7, 2, 0, 7, 2, 4, 3, 7, 2, 7, 1, 6, 7, 0, 0, 6, 7, 4, 5, 0, 7, 6, 4, 7, 5, 0, 2, 7, 0, 4, 1, 7, 3, 2, 7, 8, 5, 7, 6, 8, 0, 0, 7, 4]
    generator = SacccadeGenerator(target_times, target_location, target_index, internal_model)
    ek, hk, micro_saccade = generator.compute()
    # simuluation state variables
