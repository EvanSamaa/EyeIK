import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy.stats import norm, gamma
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/EvansToolBox/Utils')
sys.path.insert(0, '/Users/evanpan/Documents/GitHub/Gaze_project')
sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/EvansToolBox/Utils")
sys.path.insert(0, "C:/Users/evansamaa/Documents/GitHub/Gaze_project")
sys.path.insert(0, "C:/Users/evan1/Documents/GitHub/EvansToolBox/Utils")
sys.path.insert(0, "C:/Users/evan1/Documents/GitHub/Gaze_project")

from Geometry_Util import rotation_angles_frome_positions, rotation_axis_angle_from_vector, rotation_matrix_from_axis_angle, rotation_matrix_from_vectors, directions_from_rotation_angles
from Signal_processing_utils import dx_dt
from InputStructures import InputStructure

# class InternalModelCenterBias:





class InternalModelExact:
    def __init__(self, scene: InputStructure):
        self.scene = scene
    def estimate_target_pose(self, index):
        return self.scene.object_pos[index]
    def get_base_pose(self):
        return self.scene.speaker_face_direction_local
class Goude_SacccadeGenerator_axis_angle:
    # helper functions
    def mock_fixation_model(self, ts, arr):
        # ts = np.arange(0, arr.shape[0]) / fps
        arr_interp = interp1d(ts, arr, axis=0, kind="previous", bounds_error=False, fill_value="extrapolate")
        def mock_fixation_point_grabber(t):
            return arr_interp(t)
        return mock_fixation_point_grabber
    def interpolate_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times)-1):
                if self.target_times[i] <= t and self.target_times[i+1] > t:
                    return self.target_positions[i]
        print("Error")
    def __init__(self, target_times, target_positions, target_index, internal_model):
        # gaze state variables:
        # video metadata
        self.target_times = target_times
        self.target_positions = target_positions
        self.target_index = target_index
        self.movement_threshold = 2000 # use to detect intervals with no gaze shift. In which micro saccade are generated
        self.internal_model = internal_model
        
    def extend_arrays(self, t0, t1):
        # here we assume that array that stores gaze position is always the same length as the array for head look at points
        t_end = self.gaze_positions.shape[0]
        # see if the gaze_position_array need to be extended
        total_frames_needed = int(t1 / self.simulation_dt)
        if total_frames_needed >= t_end:
            # if we need to extend the array, we extend it by a little bit longer than needed
            new_frames_added = total_frames_needed - t_end + 10
            new_gaze_arr = np.zeros((new_frames_added, 3))
            for i in range(0, new_frames_added):
                new_gaze_arr[i] = self.gaze_positions[self.gaze_most_recent_index]
            new_head_arr = np.zeros((new_frames_added, 3))
            for i in range(0, new_frames_added):
                new_head_arr[i] = self.head_positions[self.head_most_recent_index]
            self.gaze_positions = np.concatenate([self.gaze_positions, new_gaze_arr], axis=0)
            self.head_positions = np.concatenate([self.head_positions, new_head_arr], axis=0)
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_frome_positions(pos0)
        rot1 = rotation_angles_frome_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        return duration / 1000
    def get_head_movement_duration(self, pos0, pos1):
        return 0.4
    def head_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(t0 / dt)
        tf = int(tf / dt)
        t = np.arange(t0, tf, 1)
        v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        return v
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        x = np.arange(1, 7)
        v = 14 * np.exp(-np.pi/4 * (x - 3) ** 2)
        v = v / np.sum(v)        
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
        starting_frame = int(t0 / self.simulation_dt)
        ending_frame = int(t1 / self.simulation_dt)
        # update current gaze_goal position
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        return None, None
        p0 = p0_not_normalized/np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized/np.linalg.norm(p1_not_normalized) * 100
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement

        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # make sure the rotation angle is positive
        if rot_angle < 0:
            rot_angle = -rot_angle
            rot_axis = -rot_axis
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        # if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
        #     # threshold the rotation speed
        #     rot_angle = np.maximum(0.0, rot_angle*0.7)
        #     # compute the actual rotation matrix
        #     rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        #     # find the displacement using the rotation
        #     submovement_direction = (rot_matrix - np.eye(3)) @ (p0)
        # # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        # else:
        #     submovement_direction = (p1 - p0)
        #     submovement_magnitude = np.linalg.norm(submovement_direction)
        #     reduced_submovement_magnitude = np.maximum(0, submovement_magnitude*0.7)
        #     submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        # modify the rotation angle such that if it's less than 20 degree, do not turn head
        modified_rotation_angle = max(0, rot_angle - 0.349)
        # modified_rotation_angle = rot_angle
        skew_symmetric_matrix_rot_axis = np.array([[0, -rot_axis[2], rot_axis[1]], [rot_axis[2], 0, -rot_axis[0]], [-rot_axis[1], rot_axis[0], 0]])
        rot_matrix = np.eye(3) + np.sin(modified_rotation_angle) * skew_symmetric_matrix_rot_axis + (1 - np.cos(modified_rotation_angle)) * skew_symmetric_matrix_rot_axis @ skew_symmetric_matrix_rot_axis
        submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)

        self.head_current_goal_position = p0 + submovement_direction
        # find the speed
        submovement_speed = self.head_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)

        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(t0 / self.simulation_dt)
        ending_frame = int(t1 / self.simulation_dt)
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05, avg_saccade_interval=0.5):
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
        
        fps = 50
        dt = 1/fps
        self.simulation_dt = dt
        inhibition_tracking = []
        # here is where I got the salient points
        get_pt = self.mock_fixation_model(self.target_times, self.target_index)
        ts = np.arange(0, (self.target_times[-1] + 10.0) * fps) / fps
        plt.plot(get_pt(ts))
        # simulation variables
        counter = 0
        next_gaze_shift = 0 # track the time for the next gaze shift ()
        next_head_shift = 0 # track the time for the next gaze shift ()
        current_gaze_target = np.array([0, 0])
        current_gaze_pos = np.array([0, 0])
        current_gaze_vel = np.array([0, 0])
        current_head_target = np.array([0, 0]) 
        current_head_pos = np.array([0, 0])
        current_head_angular_speed = 0
        current_head_rotation_axis = np.array([0, 0, 0])
        current_head_vel = np.array([0, 0])
        
        
        
        gaze_start = False
        head_start = False
        # track outputs
        gaze_points = [np.expand_dims(current_gaze_pos, axis=0)]
        head_points = [np.expand_dims(current_head_pos, axis=0)]
        # Loop through all frames in the video
        vel = []
        # useful statistical samplers:
        T_h_sampler = norm(0.15, 0.1)
        X_h_sampler = norm(40, 5)
        Y_h_sampler = norm(15, 2)
        fixation_duration_sampler = gamma(1.2394, 0.1880)
        for counter in range(0, len(ts)):
            t = counter * dt
            # =========== For gaze shift =========== 
            fixation_point = rotation_angles_frome_positions(self.target_positions[int(np.round(get_pt(t)))])
            if np.linalg.norm(fixation_point - current_gaze_target) >= 0.001:
                # ========================= swap real saccadic model with get_pt(t) =========================
                fixation_point = rotation_angles_frome_positions(self.target_positions[int(np.round(get_pt(t)))])
                # ========================= swap real saccadic model with get_pt(t) =========================
                # define the new gaze target
                current_gaze_target = fixation_point
                # find the delayed for the head shift
                delay = T_h_sampler.rvs(1)[0] 
                next_head_shift = t + delay
                gaze_start = True
            # =========== For head rotation =========== 
            # find the angle between the current head angle and the gaze target
            head_vec = directions_from_rotation_angles(np.expand_dims(current_head_pos, axis=0), 1)[0]
            gaze_target = directions_from_rotation_angles(np.expand_dims(current_gaze_target, axis=0), 1)[0]
            # H_angle_length = np.linalg.norm(head_vec - gaze_target)
            # H_angle = np.arccos(0.5*((H_angle_length)**2) - 1) / np.pi * 180
            # H_angle = np.linalg.norm(current_head_pos - current_gaze_target)
            Hrot_axis, H_angle = rotation_axis_angle_from_vector(head_vec, gaze_target)
            if H_angle < 0:
                H_angle = -H_angle
                Hrot_axis = -Hrot_axis
            H_angle = H_angle / np.pi * 180
            # update a new head target when H_angle is sufficiently large
            if t > next_head_shift:
                # update a new head target when H_angle is sufficiently large
                X_h = X_h_sampler.rvs(1)[0]
                if H_angle > X_h:
                    head_start = True
                    current_head_target = current_gaze_target
                    # update this so this won't get updated
                next_head_shift = next_head_shift + 10000
            Y_h = Y_h_sampler.rvs(1)[0]
            if H_angle <= Y_h:
                head_start = False
            if gaze_start and np.linalg.norm(current_gaze_target - current_gaze_pos) > 0.001:
                current_gaze_vel = current_gaze_target - current_gaze_pos
                if np.linalg.norm(current_gaze_vel) > 0:
                    current_gaze_vel = current_gaze_vel / np.linalg.norm(current_gaze_vel) * 100
            else:
                current_gaze_vel = np.zeros([2])
                gaze_start = False
            if head_start:
                # compute acceleration vector
                head_accel = 30
        
                # if I were to use axis angle rotation and rodrigue's rotation formula
                current_head_angular_speed += head_accel * dt
                current_head_angular_speed = np.minimum(current_head_angular_speed, 40)
                current_head_rotation_axis = Hrot_axis
            else:
                # head will decelerate in the negative direction of the velocity
                head_deccell = -30
                # the deceleration is also 30 deg/s
                # compute new velocity
                new_head_vel = current_head_angular_speed + head_deccell * dt
                
                if np.dot(current_head_rotation_axis, Hrot_axis) < 0:
                    current_head_angular_speed = 0
                    current_head_rotation_axis = np.array([0, 0, 0])
                else:
                    current_head_angular_speed = new_head_vel
                    current_head_rotation_axis = Hrot_axis
            new_gaze_pos = current_gaze_pos + current_gaze_vel * dt
            if current_head_angular_speed > 0:
                # roate head_vec by current_head_angular_speed * dt along the rotation axis
                new_head_vec = rotation_matrix_from_axis_angle(current_head_rotation_axis, current_head_angular_speed * dt * np.pi/180) @ np.expand_dims(head_vec, axis=1)
                # convert direction back to angles:
                new_head_pos = rotation_angles_frome_positions(new_head_vec.T)[0]
            else:
                new_head_pos = current_head_pos
                
                # new_head_pos = current_head_pos + current_head_vel * dt
                
            # make sure the gaze do not over shoot
            if np.dot(current_gaze_target - new_gaze_pos, current_gaze_target - current_gaze_pos) < 0:
                gaze_start = False
                current_gaze_pos = current_gaze_target
            else:
                current_gaze_pos = new_gaze_pos
            # make sure head rotation do not over shoot
            if np.dot(current_head_target - new_head_pos, current_head_target - current_head_pos) < 0 and False:
                head_start = False
                current_head_pos = current_head_target
            else:
                current_head_pos = new_head_pos
            # print(np.linalg.norm(current_gaze_vel))
            gaze_points.append(np.expand_dims(current_gaze_pos, axis=0))
            head_points.append(np.expand_dims(current_head_pos, axis=0))

        
        self.gaze_positions = np.concatenate(gaze_points, axis=0)
        self.gaze_positions = directions_from_rotation_angles(self.gaze_positions, 100)
        self.head_positions = np.concatenate(head_points, axis=0)
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0 # track the start of an interval
        end = 0 # track the end of an interval
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i-30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            end += self.simulation_dt
        prev_saccade = 0
        self.micro_saccade_kf = []
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
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]), float(self.gaze_positions[i][2])])

        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(self.head_positions[i][0]), float(self.head_positions[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf
class Goude_SacccadeGenerator:
    # helper functions
    def mock_fixation_model(self, ts, arr):
        # ts = np.arange(0, arr.shape[0]) / fps
        arr_interp = interp1d(ts, arr, axis=0, kind="previous", bounds_error=False, fill_value="extrapolate")
        def mock_fixation_point_grabber(t):
            if t < ts[0]:
                return arr_interp(ts[0])
            return arr_interp(t)
        return mock_fixation_point_grabber
    def interpolate_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times)-1):
                if self.target_times[i] <= t and self.target_times[i+1] > t:
                    return self.target_positions[i]
        print("Error")
    def __init__(self, target_times, target_positions, target_index, internal_model):
        # gaze state variables:
        # video metadata
        self.target_times = target_times
        self.target_positions = target_positions
        self.target_index = target_index
        self.movement_threshold = 2000 # use to detect intervals with no gaze shift. In which micro saccade are generated
        self.internal_model = internal_model
        
    def extend_arrays(self, t0, t1):
        # here we assume that array that stores gaze position is always the same length as the array for head look at points
        t_end = self.gaze_positions.shape[0]
        # see if the gaze_position_array need to be extended
        total_frames_needed = int(t1 / self.simulation_dt)
        if total_frames_needed >= t_end:
            # if we need to extend the array, we extend it by a little bit longer than needed
            new_frames_added = total_frames_needed - t_end + 10
            new_gaze_arr = np.zeros((new_frames_added, 3))
            for i in range(0, new_frames_added):
                new_gaze_arr[i] = self.gaze_positions[self.gaze_most_recent_index]
            new_head_arr = np.zeros((new_frames_added, 3))
            for i in range(0, new_frames_added):
                new_head_arr[i] = self.head_positions[self.head_most_recent_index]
            self.gaze_positions = np.concatenate([self.gaze_positions, new_gaze_arr], axis=0)
            self.head_positions = np.concatenate([self.head_positions, new_head_arr], axis=0)
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_frome_positions(pos0)
        rot1 = rotation_angles_frome_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        return duration / 1000
    def get_head_movement_duration(self, pos0, pos1):
        return 0.4
    def head_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(t0 / dt)
        tf = int(tf / dt)
        t = np.arange(t0, tf, 1)
        v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        return v
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        x = np.arange(1, 7)
        v = 14 * np.exp(-np.pi/4 * (x - 3) ** 2)
        v = v / np.sum(v)        
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
        starting_frame = int(t0 / self.simulation_dt)
        ending_frame = int(t1 / self.simulation_dt)
        # update current gaze_goal position
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        return None, None
        p0 = p0_not_normalized/np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized/np.linalg.norm(p1_not_normalized) * 100
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement

        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # make sure the rotation angle is positive
        if rot_angle < 0:
            rot_angle = -rot_angle
            rot_axis = -rot_axis
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        # if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
        #     # threshold the rotation speed
        #     rot_angle = np.maximum(0.0, rot_angle*0.7)
        #     # compute the actual rotation matrix
        #     rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        #     # find the displacement using the rotation
        #     submovement_direction = (rot_matrix - np.eye(3)) @ (p0)
        # # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        # else:
        #     submovement_direction = (p1 - p0)
        #     submovement_magnitude = np.linalg.norm(submovement_direction)
        #     reduced_submovement_magnitude = np.maximum(0, submovement_magnitude*0.7)
        #     submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        # modify the rotation angle such that if it's less than 20 degree, do not turn head
        modified_rotation_angle = max(0, rot_angle - 0.349)
        # modified_rotation_angle = rot_angle
        skew_symmetric_matrix_rot_axis = np.array([[0, -rot_axis[2], rot_axis[1]], [rot_axis[2], 0, -rot_axis[0]], [-rot_axis[1], rot_axis[0], 0]])
        rot_matrix = np.eye(3) + np.sin(modified_rotation_angle) * skew_symmetric_matrix_rot_axis + (1 - np.cos(modified_rotation_angle)) * skew_symmetric_matrix_rot_axis @ skew_symmetric_matrix_rot_axis
        submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)

        self.head_current_goal_position = p0 + submovement_direction
        # find the speed
        submovement_speed = self.head_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)

        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(t0 / self.simulation_dt)
        ending_frame = int(t1 / self.simulation_dt)
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05, avg_saccade_interval=0.5):
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
        
        fps = 50
        dt = 1/fps
        self.simulation_dt = dt
        inhibition_tracking = []
        # here is where I got the salient points
        get_pt = self.mock_fixation_model(self.target_times, self.target_index)
        ts = np.arange(0, (self.target_times[-1] + 10.0) * fps) / fps
        # plt.plot(get_pt(ts))
        # simulation variables
        counter = 0
        next_gaze_shift = 0 # track the time for the next gaze shift ()
        next_head_shift = 0 # track the time for the next gaze shift ()
        current_gaze_target = np.array([0, 0])
        current_gaze_pos = np.array([0, 0])
        current_gaze_vel = np.array([0, 0])
        current_head_target = np.array([0, 0]) 
        current_head_pos = np.array([0, 0])
        current_head_angular_speed = 0
        current_head_rotation_axis = np.array([0, 0, 0])
        current_head_vel = np.array([0, 0])
        
        
        
        gaze_start = False
        head_start = False
        # track outputs
        gaze_points = [np.expand_dims(current_gaze_pos, axis=0)]
        head_points = [np.expand_dims(current_head_pos, axis=0)]
        # Loop through all frames in the video
        vel = []
        # useful statistical samplers:
        T_h_sampler = norm(0.15, 0.1)
        X_h_sampler = norm(40, 5)
        Y_h_sampler = norm(15, 2)
        fixation_duration_sampler = gamma(1.2394, 0.1880)
        for counter in range(0, len(ts)):
            t = counter * dt
            # =========== For gaze shift =========== 
            fixation_point = rotation_angles_frome_positions(self.target_positions[int(np.round(get_pt(t)))])
            if np.linalg.norm(fixation_point - current_gaze_target) >= 0.001:
                # ========================= swap real saccadic model with get_pt(t) =========================
                fixation_point = rotation_angles_frome_positions(self.target_positions[int(np.round(get_pt(t)))])
                # ========================= swap real saccadic model with get_pt(t) =========================
                # define the new gaze target
                current_gaze_target = fixation_point
                # find the delayed for the head shift
                delay = T_h_sampler.rvs(1)[0] 
                next_head_shift = t + delay
                gaze_start = True
            # =========== For head rotation =========== 
            # find the angle between the current head angle and the gaze target
            head_vec = directions_from_rotation_angles(np.expand_dims(current_head_pos, axis=0), 1)[0]
            gaze_target = directions_from_rotation_angles(np.expand_dims(current_gaze_target, axis=0), 1)[0]
            # H_angle_length = np.linalg.norm(head_vec - gaze_target)
            # H_angle = np.arccos(0.5*((H_angle_length)**2) - 1) / np.pi * 180
            # H_angle = np.linalg.norm(current_head_pos - current_gaze_target)
            Hrot_axis, H_angle = rotation_axis_angle_from_vector(head_vec, gaze_target)
            if H_angle < 0:
                H_angle = -H_angle
                Hrot_axis = -Hrot_axis
            H_angle = H_angle / np.pi * 180
            # update a new head target when H_angle is sufficiently large
            if t > next_head_shift:
                # update a new head target when H_angle is sufficiently large
                X_h = X_h_sampler.rvs(1)[0]
                if H_angle > X_h:
                    head_start = True
                    current_head_target = current_gaze_target
                    # update this so this won't get updated
                next_head_shift = next_head_shift + 10000
            Y_h = Y_h_sampler.rvs(1)[0]
            if H_angle <= Y_h:
                head_start = False
            if gaze_start and np.linalg.norm(current_gaze_target - current_gaze_pos) > 0.001:
                current_gaze_vel = current_gaze_target - current_gaze_pos
                if np.linalg.norm(current_gaze_vel) > 0:
                    current_gaze_vel = current_gaze_vel / np.linalg.norm(current_gaze_vel) * 100
            else:
                current_gaze_vel = np.zeros([2])
                gaze_start = False
            if head_start:
                # compute acceleration vector
                current_head_accel = current_head_target - current_head_pos
                # normalize the acceleration will be 30 deg/s^2
                if np.linalg.norm(current_head_accel) > 0:
                    current_head_accel = current_head_accel / np.linalg.norm(current_head_accel) * 30
                # compute head velocity based on acceleration
                current_head_vel = current_head_vel + current_head_accel * dt
                # velocity is normalized to be less than 40 deg/s
                head_vel_norm = np.linalg.norm(current_head_vel)
                clipped_head_vel_norm = np.minimum(head_vel_norm, 40)
                current_head_vel = current_head_vel / head_vel_norm * clipped_head_vel_norm
            else:
                # head will decelerate in the negative direction of the velocity
                head_decelleration = (-current_head_vel)
                # the deceleration is also 30 deg/s
                if np.linalg.norm(head_decelleration) > 0:
                    head_decelleration = head_decelleration / np.linalg.norm(head_decelleration) * 30
                # compute new velocity
                new_head_vel = current_head_vel + head_decelleration * dt
                if np.dot(new_head_vel, current_head_vel) < 0:
                    current_head_vel = 0
                else:
                    current_head_vel = new_head_vel
            new_gaze_pos = current_gaze_pos + current_gaze_vel * dt
            new_head_pos = current_head_pos + current_head_vel * dt
            # make sure the gaze do not over shoot
            if np.dot(current_gaze_target - new_gaze_pos, current_gaze_target - current_gaze_pos) < 0:
                gaze_start = False
                current_gaze_pos = current_gaze_target
            else:
                current_gaze_pos = new_gaze_pos
            # make sure head rotation do not over shoot
            if np.dot(current_head_target - new_head_pos, current_head_target - current_head_pos) < 0:
                head_start = False
                current_head_pos = current_head_target
            else:
                current_head_pos = new_head_pos
            # print(np.linalg.norm(current_gaze_vel))
            gaze_points.append(np.expand_dims(current_gaze_pos, axis=0))
            head_points.append(np.expand_dims(current_head_pos, axis=0))

        self.gaze_positions = np.concatenate(gaze_points, axis=0)
        self.gaze_positions = directions_from_rotation_angles(self.gaze_positions, 100)
        self.head_positions = np.concatenate(head_points, axis=0)
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0 # track the start of an interval
        end = 0 # track the end of an interval
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i-30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            end += self.simulation_dt
        prev_saccade = 0
        self.micro_saccade_kf = []
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
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]), float(self.gaze_positions[i][2])])

        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(self.head_positions[i][0]), float(self.head_positions[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf






    


            






if __name__ == "__main__":
    target_times = [0.0, 0.08, 0.27, 0.42, 0.55, 0.85, 0.91, 1.06, 1.31, 1.65, 1.68, 2.0, 2.36, 2.62, 2.87, 2.97, 3.41, 3.61, 3.86, 4.16, 4.31, 4.65, 4.72, 5.13, 5.69, 5.72, 6.08, 6.81, 7.39, 7.61, 7.78, 7.97, 8.19, 8.28, 8.52, 8.83, 8.93, 9.07, 9.21, 9.46, 9.58, 9.89, 9.93, 10.39, 10.71, 11.07, 11.15, 11.44, 11.5, 11.63, 11.93, 11.99, 12.31, 12.43, 12.79, 13.16, 13.32, 13.57, 13.88, 14.16, 14.25, 14.52, 14.6, 14.94]
    target_location = [[18.606098592763445, 0.9596711559607343, 161.84341837309665], [77.6309886957713, -95.75313544830048, 66.38282084600831], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [149.60678466738037, -52.506574840153036, 125.18509402113789], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [99.19101647520384, -131.92320385473886, 110.48347469966897], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [-392.0731411936236, -63.456958472377835, 90.06554001765248], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [-76.27964963815825, -162.53685153909328, 126.67698267508115], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665]]
    target_location = np.array(target_location)
    pass
    # simuluation state variables
