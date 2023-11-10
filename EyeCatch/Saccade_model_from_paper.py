import numpy as np
from matplotlib import pyplot as plt
from prototypes.InputDataStructures import Dietic_Conversation_Gaze_Scene_Info
from Geometry_Util import rotation_angles_frome_positions, rotation_axis_angle_from_vector, rotation_matrix_from_axis_angle, rotation_matrix_from_vectors
from Signal_processing_utils import dx_dt


# class InternalModelCenterBias:
class InternalModelExact:
    def __init__(self, scene: Dietic_Conversation_Gaze_Scene_Info):
        self.scene = scene
    def estimate_target_pose(self, index):
        return self.scene.object_pos[index]
    def get_base_pose(self):
        return self.scene.speaker_face_direction_local
class SacccadeGenerator:
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
        self.gaze_current_goal_position = internal_model.get_base_pose()
        self.head_current_goal_position = internal_model.get_base_pose()
        self.gaze_most_recent_index = 0
        self.head_most_recent_index = 0

        # meta parameters:
        self.simulation_dt = 0.02
        self.submovement_dt = 0.200
        self.movement_threshold = 2000 # use to detect intervals with no gaze shift. In which micro saccade are generated

        # Internal Model
        self.internal_model = internal_model

        # simulation state variables:
        self.target_times = target_times
        self.target_positions = np.array(target_positions)
        self.target_index = target_index
        self.t = 0

        # entire state history (positions)
        end_t = self.target_times[-1] + 10.0
        end_t = int(np.round(end_t / self.simulation_dt))
        self.gaze_positions = internal_model.get_base_pose().astype(np.float32)
        self.gaze_positions = np.expand_dims(self.gaze_positions, axis = 0)
        self.gaze_positions = np.tile(self.gaze_positions, [end_t, 1])
        self.head_positions = internal_model.get_base_pose().astype(np.float32)
        self.head_positions = np.expand_dims(self.head_positions, axis=0)
        self.head_positions = np.tile(self.head_positions, [end_t, 1])
        self.micro_saccade_kf = []

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
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(t0 / dt)
        tf = int(tf / dt)
        t = np.arange(t0, tf, 1)
        v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
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
        p0 = p0_not_normalized/np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized/np.linalg.norm(p1_not_normalized) * 100
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
            rot_angle = np.maximum(0.0, rot_angle*0.7)
            # compute the actual rotation matrix
            rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
            # find the displacement using the rotation
            submovement_direction = (rot_matrix - np.eye(3)) @ (p0)
        # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        else:
            submovement_direction = (p1 - p0)
            submovement_magnitude = np.linalg.norm(submovement_direction)
            reduced_submovement_magnitude = np.maximum(0, submovement_magnitude*0.7)
            submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        self.head_current_goal_position = p0 + submovement_direction
        # find the speed
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
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
        # add an end time to the sequence
        end_t = self.target_times[-1] + 10.0

        # store the current list of submovments that gets updated
        gaze_submovements = [] # use to store a list of existing submovements
        gaze_submovements_indexes = [] # use to store the index of each submovement
        head_submovements = [] # use to store a list of existing submovements
        head_submovements_indexes = [] # use to store the index of each submovement
        while round(self.t) < end_t:
            t_index = int(np.round(self.t / self.simulation_dt))
            # use to store the gaze and head submovements that have expired
            expired_gaze = []
            expired_head = []
            # update the gaze positions
            self.gaze_positions[t_index] = self.gaze_positions[max(t_index - 1, 0)]
            for i in range(0, len(gaze_submovements)):
                if t_index < gaze_submovements_indexes[i][1]:
                    self.gaze_positions[t_index] += gaze_submovements[i][t_index - gaze_submovements_indexes[i][0]]
                else:
                    expired_gaze.append(i)
            # update the head positions
            self.head_positions[t_index] = self.head_positions[max(t_index - 1, 0)]
            for i in range(0, len(head_submovements)):
                if t_index < head_submovements_indexes[i][1]:
                    self.head_positions[t_index] += head_submovements[i][t_index - head_submovements_indexes[i][0]]
                else:
                    expired_head.append(i)
            # only generate saccade at fixed intervals i.e. if saccade_generation_test is an integer
            saccade_generation_test = self.t / self.submovement_dt
            if abs(saccade_generation_test - int(round(saccade_generation_test))) <= 0.001:
                # obtain gaze shift duration as per properties of main sequence
                gaze_movement_duration = self.get_saccade_duration(self.gaze_current_goal_position, self.interpolate_goal(self.t))
                # add the gaze submovement
                gaze_submovement, gaze_submovement_range = self.add_gaze_submovement(self.t, self.t + gaze_movement_duration, self.gaze_current_goal_position, self.interpolate_goal(self.t))
                if not gaze_submovement is None:
                    gaze_submovements.append(gaze_submovement)
                    gaze_submovements_indexes.append(gaze_submovement_range)
                    self.gaze_positions[t_index] += gaze_submovement[0]

                # obtain head shift duration
                head_movement_duration = .4
                # add the head movement
                head_submovement, head_submovement_range = self.add_head_submovement(self.t, self.t + head_movement_duration, self.head_current_goal_position, self.gaze_current_goal_position)
                if not head_submovement is None:
                    head_submovements.append(head_submovement)
                    head_submovements_indexes.append(head_submovement_range)
                    self.head_positions[t_index] += head_submovement[0]

            # remove the submovements that have been expired, doing this backwards to not mess with indexing order
            for i in range(len(expired_head)-1, -1, -1):
                head_submovements.pop(expired_head[i])
                head_submovements_indexes.pop(expired_head[i])
            # remove the submovements that have been expired
            for i in range(len(expired_gaze)-1, -1, -1):
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
        print(stable_windows)
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
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]), float(self.gaze_positions[i][2])])

        # turn the head look at point into angles
        head_rotations = rotation_angles_frome_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(head_rotations[i][0]), float(head_rotations[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf






    


            






if __name__ == "__main__":
    target_times = [0.0, 0.08, 0.27, 0.42, 0.55, 0.85, 0.91, 1.06, 1.31, 1.65, 1.68, 2.0, 2.36, 2.62, 2.87, 2.97, 3.41, 3.61, 3.86, 4.16, 4.31, 4.65, 4.72, 5.13, 5.69, 5.72, 6.08, 6.81, 7.39, 7.61, 7.78, 7.97, 8.19, 8.28, 8.52, 8.83, 8.93, 9.07, 9.21, 9.46, 9.58, 9.89, 9.93, 10.39, 10.71, 11.07, 11.15, 11.44, 11.5, 11.63, 11.93, 11.99, 12.31, 12.43, 12.79, 13.16, 13.32, 13.57, 13.88, 14.16, 14.25, 14.52, 14.6, 14.94]
    target_location = [[18.606098592763445, 0.9596711559607343, 161.84341837309665], [77.6309886957713, -95.75313544830048, 66.38282084600831], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [149.60678466738037, -52.506574840153036, 125.18509402113789], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [99.19101647520384, -131.92320385473886, 110.48347469966897], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [-392.0731411936236, -63.456958472377835, 90.06554001765248], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [-76.27964963815825, -162.53685153909328, 126.67698267508115], [18.606098592763445, 0.9596711559607343, 161.84341837309665], [18.606098592763445, 0.9596711559607343, 161.84341837309665]]
    target_location = np.array(target_location)
    pass
    # simuluation state variables
