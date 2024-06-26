import pickle as pkl
import math
import json

def load_juggling_animation(json_path):
    json_data = json.load(open(json_path, 'r'))
    time = json_data['time']
    look_at = json_data["look_at_indices"]
    ball_positions = []
    for i in range(len(json_data.keys())-2):
        ball_positions.append(json_data["ball_{}_positions".format(i)])
    # ball indices
    ball_indices = set([0, 1, 2])
    print(ball_positions)
    for i in range(len(ball_positions)):     
        ball_path = ball_positions[i]
        sp, _ = cmds.sphere(r = 1.5, pivot =[0, 0, 0])
        for index_t in range(len(time)):
            cmds.setKeyframe(sp, attribute = 'translateX', value = ball_path[index_t][0], time = 30 * time[index_t])
            cmds.setKeyframe(sp, attribute = 'translateY', value = 160 + ball_path[index_t][1], time = 30 * time[index_t])        
            cmds.setKeyframe(sp, attribute = 'translateZ', value = 30, time = 30 * time[index_t])        
            
            ball_index_looking_at = look_at[index_t]
            if i == ball_index_looking_at:
                # keyframe the ball to be red:
                cmds.setKeyframe(sp, attribute = 'scaleX', value = 1.5, time = 30 * time[index_t])
                cmds.setKeyframe(sp, attribute = 'scaleY', value = 1.5, time = 30 * time[index_t])
                cmds.setKeyframe(sp, attribute = 'scaleZ', value = 1.5, time = 30 * time[index_t])
            else:
                # otherwise set it to blue
                cmds.setKeyframe(sp, attribute = 'scaleX', value = 1, time = 30 * time[index_t])
                cmds.setKeyframe(sp, attribute = 'scaleY', value = 1, time = 30 * time[index_t])
                cmds.setKeyframe(sp, attribute = 'scaleZ', value = 1, time = 30 * time[index_t])
                
# load_juggling_animation("C://Users//evan1//OneDrive//Documents//GitHub//EyeIK//Juggling//juggling_TAP_output//alternating_column_outputpositions.json")
load_juggling_animation("C:/Users/evansamaa/Documents/GitHub/EyeIK/Reading/reading_output/reading_ver1.pkl")