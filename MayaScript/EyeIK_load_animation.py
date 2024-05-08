import pickle as pkl
import math
import json

def create_neck_addition_node(rig):
    try:
        cmds.cutKey("jNeck_ctl.jNeck_xRotate")
        cmds.cutKey("jNeck_ctl.jNeck_yRotate")
        cmds.cutKey("jNeck_ctl.jNeck_zRotate")
    except:
        pass
    try:        
        cmds.delete("jSync1_Vanilla_B1")
    except:
        pass
    try:
        cmds.delete("jSync1_Vanilla_B2")
    except:
        pass
    try:
        cmds.delete("jSync1_Vanilla_B3")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_gaze")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_speech")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_blend")
    except:
        pass
    # the blend node to blend Jali neck and ambient neck
    try:
        cmds.delete("xneck_pairBlend_ambient_speech")
    except:
        pass
    try:
        cmds.delete("xneck_pairBlend_ambient")
    except:
        pass
    # make node for pairblend
    cmds.createNode('plusMinusAverage', n="xneck_pairBlend")
    cmds.createNode('plusMinusAverage', n="xneck_pairBlend_ambient_speech")
    
    # make node to store animation curves
    cmds.createNode('animCurveTL', n='xneck_pairBlend_gaze')
    cmds.createNode('animCurveTL', n='xneck_pairBlend_speech')
    cmds.createNode('animCurveTL', n='xneck_pairBlend_blend_speech1') # 1 is gaze, 0 is speech
    # this is for ambient and jali neck
    cmds.createNode('animCurveTL', n='xneck_pairBlend_blend_ambient_speech') # 1 is gaze, 0 is speech
    cmds.createNode('animCurveTL', n='xneck_pairBlend_ambient')
    
    # make them connection
    # cmds.connectAttr('xneck_pairBlend_blend_speech1.output', 'xneck_pairBlend.weight')
    cmds.connectAttr('xneck_pairBlend_speech.output', 'xneck_pairBlend_ambient_speech.input1D[0]')
    cmds.connectAttr('xneck_pairBlend_ambient.output', 'xneck_pairBlend_ambient_speech.input1D[1]')
    cmds.connectAttr('xneck_pairBlend_gaze.output', 'xneck_pairBlend.input1D[0]')    
    cmds.connectAttr('xneck_pairBlend_ambient_speech.output1D', 'xneck_pairBlend.input1D[1]')    
    cmds.connectAttr('xneck_pairBlend.output1D', 'jNeck_ctl.jNeck_xRotate')
    try:
        cmds.delete("yneck_pairBlend")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_gaze")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_speech")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_blend")
    except:
        pass
    # the blend node to blend Jali neck and ambient neck
    try:
        cmds.delete("yneck_pairBlend_ambient_speech")
    except:
        pass
    try:
        cmds.delete("yneck_pairBlend_ambient")
    except:
        pass
    # make node for pairblend
    cmds.createNode('plusMinusAverage', n="yneck_pairBlend")
    cmds.createNode('plusMinusAverage', n="yneck_pairBlend_ambient_speech")
    # make node to store animation curves
    cmds.createNode('animCurveTL', n='yneck_pairBlend_gaze')
    cmds.createNode('animCurveTL', n='yneck_pairBlend_speech')
    cmds.createNode('animCurveTL', n='yneck_pairBlend_blend_speech1') # 1 is gaze, 0 is speech
    # this is for ambient and jali neck
    cmds.createNode('animCurveTL', n='yneck_pairBlend_blend_ambient_speech') # 1 is gaze, 0 is speech
    cmds.createNode('animCurveTL', n='yneck_pairBlend_ambient')
    # make them connection
    # cmds.connectAttr('yneck_pairBlend_blend_speech1.output', 'yneck_pairBlend.weight')
    cmds.connectAttr('yneck_pairBlend_speech.output', 'yneck_pairBlend_ambient_speech.input1D[0]')
    cmds.connectAttr('yneck_pairBlend_ambient.output', 'yneck_pairBlend_ambient_speech.input1D[1]')
    cmds.connectAttr('yneck_pairBlend_gaze.output', 'yneck_pairBlend.input1D[0]')    
    cmds.connectAttr('yneck_pairBlend_ambient_speech.output1D', 'yneck_pairBlend.input1D[1]')    
    cmds.connectAttr('yneck_pairBlend.output1D', 'jNeck_ctl.jNeck_yRotate')
    
    try:
        cmds.delete("zneck_pairBlend_ambient_speech")
    except:
        pass
    # the blend node to blend Jali neck and ambient neck (for z axis)
    try:
        cmds.delete("zneck_pairBlend_ambient")
    except:
        pass
    try:
        cmds.delete("zneck_pairBlend_speech")
    except:
        pass
    cmds.createNode('plusMinusAverage', n="zneck_pairBlend_ambient_speech")
    cmds.createNode('animCurveTL', n='zneck_pairBlend_ambient')
    cmds.createNode('animCurveTL', n='zneck_pairBlend_speech')
    cmds.connectAttr('zneck_pairBlend_ambient.output', 'zneck_pairBlend_ambient_speech.input1D[0]')    
    cmds.connectAttr('zneck_pairBlend_speech.output', 'zneck_pairBlend_ambient_speech.input1D[1]')    
    cmds.connectAttr('zneck_pairBlend_ambient_speech.output1D', 'jNeck_ctl.jNeck_zRotate')
    
    return
def load_gaze(filename, rig, tobii = False):
    # create_neck_blend_node(rig)
    create_neck_addition_node(rig)
    # filename = "C:/Users/evan1/Documents/Gaze_project/data/look_at_points/prototype2p2.pkl"
    # filename = "C:/Users/evansamaa/Desktop/Gaze_project/data/look_at_points/prototype2p2.pkl"
    fps = mel.eval('float $fps = `currentTimeUnitToFPS`')
    try:
        if rig == "jali":
            cmds.cutKey("jNeck_ctl.jNeck_yRotate")
            cmds.cutKey("jNeck_ctl.jNeck_xRotate")
            cmds.cutKey("jNeck_ctl.jNeck_zRotate")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateX")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateY")
            cmds.cutKey("eyeStare_world.eyeStare_world_translateZ")
            cmds.cutKey("CNT_BOTH_EYES.LeftRight_eyes")
            cmds.cutKey("CNT_BOTH_EYES.DownUp_eyes")
        elif rig == "waitress":
            cmds.cutKey("neck_1_ctrl.rotateY")
            cmds.cutKey("neck_1_ctrl.rotateZ")
            cmds.cutKey("neck_1_ctrl.rotateX")
            cmds.cutKey("Eye_aim_ctrls.translateX")
            cmds.cutKey("Eye_aim_ctrls.translateY")
            cmds.cutKey("Eye_aim_ctrls.translateZ")
            cmds.cutKey("R_eye_aim_ctrl.translateX")
            cmds.cutKey("R_eye_aim_ctrl.translateY")
            cmds.cutKey("neck_pairBlend_speechL_eye_aim_ctrl.translateX")
            cmds.cutKey("L_eye_aim_ctrl.translateY")
    except:
        pass
    if tobii:
        return load_tobii(filename, rig)
    motion_components = pkl.load(open(filename))
    ek = motion_components["eye_frames"]
    hk = motion_components["head_frames"]
    ms = motion_components["micro_saccade"]
    neck = motion_components["other_neck"]
    envelope = motion_components["envelope"]
    try:
        ambient_neck = motion_components["ambient_neck"]
    except:
        ambient_neck = []
    
    
    
    cmds.setAttr("lookMaster.headWorldBlend", 10)
    # micro-saccade
    for i in range(0, len(ms)):
        interval = ms[i]
        for j in range(0, len(interval)):
            if rig == "jali":
                cmds.setKeyframe("CNT_BOTH_EYES.LeftRight_eyes", t=interval[j][0] * fps, v=interval[j][1])
                cmds.setKeyframe("CNT_BOTH_EYES.DownUp_eyes", t=interval[j][0] * fps, v=-interval[j][2])
            elif rig == "waitress":
                cmds.setKeyframe("R_eye_aim_ctrl.translateX", t=interval[j][0] * fps, v=interval[j][1])
                cmds.setKeyframe("R_eye_aim_ctrl.translateY", t=interval[j][0] * fps, v=-interval[j][2])
                cmds.setKeyframe("L_eye_aim_ctrl.translateX", t=interval[j][0] * fps, v=interval[j][1])
                cmds.setKeyframe("L_eye_aim_ctrl.translateY", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(hk)):
        interval = hk[i]
        for j in range(0, len(interval)):
            # x direction is vertical (elevation)
            cmds.setKeyframe("xneck_pairBlend_gaze", v=-interval[j][2], t=interval[j][0] * fps)
            # y direction is horizontal (azimuth) 
            cmds.setKeyframe("yneck_pairBlend_gaze", v=interval[j][1], t=interval[j][0] * fps)
            # cmds.setKeyframe("jNeck_ctl.jNeck_xRotate", t=interval[j][0] * fps, v=-interval[j][2])
    
    for i in range(0, len(ek)):
        interval = ek[i]
        for j in range(0, len(interval)):
            if rig == "jali":
                cmds.setKeyframe("eyeStare_world.eyeStare_world_translateX", time=interval[j][0] * fps, value=interval[j][1])
                cmds.setKeyframe("eyeStare_world.eyeStare_world_translateY", time=interval[j][0] * fps, value=interval[j][2])
                cmds.setKeyframe("eyeStare_world.eyeStare_world_translateZ", time=interval[j][0] * fps, value=interval[j][3])
            elif rig == "waitress":
                cmds.setKeyframe("Eye_aim_ctrls.translateX", time=interval[j][0] * fps, value=interval[j][1])
                cmds.setKeyframe("Eye_aim_ctrls.translateY", time=interval[j][0] * fps, value=interval[j][2])
                cmds.setKeyframe("Eye_aim_ctrls.translateZ", time=interval[j][0] * fps, value=interval[j][3])
    dims = ["x", "y", "z"]
    cmds.setKeyframe("xneck_pairBlend_blend_speech1", v=0.5, t=0)
    cmds.setKeyframe("yneck_pairBlend_blend_speech1", v=0.5, t=0)
    if len(neck) > 0:
        for k in range(0, 3):
            t = neck[k*2]
            v = neck[k*2+1]
            for i in range(len(t)):
                cmds.setKeyframe("{}neck_pairBlend_speech".format(dims[k]), v=-(v[i]),
                t=t[i] * fps)
    if len(ambient_neck) > 0:
        for k in range(0, 3):
            t = ambient_neck[k*2]
            v = ambient_neck[k*2+1]
            for i in range(len(t)):
                if dims[k] == "x" or dims[k] == "y" or dims[k] == "z":
                    cmds.setKeyframe("{}neck_pairBlend_ambient".format(dims[k]), v=-(v[i]),
                    t=t[i] * fps)
        
# load_video_annotation('/Users/evanpan/Documents/GitHub/Gaze_project/data/look_at_points/video_annotation.json', "all_head")
# load_gaze("C:/Users/evansamaa/Desktop/Gaze_project/data/prototype2p2.pkl", "jali")
# load_gaze("C:/Users/evansamaa/Desktop/Gaze_project/data/tobii_data/shakira/tobii_rotation.pkl", "jali", True)

# load_gaze("F:/MASC/JALI_gaze/Animations/green_book_letter/outputs/raw_clip_neural_0.pkl", "jali")
load_gaze("C:/Users/evan1/OneDrive/Documents/GitHub/EyeIK/Juggling/output\\task_alternating_column_output_Proposed_graph_smooth_pursuit_with_saccade_consider_start_and_end_pos.pkl", "jali")
# load_gaze("F:/MASC/JALI_gaze/animations/eval_royal_with_cheese/non_conversational_output/pulp_fiction_1.pkl", "jali")
