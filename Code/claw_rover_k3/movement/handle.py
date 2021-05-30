import claw_rover_k3.settings.camera_positions as cm
import claw_rover_k3.kinematics.inverse_kinematics as ik
import numpy as np
import coppelia_sim.sim as sim
import time
import cv2

stackobjects = ['Cuboid2', 'Cuboid2','Cuboid1', 'Cuboid1', 'Cuboid0', 'Cuboid0', 'Cuboid', 'Cuboid']


def detect_object(crk3, clientID, leftright):
    retCode, resolution, image=sim.simxGetVisionSensorImage(clientID,crk3.picam2,0,sim.simx_opmode_oneshot_wait)
    img=np.array(image,dtype=np.uint8)
    img.resize([resolution[1],resolution[0],3])
    if leftright == 0:
        img = np.flip(img, axis = 0)
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    '''img = img[range(128,384),:,:]'''
    ret, thresh = cv2.threshold(img_GRAY,150,255,cv2.THRESH_BINARY)
    detector=cv2.SimpleBlobDetector_create()
    keypoints=detector.detect(thresh)
    return keypoints

def move_arm(crk3, theta1, theta2, theta3, clientID):
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint1, theta1, sim.simx_opmode_oneshot) 
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint2, theta2, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint3, theta3, sim.simx_opmode_oneshot)
    time.sleep(2)

def move_claw(crk3, cl, clientID):
    res,retInts,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(clientID,"ROBOTIQ_85", sim.sim_scripttype_childscript,"gripper",[cl],[],[],"", sim.simx_opmode_blocking)
    return res


def take_trash(crk3, clientID, q):
    move_arm(crk3, q[0], q[1], 0.5, clientID) 
    move_arm(crk3, q[0], q[1], q[2], clientID)
    move_claw(crk3, 1, clientID)
    retCode,cuboid=sim.simxGetObjectHandle(clientID,stackobjects.pop(),sim.simx_opmode_blocking)
    retCode,clr=sim.simxGetObjectHandle(clientID,'ROBOTIQ_85_prismatic2',sim.simx_opmode_blocking)
    sim.simxSetObjectParent(clientID,cuboid,clr, True, sim.simx_opmode_oneshot)
    sim.simxSetModelProperty(clientID,cuboid,sim.sim_modelproperty_not_dynamic,True)
    time.sleep(2)
    
def throw_trash(crk3, clientID, rt):
    q = [0,1.7,2.1]
    '''posició escombraries'''
    move_arm(crk3, rt, q[1], q[2], clientID) 
    move_arm(crk3, q[0], q[1], q[2], clientID)
    retCode,cuboid=sim.simxGetObjectHandle(clientID,stackobjects.pop(),sim.simx_opmode_blocking)
    retCode,parent=sim.simxGetObjectHandle(clientID,'Parent',sim.simx_opmode_blocking)
    sim.simxSetObjectParent(clientID,cuboid,parent, True, sim.simx_opmode_oneshot)
    sim.simxSetModelProperty(clientID,cuboid,sim.sim_boolparam_dynamics_handling_enabled,True)
    move_claw(crk3, 0, clientID) 
    time.sleep(2)

def handle_object(crk3, clientID, leftright):
    cm.handling_position(crk3, clientID, leftright)
    objects = detect_object(crk3, clientID, leftright)
    for i in objects:
        y,x,z = (0.3-i.pt[0]*0.3/256), (0.3-i.pt[1]*0.3/256), 0.03
        if leftright == 0:
            x = x - 0.15 
            y = y + 0.09 - 0.01
        else:
            x = x - 0.15
            y = -y - 0.09 + 0.01
        
        q = ik.inverse_kinematics(crk3, x, y, z)
        take_trash(crk3, clientID, q) 
        throw_trash(crk3, clientID, q[0]) 
        cm.handling_position(crk3, clientID, leftright)
