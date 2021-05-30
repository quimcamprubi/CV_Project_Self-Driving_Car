import time
import cv2
import coppelia_sim.sim as sim
import sympy as sp
from PIL import Image
import scipy.misc
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import camera_positions as cp
import computer_vision as cv
import math
import car as c
import array

def SetSpeed(clientID, crk3, v_r, v_l):
    sim.simxSetJointTargetVelocity(clientID, crk3.jointw2, -v_r, sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID, crk3.jointw0, -v_l, sim.simx_opmode_streaming)


def UnicycleModel(crk3, steering_angle):
    # Now that we have the steering angle, we can compute the speed for each servo.
    # We use the Unicycle Model to compute the speed of each servo.
    # https://www.youtube.com/watch?v=aE7RQNhwnPQ
    # https://www.youtube.com/watch?v=Lgy92yXiyqQ
    v_r = (2 * crk3.speed + steering_angle * crk3.wheelbase) / 2 * crk3.wheel_radius
    v_l = (2 * crk3.speed - steering_angle * crk3.wheelbase) / 2 * crk3.wheel_radius

    return v_r, v_l


def DriveAdaptative(crk3, clientID, fps, p_gain, d_gain, i_gain):
    stop = False
    cp.driving_position(crk3, clientID)
    # Previous offset and accumulated offset are used to calculate steering angle with a PID Controller
    prev_offset = 0
    integral = 0
    while not stop:
        # Start the clock
        # Read and process frame
        img, offset, detected_trash = cv.AnalyzeFrameAdaptative(crk3, clientID, prev_offset)
        height,width,_ = img.shape
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0,0,255)
        cv2.putText(img, 'Vehicle offset: {:.4f}'.format(offset), (round(width/5), height-10), font, fontScale, fontColor, 2)
        cv2.imshow('Lane detection', img)
        
        off_delta = prev_offset - offset
        if abs(off_delta) > 30:
            offset = prev_offset
        else:
            # Compute the steering angle using a PID Controller
            # P = Proportional --> offset. Needed to correct trajectory to reach ideal line.
            # I = Integral --> integral. Integral gain will be necessary when avoiding obstacles.
            # D = Differential --> diff. Needed to make driving smoother.
            diff = offset - prev_offset
            integral += offset
            steering_angle = p_gain * offset + d_gain * diff + i_gain * integral # PID Controller
            prev_offset = offset
            # Compute the speed for each servo using the Unicycle Model.
        
            # Adaptative velocity            
            if detected_trash != -1:
                stop = True
            else:
                oftmp = abs(offset)
                if (oftmp < 0.5):
                    crk3.speed = (30+crk3.speed)/2
                elif (oftmp < 1):
                    crk3.speed = (25+crk3.speed)/2
                elif (oftmp < 3):
                    crk3.speed = (20+crk3.speed)/2
                elif (oftmp < 5):
                    crk3.speed = (15+crk3.speed)/2
                elif (oftmp < 7):
                    crk3.speed = (12+crk3.speed)/2
                elif (oftmp < 15):
                    crk3.speed = (10+crk3.speed)/2
                elif (oftmp < 25):
                    crk3.speed = (7+crk3.speed)/2
                else:
                    crk3.speed = 5
                
            if not stop:
                v_r, v_l = UnicycleModel(crk3, steering_angle)
                if v_r == -1 and v_l == -1:
                    raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
                SetSpeed(clientID, crk3, v_r, v_l)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
         
    cv2.destroyAllWindows()
    return detected_trash


