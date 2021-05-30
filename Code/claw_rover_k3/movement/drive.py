import time
import cv2
import coppelia_sim.sim as sim
import sympy as sp
from PIL import Image
import scipy.misc
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import claw_rover_k3.settings.camera_positions as cp
import claw_rover_k3.movement.computer_vision as cv
import math
import claw_rover_k3.settings.car as c
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


def GoToTrashAdaptative(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset):
    diff = offset - prev_offset
    integral += offset
    steering_angle = p_gain * offset + d_gain * diff + i_gain * integral
    prev_offset = offset
    v_r, v_l = UnicycleModel(crk3, steering_angle)
    if v_r == -1 and v_l == -1:
        raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
    SetSpeed(clientID, crk3, v_r, v_l)
    detected_trash1 = 0
    detected_trash2 = 0
    while detected_trash1 != -1 or detected_trash2 != -1:
        # Start the clock
        if (detected_trash1 != -1):
        # Read and process frame
            offset, detected_trash1 = cv.AnalyzeFrameAdaptative(crk3, clientID, prev_offset)
        else:
            offset, detected_trash2 = cv.AnalyzeFrameAdaptative(crk3, clientID, prev_offset)
            detected_trash1 = detected_trash2
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
            
            v_r, v_l = UnicycleModel(crk3, steering_angle)
            if v_r == -1 and v_l == -1:
                raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
            SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing
        
    crono = time.time()
    while time.time() - crono < 10:
        offset, detected_trash = cv.AnalyzeFrameAdaptative(crk3, clientID, prev_offset)
        off_delta = prev_offset - offset
        
        if abs(off_delta) > 30:
            offset = prev_offset
        else:            
            diff = offset - prev_offset
            integral += offset
            steering_angle = p_gain * offset + d_gain * diff + i_gain * integral # PID Controller
            prev_offset = offset
            v_r, v_l = UnicycleModel(crk3, steering_angle)
            if v_r == -1 and v_l == -1:
                raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
            SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing   
    SetSpeed(clientID, crk3, 0, 0)
    
def GoToTrashReal(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset):
    diff = offset - prev_offset
    integral += offset
    steering_angle = p_gain * offset + d_gain * diff + i_gain * integral
    prev_offset = offset
    v_r, v_l = UnicycleModel(crk3, steering_angle)
    if v_r == -1 and v_l == -1:
        raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
    SetSpeed(clientID, crk3, v_r, v_l)
    detected_trash1 = 0
    detected_trash2 = 0
    while detected_trash1 != -1 or detected_trash2 != -1:
        # Start the clock
        if (detected_trash1 != -1):
        # Read and process frame
            offset, detected_trash1, speed = cv.AnalyzeFrameReal(crk3, clientID, prev_offset)
        else:
            offset, detected_trash2, speed = cv.AnalyzeFrameReal(crk3, clientID, prev_offset)
            detected_trash1 = detected_trash2
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
            
            v_r, v_l = UnicycleModel(crk3, steering_angle)
            if v_r == -1 and v_l == -1:
                raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
            SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing
        
    crono = time.time()
    while time.time() - crono < 10:
        offset, detected_trash, speed = cv.AnalyzeFrameReal(crk3, clientID, prev_offset)
        off_delta = prev_offset - offset
        
        if abs(off_delta) > 30:
            offset = prev_offset
        else:            
            diff = offset - prev_offset
            integral += offset
            steering_angle = p_gain * offset + d_gain * diff + i_gain * integral # PID Controller
            prev_offset = offset
            v_r, v_l = UnicycleModel(crk3, steering_angle)
            if v_r == -1 and v_l == -1:
                raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
            SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing   
    SetSpeed(clientID, crk3, 0, 0)



def DriveAdaptative(crk3, clientID, fps, p_gain, d_gain, i_gain):
    stop = False
    cp.driving_position(crk3, clientID)
    # Previous offset and accumulated offset are used to calculate steering angle with a PID Controller
    prev_offset = 0
    integral = 0
    while not stop:
        # Start the clock
        # Read and process frame
        offset, detected_trash = cv.AnalyzeFrameAdaptative(crk3, clientID, prev_offset)
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
                antspeed = crk3.speed
                crk3.speed = 5            
                GoToTrashAdaptative(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset)
                crk3.speed = antspeed
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
        # End of frame processing   
    cv2.destroyAllWindows()
    return detected_trash

def DriveReal(crk3, clientID, fps, p_gain, d_gain, i_gain):
    stop = False
    cp.driving_position(crk3, clientID)
    # Previous offset and accumulated offset are used to calculate steering angle with a PID Controller
    prev_offset = 0
    integral = 0
    stop_sign = False
    last1 = crk3.speed
    last2 = crk3.speed
    last3 = crk3.speed
    last4 = crk3.speed
    last5 = crk3.speed
    while not stop:
        # Start the clock
        # Read and process frame
        offset, detected_trash, speed = cv.AnalyzeFrameReal(crk3, clientID, prev_offset)
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
        
            if detected_trash != -1:
                antspeed = crk3.speed
                crk3.speed = 5            
                GoToTrashReal(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset)
                crk3.speed = antspeed
                stop = True
            else:
                last5 = last4
                last4 = last3
                last3 = last2
                last2 = last1
                last1 = speed
                a = 0
                if speed == 0 and stop_sign == False:
                    SetSpeed(clientID, crk3, 0, 0)
                    time.sleep(3)
                    stop_sign = True
                elif speed != -1 and speed != 0:
                    
                    if last2 != -1 and last2 != speed:
                        if last3 != -1 and last3 != speed:
                            if last4 != -1 and last4 != speed:
                                if last5 != -1 and last5 != speed:
                                    last1 = last2
                                    print('Lectura Errònia')
                                    a = 1
                    if (a == 0):
                        stop_sign = False
                        crk3.speed = (speed+crk3.speed*2)/3                        
                elif speed == -1:
                    stop_sign = False
                
            if not stop:
                v_r, v_l = UnicycleModel(crk3, steering_angle)
                if v_r == -1 and v_l == -1:
                    raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
                SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing    
    cv2.destroyAllWindows()
    return detected_trash

#Same code with comments and console messages
'''
def GoToTrash(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset, start):
    diff = offset - prev_offset
    integral += offset
    steering_angle = p_gain * offset + d_gain * diff + i_gain * integral
    prev_offset = offset
    v_r, v_l = UnicycleModel(crk3, steering_angle)
    print("Vr = ", v_r, " | Vl = ", v_l)
    print("")
    if v_r == -1 and v_l == -1:
        raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
    SetSpeed(clientID, crk3, v_r, v_l)
    t_frame = time.time() - start
    print("Frame processed in", t_frame, "seconds.")
    
    detected_trash1 = 0
    detected_trash2 = 0
    while detected_trash1 != -1 or detected_trash2 != -1:
        # Start the clock
        start = time.time()
        if (detected_trash1 != -1):
        # Read and process frame
            img, offset, detected_trash1, speed = cv.AnalyzeFrame(crk3, clientID, prev_offset)
        else:
            img, offset, detected_trash2, speed = cv.AnalyzeFrame(crk3, clientID, prev_offset)
            detected_trash1 = detected_trash2
         
        cv2.imshow('Detected lane', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        off_delta = prev_offset - offset
        
        print("Offset delta = ", off_delta)
        if abs(off_delta) > 30:
            offset = prev_offset
        
        # Else, update trajectory and keep going
        if abs(off_delta) < 30:
            print("Offset = ", offset)
            
            height, width, channels = img.shape
            # Compute the steering angle using a PID Controller
            # P = Proportional --> offset. Needed to correct trajectory to reach ideal line.
            # I = Integral --> integral. Integral gain will be necessary when avoiding obstacles.
            # D = Differential --> diff. Needed to make driving smoother.
            diff = offset - prev_offset
            integral += offset
            steering_angle = p_gain * offset + d_gain * diff + i_gain * integral # PID Controller
            prev_offset = offset
            # Compute the speed for each servo using the Unicycle Model.
            
            v_r, v_l = UnicycleModel(crk3, steering_angle)
            print("Vr = ", v_r, " | Vl = ", v_l)
            print("")
            if v_r == -1 and v_l == -1:
                raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
            SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing
        t_frame = time.time() - start
        print("Frame processed in", t_frame, "seconds.")
        
    crono = time.time()
    while time.time() - crono < 10:
        start = time.time()
        img, offset, detected_trash, speed = cv.AnalyzeFrame(crk3, clientID, prev_offset)

        
        cv2.imshow('Detected lane', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        off_delta = prev_offset - offset
        
        print("Offset delta = ", off_delta)
        if abs(off_delta) > 30:
            offset = prev_offset
        else:
            print("Offset = ", offset)
            
            height, width, channels = img.shape
            diff = offset - prev_offset
            integral += offset
            steering_angle = p_gain * offset + d_gain * diff + i_gain * integral # PID Controller
            prev_offset = offset
            v_r, v_l = UnicycleModel(crk3, steering_angle)
            
            print("Vr = ", v_r, " | Vl = ", v_l)
            print("")
            if v_r == -1 and v_l == -1:
                raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
            SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing
        t_frame = time.time() - start
        print("Frame processed in", t_frame, "seconds.")    
    SetSpeed(clientID, crk3, 0, 0)



def DriveAdaptative(crk3, clientID, fps, p_gain, d_gain, i_gain):
    stop = False
    cp.driving_position(crk3, clientID)
    # Previous offset and accumulated offset are used to calculate steering angle with a PID Controller
    prev_offset = 0
    integral = 0
    stop_sign = False
    last1 = crk3.speed
    last2 = crk3.speed
    last3 = crk3.speed
    last4 = crk3.speed
    last5 = crk3.speed
    while not stop:
        # Start the clock
        start = time.time()
        # Read and process frame
        img, offset, detected_trash, speed = cv.AnalyzeFrame(crk3, clientID, prev_offset)
        
        cv2.imshow('Detected lane', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        off_delta = prev_offset - offset
        print("Offset delta = ", off_delta)
        if abs(off_delta) > 30:
            offset = prev_offset
        else:
            print("Offset = ", offset)
            height, width, channels = img.shape
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
                antspeed = crk3.speed
                crk3.speed = 5            
                GoToTrash(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset, start)
                crk3.speed = antspeed
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
                print("Vr = ", v_r, " | Vl = ", v_l)
                print("")
                if v_r == -1 and v_l == -1:
                    raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
                SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing
        t_frame = time.time() - start
        print("Frame processed in", t_frame, "seconds.")
    
    cv2.destroyAllWindows()
    return detected_trash

def DriveReal(crk3, clientID, fps, p_gain, d_gain, i_gain):
    stop = False
    cp.driving_position(crk3, clientID)
    # Previous offset and accumulated offset are used to calculate steering angle with a PID Controller
    prev_offset = 0
    integral = 0
    stop_sign = False
    last1 = crk3.speed
    last2 = crk3.speed
    last3 = crk3.speed
    last4 = crk3.speed
    last5 = crk3.speed
    while not stop:
        # Start the clock
        start = time.time()
        # Read and process frame
        img, offset, detected_trash, speed = cv.AnalyzeFrame(crk3, clientID, prev_offset)
        
        cv2.imshow('Detected lane', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        off_delta = prev_offset - offset
        print("Offset delta = ", off_delta)
        if abs(off_delta) > 30:
            offset = prev_offset
        else:
            print("Offset = ", offset)
            height, width, channels = img.shape
            # Compute the steering angle using a PID Controller
            # P = Proportional --> offset. Needed to correct trajectory to reach ideal line.
            # I = Integral --> integral. Integral gain will be necessary when avoiding obstacles.
            # D = Differential --> diff. Needed to make driving smoother.
            diff = offset - prev_offset
            integral += offset
            steering_angle = p_gain * offset + d_gain * diff + i_gain * integral # PID Controller
            prev_offset = offset
            # Compute the speed for each servo using the Unicycle Model.
        
            if detected_trash != -1:
                antspeed = crk3.speed
                crk3.speed = 5            
                GoToTrash(clientID, crk3, p_gain, d_gain, i_gain, integral, prev_offset, offset, start)
                crk3.speed = antspeed
                stop = True
            else:
                last5 = last4
                last4 = last3
                last3 = last2
                last2 = last1
                last1 = speed
                a = 0
                if speed == 0 and stop_sign == False:
                    SetSpeed(clientID, crk3, 0, 0)
                    time.sleep(3)
                    stop_sign = True
                elif speed != -1 and speed != 0:
                    
                    if last2 != -1 and last2 != speed:
                        if last3 != -1 and last3 != speed:
                            if last4 != -1 and last4 != speed:
                                if last5 != -1 and last5 != speed:
                                    last1 = last2
                                    print('Lectura Errònia')
                                    a = 1
                    if (a == 0):
                        stop_sign = False
                        crk3.speed = (speed+crk3.speed*2)/3
                    print()  
                    print(crk3.speed)
                    print()
                        
                elif speed == -1:
                    stop_sign = False
                
            if not stop:
                v_r, v_l = UnicycleModel(crk3, steering_angle)
                print("Vr = ", v_r, " | Vl = ", v_l)
                print("")
                if v_r == -1 and v_l == -1:
                    raise ValueError("Error, could not compute the servo speeds because no line was detected. ")
                SetSpeed(clientID, crk3, v_r, v_l)
        # End of frame processing
        t_frame = time.time() - start
        print("Frame processed in", t_frame, "seconds.")
    
    cv2.destroyAllWindows()
    return detected_trash
'''
