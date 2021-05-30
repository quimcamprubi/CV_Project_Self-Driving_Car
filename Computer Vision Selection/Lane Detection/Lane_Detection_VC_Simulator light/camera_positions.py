import coppelia_sim.sim as sim
import numpy as np
import time

def driving_position(crk3, clientID):
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint1, 0, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint2, 4*np.pi/6, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint3, -np.pi/3, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.claw, 0, sim.simx_opmode_oneshot)
    time.sleep(3)

def handling_position(crk3, clientID, leftright):
    if leftright == 0:
        retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint1, np.pi/2, sim.simx_opmode_oneshot)    
    else:
        retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint1, -np.pi/2, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint2, np.pi/4, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.joint3, -np.pi/4, sim.simx_opmode_oneshot)
    retCode = sim.simxSetJointTargetPosition(clientID, crk3.claw, 0, sim.simx_opmode_oneshot)
    time.sleep(3)