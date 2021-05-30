import time
import coppelia_sim.sim as sim
import car as c
import set_simulation as ss
import camera_positions as cp
import drive as dr

def run_Claw_Rover_K3():
    clientID, crk3 = ss.set_simulation(19999)
    leftright = 0
    while(True):
        p_gain = 0.07
        d_gain = 0.006
        i_gain = 0.000
        leftright = dr.DriveAdaptative(crk3, clientID, 20, p_gain, d_gain, i_gain)
        