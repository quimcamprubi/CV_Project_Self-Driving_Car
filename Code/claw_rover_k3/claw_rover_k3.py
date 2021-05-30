import time
import coppelia_sim.sim as sim
import claw_rover_k3.settings.car as c
import claw_rover_k3.settings.set_simulation as ss
import claw_rover_k3.settings.camera_positions as cp
import claw_rover_k3.movement.drive as dr
import claw_rover_k3.movement.handle as ha



def run_Claw_Rover_K3():
    clientID, crk3 = ss.set_simulation(19999)
    leftright = 0
    while(True):
        #p_gain = 0.045
        p_gain = 0.07
        d_gain = 0.006
        i_gain = 0.000
        #leftright = dr.DriveAdaptative(crk3, clientID, 20, p_gain, d_gain, i_gain)
        leftright = dr.DriveReal(crk3, clientID, 20, p_gain, d_gain, i_gain)
        if leftright == 1:
            ha.handle_object(crk3, clientID, 1)
            ha.handle_object(crk3, clientID, 0)
        else:
            ha.handle_object(crk3, clientID, 0)
            ha.handle_object(crk3, clientID, 1)

        
