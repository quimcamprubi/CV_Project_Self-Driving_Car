import coppelia_sim.sim as sim
import numpy as np
import claw_rover_k3.settings.car as c

def connect(port):
# Establece la conexión a VREP
# port debe coincidir con el puerto de conexión en VREP
# retorna el número de cliente o -1 si no puede establecer conexión
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5) # Conectarse
    if clientID != 0: print("no se pudo conectar")
    return clientID

def set_simulation(port):
    clientID = connect(port)
    retCode,picam=sim.simxGetObjectHandle(clientID,'Vision_sensor1_1',sim.simx_opmode_blocking)
    retCode,picam2=sim.simxGetObjectHandle(clientID,'Vision_sensor1_2',sim.simx_opmode_blocking)
    retCode,joint1=sim.simxGetObjectHandle(clientID,'Joint1',sim.simx_opmode_blocking)
    retCode,joint2=sim.simxGetObjectHandle(clientID,'Joint2',sim.simx_opmode_blocking)
    retCode,joint3=sim.simxGetObjectHandle(clientID,'Joint3',sim.simx_opmode_blocking)
    retCode,claw=sim.simxGetObjectHandle(clientID,'ROBOTIQ_85',sim.simx_opmode_blocking)
    retCode,jointw0=sim.simxGetObjectHandle(clientID,'Jointw0',sim.simx_opmode_blocking)
    retCode,jointw2=sim.simxGetObjectHandle(clientID,'Jointw2',sim.simx_opmode_blocking)

    d1 = 0.13
    d2 = 0.20
    d3 = 0.20
    range11 = -2*np.pi
    range12 = 2*np.pi
    range21 = -2*np.pi
    range22 = 2*np.pi
    range31 = -2*np.pi
    range32 = 2*np.pi
    wheel_radius = 0.36
    wheelbase = 2.05
    speed = 15
    
    return clientID, c.car(d1, d2, d3, range11, range12, range21, range22, range31, range32, joint1, joint2, joint3, claw, picam, picam2, wheel_radius, wheelbase, jointw0, jointw2, speed)
