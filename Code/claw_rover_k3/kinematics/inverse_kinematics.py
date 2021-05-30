import claw_rover_k3.settings.car as c
import numpy as np
from sympy import *
from sympy.physics.mechanics import dynamicsymbols

def inverse_kinematics(crk3, x, y, z):
    q = []
    
    b = crk3.d2
    ab=crk3.d3
    H = crk3.d1
    q.append(np.arctan2(y,x))
    
    xprima=float(sqrt(pow(x,2)+pow(y,2)))
    
    B = xprima
    A = z - H
    
    Hip = float(sqrt(pow(A,2)+pow(B,2)))
    
    alfa=np.arctan2(A,B)
    
    beta=np.arccos((pow(b,2)-pow(ab,2)+pow(Hip,2))/(2*b*Hip))
    
    q.append(alfa+beta)
    
    gamma = np.arccos((pow(b,2)+pow(ab,2)-pow(Hip,2))/(2*b*ab))
    
    q.append(gamma-np.pi)
    
    return q


def inverse_kinematics_nsolve(crk3, x, y, z):
    theta1, theta2, theta3, l1, l2, l3, theta, alpha, a, d = dynamicsymbols('theta1 theta2 theta3 l1 l2 l3 theta alpha a d')
    eq1 = (crk3.d2 * cos(theta2) + crk3.d3 * cos(theta2 + theta3))*cos(theta1) - x
    eq2 = (crk3.d2 * cos(theta2) + crk3.d3 * cos(theta2 + theta3))*sin(theta1) - y
    eq3 = crk3.d1+crk3.d2*sin(theta2)+crk3.d3*sin(theta3+theta2) - z
    q=nsolve((eq1,eq2,eq3),(theta1,theta2,theta3),(1,2,1),verify=False)
    
    if crk3.range11 > q[0] or crk3.range12 < q[0]:
        raise ValueError('Coordenades fora del rang del robot.')
    if crk3.range21 > q[1] or crk3.range22 < q[1]:
        raise ValueError('Coordenades fora del rang del robot')
    if crk3.range31 > q[2] or crk3.range32 < q[2]:
        raise ValueError('Coordenades fora del rang del robot')    
        
    return q
