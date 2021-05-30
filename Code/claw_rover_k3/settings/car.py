import coppelia_sim.sim as sim

class car:
    def __init__(self, d1, d2, d3, range11, range12, range21, range22, range31, range32, joint1, joint2, joint3, claw, picam, picam2, wheel_radius, wheelbase, jointw0, jointw2, speed):
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.range11 = range11
        self.range12 = range12
        self.range21 = range21
        self.range22 = range22
        self.range31 = range31
        self.range32 = range32
        self.joint1 = joint1
        self.joint2 = joint2
        self.joint3 = joint3
        self.claw = claw
        self.picam = picam
        self.picam2 = picam2
        self.wheel_radius = wheel_radius
        self.wheelbase = wheelbase
        self.jointw0 = jointw0
        self.jointw2 = jointw2
        self.speed = speed
    
