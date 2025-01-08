# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
import math

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.
        self.integralPsiError = 0
        self.previousPsiError = 0
        self.previousXdotError = 0
        self.previousDistError = 0

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        
        look_ahead = 50
        (distError, minIndex) = closestNode(X, Y, trajectory)
        nextPoint = trajectory[clamp(minIndex + look_ahead, 0, len(trajectory)-1)]
        
        distErrorDot = (distError - self.previousDistError) / delT
        
        psiError = wrapToPi(math.atan2(nextPoint[1] - Y, nextPoint[0] - X) - psi)
        psiErrorDot = (psiError - self.previousPsiError) / delT
        
        x = np.array([[distError], \
                      [distErrorDot], \
                      [psiError], \
                      [psiErrorDot]])
                      
        self.previousDistError = distError
        self.previousPsiError = psiError
        
        A = np.array([[0, 1, 0, 0], \
                      [0, -4*Ca/(m*xdot), 4*Ca/m, -(2*Ca*(lf-lr))/(m*xdot)], \
                      [0, 0, 0, 1], \
                      [0, -(2*Ca*(lf-lr))/(Iz*xdot), (2*Ca*(lf-lr))/Iz, -(2*Ca*(lf**2 + lr**2))/(Iz*xdot)]])
                      
        B = np.array([[0], \
                      [2*Ca/m], \
                      [0], \
                      [2*Ca*lf/Iz]])
                      
        desiredPoles = [-0.2+0.05j, -0.2-0.05j, -50, -49]
               
        K = signal.place_poles(A, B, desiredPoles, method='YT').gain_matrix
        
        delta = clamp(np.matmul(K, x)[0][0], -math.pi/6, math.pi/6)

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        
        # PID Gains
        kp = 100
        ki = 10
        kd = 20
        
        # Reference value for PID to tune to
        desiredVelocity = 5
        
        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError
        
        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
