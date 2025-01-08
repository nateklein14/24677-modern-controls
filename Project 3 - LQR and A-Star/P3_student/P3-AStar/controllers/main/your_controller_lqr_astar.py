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
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        look_ahead = 100
        (dist, minIndex) = closestNode(X, Y, trajectory)
        closePoint = trajectory[minIndex]
        nextPoint = trajectory[clamp(minIndex + look_ahead, 0, len(trajectory)-1)]
        
        psiDesired = np.arctan2(nextPoint[1]-closePoint[1], nextPoint[0]-closePoint[0])
        
        distError = (Y - nextPoint[1])*np.cos(psiDesired) - (X - nextPoint[0])*np.sin(psiDesired)
        distErrorDot = (distError - self.previousDistError) / delT
        
        psiError = wrapToPi(psi - psiDesired)
        psiErrorDot = psidot
        
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
        
        C = np.identity(4)
        
        D = np.zeros((4,1))
                      
        cont_sys = (A, B, C, D)
        disc_sys = signal.cont2discrete(cont_sys, delT)
        A_d = disc_sys[0]
        B_d = disc_sys[1]
                      
        q = np.array([[200, 0, 0, 0], \
                      [0, 50, 0, 0], \
                      [0, 0, 100, 0], \
                      [0, 0, 0, 100]])
                      
        r = np.array([[3000]])
        
        S = np.matrix(linalg.solve_discrete_are(A_d, B_d, q, r))
        K = -linalg.inv(B_d.T@S@B_d+r)@B_d.T@S@A_d
       
        delta = clamp(float(K@x), -math.pi, math.pi)

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """

        # PID Gains
        kp = 350
        ki = 5
        kd = 0
        
        # Reference value for PID to tune to
        desiredVelocity = 11.7
        
        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError
        
        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
