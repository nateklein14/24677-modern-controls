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
        self.prev_e_lat = 0
    
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        prev_e_lat = self.prev_e_lat

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        dist, minIndex = closestNode(X, Y, trajectory)
        
        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        """
        k_p_lat = 0.25
        k_i_lat = 0
        k_d_lat = 0.05
        nextPoint = trajectory[minIndex]
        e_lat = wrapToPi(math.atan2(nextPoint[1] - Y, nextPoint[0] - X) - psi)
        de_lat = (e_lat - prev_e_lat) / delT
        delta = k_p_lat * e_lat + k_d_lat * de_lat
        delta = clamp(delta, -math.pi/6, math.pi/6)
        self.prev_e_lat = e_lat

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        k_p_long = 3100
        k_i_long = 0
        k_d_long = 0
        e_long = dist
        F = 5500 - k_p_long * e_long
        F = clamp(F, 0, 15736)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
