# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:09:40 2018

@author: Patrick Rüdiger

Student project/thesis: Verfahrensvergleich zur Trajektorienplanung für dynamische Systeme (comparison of trajectory planning methods for dynamical systems)

The ocp class implements a general representation of an optimal control problem (ocp) and further includes example problems

The method indirect_method() determines ode and input parametrization for indirect methods based on necessary optimality conditions

"""

# Some examples taken from https://pytrajectory.readthedocs.io/en/master/guide/examples/index.html
# Kunze, Andreas, Knoll, Carsten, & Schnabel, Oliver. (2017, February 8). PyTrajectory ‒ Python library for trajectory generation for nonlinear control systems (Version v1.3.0). Zenodo. http://doi.org/10.5281/zenodo.276212

# Some examples based on http://nbviewer.jupyter.org/github/cknoll/beispiele/tree/master/

import sympy as sp
import numpy as np
from joblib import load


# lists of problems by dimensions
dim21_problems = ['double_int', 'pend']
dim41_problems = ['pend_cart_pl', 'pend_cart', 'ua_manipulator_pl', 'acrobot_pl']
dim61_problems = ['dual_pend_cart_pl', 'dual_pend_cart', 'double_pend_cart_pl']
dim62_problems = ['vtol']
dim81_problems = ['triple_pend_cart_pl']

# problem dimensions
dimensions = {}
dimensions.update({problem: [2, 1] for problem in dim21_problems})
dimensions.update({problem: [4, 1] for problem in dim41_problems})
dimensions.update({problem: [6, 1] for problem in dim61_problems})
dimensions.update({problem: [6, 2] for problem in dim62_problems})
dimensions.update({problem: [8, 1] for problem in dim81_problems})


pi = 3.1415926535897932384626433832795028841971693993


class ocp(object):


    def __init__(self, name, has_objective=True, c=None):
        self.x_dim = None   # state dimension
        self.u_dim = None   # input dimension
        self.x = None       # symbolic variable for state
        self.u = None       # symbolic variable for input
        self.x_dot = None   # symbolic expression of state space model
        self.x_0 = None     # initial state
        self.x_f = None     # final state
        self.u_0 = None     # initial input
        self.u_f = None     # final input
        self.t_0 = None     # initial time
        self.t_f = None     # final time
        self.x_min = None   # lower bound of state (state constraint)
        self.x_max = None   # upper bound of state (state constraint)
        self.u_min = None   # lower bound of input (input constraint)
        self.u_max = None   # upper bound of input (input constraint)
        self.L = None       # integral cost
        self.Q = None       # weight matrix for state in integral cost
        self.R = None       # weight matrix for input in integral cost
        self.x_dict = None  # describes state components
        self.u_dict = None  # describes input components
        self.H = None       # Hamiltonian (needed for necessary optimality conditions)
        self.y_dot = None   # ode for co-state based on necessary optimality conditions
        self.Hu = None      # Jacobian of Hamiltonian w.r.t. u
        self.u_noc = None   # input parametrization based on necessary optimality conditions
        self.z_dot = None   # ode based on necessary optimality conditions

        self.has_objective = has_objective  # specifies whether ocp has a nonzero cost functional
        self.c = c                          # optional constant to modify problem parameters
        self.name = name                    # ocp name
        if self.name not in [*dim21_problems, *dim41_problems, *dim61_problems, *dim62_problems, *dim81_problems]: print('Error: Not Implemented')
        assert self.name in [*dim21_problems, *dim41_problems, *dim61_problems, *dim62_problems, *dim81_problems]
        self.x_dim, self.u_dim = dimensions[self.name]
        self.x = sp.Matrix(sp.symbols('x[:'+str(self.x_dim)+']'))
        self.y = sp.Matrix(sp.symbols('y[:'+str(self.x_dim)+']'))
        self.u = sp.Matrix(sp.symbols('u[:'+str(self.u_dim)+']'))

        if self.name in dim21_problems:
            x1, x2 = self.x
            u1 = self.u[0]

        elif self.name in dim41_problems:
            x1, x2, x3, x4 = self.x
            u1 = self.u[0]

        elif self.name in dim61_problems:
            x1, x2, x3, x4, x5, x6 = self.x
            u1 = self.u[0]

        elif self.name in dim62_problems:
            x1, x2, x3, x4, x5, x6 = self.x
            u1, u2 = self.u

        elif self.name in dim81_problems:
            x1, x2, x3, x4, x5, x6, x7, x8 = self.x
            u1 = self.u[0]

        # example problems
        if self.name == 'double_int':
            # double integrator
            self.x_dict = {'1': 's', '2': 'v'}
            self.u_dict = {'1': 'a'}

            self.x_dot = sp.Matrix([x2,
                                    u1])

            self.x_0 = np.array([0, 0])
            self.x_f = np.array([1, 0])
            self.t_0 = 0
            self.t_f = 2

            self.x_min[1] = 0
            self.x_max[1] = 0.65
            self.u_min = -2*np.ones(self.u_dim)
            self.u_max = 2*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name == 'pend':
            # simple pendulum
            l = 0.5     # length of the pendulum
            g = 9.81    # gravitational acceleration

            self.x_dict = {'1': 'theta', '2': 'omega'}
            self.u_dict = {'1': 'alpha'}

            self.x_dot = sp.Matrix([x2,
                                    u1 + g/l*sp.sin(x1)])

            self.x_0 = np.array([np.pi, 0])
            self.x_f = np.array([0, 0])
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name in ['pend_cart_pl', 'pend_cart']:
            # (partially linearized) pendulum on cart
            s1 = 0.25           # center of mass of the pendulum
            m1 = 0.1            # mass of the pendulum
            m0 = 1.0            # mass of the cart
            g = 9.81            # gravitational acceleration

            self.x_dict = {'1': 's', '2': 'v', '3': 'theta', '4': 'omega'}
            if 'pl' in self.name: self.u_dict = {'1': 'a'}
            else: self.u_dict = {'1': 'F'}

            self.x_dot = eval(load('examples/'+self.name+'.str')['x_dot_str'])

            self.x_0 = np.array([0, 0, np.pi, 0])
#            self.x_0 = np.array([0, 0, 45/180*np.pi, 0])
            self.x_f = np.array([0, 0, 0, 0])
            self.t_0 = 0
            self.t_f = 1
#            self.t_f = self.c

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)
#            self.u_min = -100*np.ones(self.u_dim)
#            self.u_max = 100*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name in ['dual_pend_cart_pl', 'dual_pend_cart']:
            # (partially linearized) dual pendulum on cart
            # center of mass, and mass of the pendulums
#            s1 = 0.7
            s1 = self.c
#            m1 = 0.7
            s2 = 0.5
#            m2 = 0.5
            m0 = 1.0    # mass of the cart
            g = 9.81    # gravitational acceleration

            self.x_dict = {'1': 's', '2': 'v', '3': 'theta1', '4': 'omega1', '5': 'theta2', '6': 'omega2'}
            if 'pl' in self.name: self.u_dict = {'1': 'a'}
            else: self.u_dict = {'1': 'F'}

            self.x_dot = eval(load('examples/'+self.name+'.str')['x_dot_str'])

            self.x_0 = np.array([0, 0, np.pi, 0, np.pi, 0])
            self.x_f = np.array([0, 0, 0, 0, 0, 0])
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
#            self.u_min = -np.inf*np.ones(self.u_dim)
#            self.u_max = np.inf*np.ones(self.u_dim)
            self.u_min = -100*np.ones(self.u_dim)
            self.u_max = 100*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name == 'vtol':
            # PyTrajectory: ex3 (vertical take-off and landing aircraft)
            # coordinates for the points in which the engines engage [m]
            l = 1.0
            h = 0.1

            g = 9.81    # graviational acceleration [m/s^2]
            M = 50.0    # mass of the aircraft [kg]
            J = 25.0    # moment of inertia about M [kg*m^2]

            alpha = 5/360.0*2*pi # deflection of the engines

            sa = sp.sin(alpha)
            ca = sp.cos(alpha)

            s = sp.sin(x5)
            c = sp.cos(x5)

            self.x_dict = {'1': 'z1', '2': 'v1', '3': 'z2', '4': 'v2', '5': 'theta', '6': 'omega'}
            self.u_dict = {'1': 'F1', '2': 'F2'}

            self.x_dot = sp.Matrix([x2,
                                    -s/M*(u1+u2) + c/M*(u1-u2)*sa,
                                    x4,
                                    -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
                                    x6,
                                    1/J*(u1-u2)*(l*ca+h*sa)])

            self.x_0 = np.array([0, 0, 0, 0, 0, 0])
            self.x_f = np.array([10, 0, 5, 0, 0, 0])
#            self.u_0 = np.array([0.5*9.81*50.0/(sp.cos(5/360.0*2*np.pi)), 0.5*9.81*50.0/(sp.cos(5/360.0*2*np.pi))])
#            self.u_f = np.array([0.5*9.81*50.0/(sp.cos(5/360.0*2*np.pi)), 0.5*9.81*50.0/(sp.cos(5/360.0*2*np.pi))])
            self.t_0 = 0
            self.t_f = 3

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name == 'ua_manipulator_pl':
            # PyTrajectory: ex4 (underactuated manipulator, partially linearized)
            e = 0.9     # inertia coupling
            s = sp.sin(x3)
            c = sp.cos(x3)

            self.x_dict = {'1': 'theta1', '2': 'omega1', '3': 'theta2', '4': 'omega2'}
            self.u_dict = {'1': 'alpha1'}

            self.x_dot = sp.Matrix([x2,
                                    u1,
                                    x4,
                                    -e*x2**2*s-(1+e*c)*u1])

            self.x_0 = np.array([0, 0, 0.4*np.pi, 0])
            self.x_f = np.array([0.2*np.pi, 0, 0.2*np.pi, 0])
            self.t_0 = 0
            self.t_f = 1.8

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name == 'acrobot_pl':
            # PyTrajectory: ex5 (Acrobot, partially linearized)
            m = 1.0             # masses of the rods [m1 = m2 = m]
            l = 0.5             # lengths of the rods [l1 = l2 = l]

            I = 1/3.0*m*l**2    # moments of inertia [I1 = I2 = I]
            g = 9.81            # gravitational acceleration

            lc = l/2.0

            d11 = m*lc**2+m*(l**2+lc**2+2*l*lc*sp.cos(x1))+2*I
            h1 = -m*l*lc*sp.sin(x1)*(x2*(x2+2*x4))
            d12 = m*(lc**2+l*lc*sp.cos(x1))+I
            phi1 = (m*lc+m*l)*g*sp.cos(x3)+m*lc*g*sp.cos(x1+x3)

            self.x_dict = {'1': 'theta2', '2': 'omega2', '3': 'theta1', '4': 'omega1'}
            self.u_dict = {'1': 'alpha2'}

            self.x_dot = sp.Matrix([x2,
                                    u1,
                                    x4,
                                    -1/d11*(h1+phi1+d12*u1)])

            self.x_0 = np.array([0, 0, 1.5*np.pi, 0])
            self.x_f = np.array([0, 0, 0.5*np.pi, 0])
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name == 'double_pend_cart_pl':
            # partially linearized double pendulum on cart
            # length, center of mass, mass, and moment of inertia of the pendulums
            l1 = 0.5
            l2 = 0.5
            s1 = l1/2
            s2 = l2/2
            m1 = 0.1
            m2 = 0.1
            J1 = 4/3*m1*l1**2
            J2 = 4/3*m2*l2**2

            m0 = 1.0     # mass of the cart
            g = 9.81    # gravitational acceleration

            self.x_dict = {'1': 's', '2': 'v', '3': 'theta1', '4': 'omega1', '5': 'theta2', '6': 'omega2'}
            self.u_dict = {'1': 'a'}

            self.x_dot = eval(load('examples/'+self.name+'.str')['x_dot_str'])

            self.x_0 = np.array([0, 0, np.pi, 0, 0, 0])
            self.x_f = np.array([0, 0, 0, 0, 0, 0])
            self.t_0 = 0
            self.t_f = 3

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -100*np.ones(self.u_dim)
            self.u_max = 100*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        elif self.name == 'triple_pend_cart_pl':
            # partially linearized triple pendulum on cart
            # length, center of mass, mass, and moment of inertia of the pendulums
            l1 = 0.5
            l2 = 0.5
            l3 = 0.5
            s1 = l1/2
            s2 = l2/2
            s3 = l3/2
            m1 = 0.1
            m2 = 0.1
            m3 = 0.1
            J1 = 4/3*m1*l1**2
            J2 = 4/3*m2*l2**2
            J3 = 4/3*m3*l3**2

            m0 = 1.0    # mass of the cart
            g = 9.81    # gravitational acceleration

            self.x_dict = {'1': 's', '2': 'v', '3': 'theta1', '4': 'omega1', '5': 'theta2', '6': 'omega2', '7': 'theta3', '8': 'omega3'}
            self.u_dict = {'1': 'a'}

            self.x_dot = eval(load('examples/'+self.name+'.str')['x_dot_str'])

            self.x_0 = np.array([0, 0, np.pi, 0, 0, 0, 0, 0])
            self.x_f = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            self.t_0 = 0
            self.t_f = 4

#            self.x_min = -np.inf*np.ones(self.x_dim)
#            self.x_max = np.inf*np.ones(self.x_dim)
            self.x_min = -np.array([np.inf, np.inf, 2*np.pi, np.inf, np.pi, np.inf, 2*np.pi, np.inf])
            self.x_max = np.array([np.inf, np.inf, 2*np.pi, np.inf, np.pi, np.inf, 2*np.pi, np.inf])
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)
#            self.u_min = -100*np.ones(self.u_dim)
#            self.u_max = 100*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))


        # setup objective (cost functional)
        if self.has_objective:
            self.L = ((self.x.T - np.reshape(self.x_f, (1,self.x_dim)))*self.Q*(self.x - np.reshape(self.x_f, (self.x_dim,1)))
                    + self.u.T*self.R*self.u)

            self.has_objective = (np.any(np.array(self.Q) != 0) or np.any(np.array(self.R) != 0))
        else:
            self.L = [0]


    def indirect_method(self): # determines ode and input parametrization for indirect methods based on necessary optimality conditions
        self.H = self.L + self.y.T*self.x_dot
        self.y_dot = sp.simplify(-self.H.jacobian(self.x))
        self.Hu = sp.simplify(self.H.jacobian(self.u))

        u_noc = sp.solve(self.Hu, self.u)
        self.u_noc = sp.Matrix([val for (key, val) in u_noc.items()])
        self.z_dot = sp.Matrix([*self.x_dot, *self.y_dot]).subs([(self.u[i], self.u_noc[i]) for i in range(self.u_dim)])