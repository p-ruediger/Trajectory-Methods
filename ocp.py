# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:09:40 2018

@author: Patrick Rüdiger

"""

# Some examples taken from https://pytrajectory.readthedocs.io/en/master/guide/examples/index.html
# Kunze, Andreas, Knoll, Carsten, & Schnabel, Oliver. (2017, February 8). PyTrajectory ‒ Python library for trajectory generation for nonlinear control systems (Version v1.3.0). Zenodo. http://doi.org/10.5281/zenodo.276212

# Some examples based on http://nbviewer.jupyter.org/github/cknoll/beispiele/tree/master/

import sympy as sp
import numpy as np
from joblib import load


def get_x_dot_matrices(q_dot, M, K):
    # q_dot: specific elements of self.x
    return sp.simplify(sp.Matrix([q_dot, -M.inv()*K]))


# lists of problems by dimensions
dim21_problems = ['double_int', 'simple_pend']
dim41_problems = ['simple_pend_cart_pl', 'simple_pend_cart', 'pend_cart_pl', 'pend_cart', 'ua_manipulator_pl', 'acrobot_pl']
dim61_problems = ['simple_dual_pend_cart_pl', 'simple_dual_pend_cart', 'dual_pend_cart_pl', 'dual_pend_cart', 'double_pend_cart_pl']
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


    def __init__(self, name):
        self.x_dim = None
        self.u_dim = None
        self.x = None
        self.u = None
        self.x_dot = None
        self.x_0 = None
        self.x_f = None
        self.u_0 = None
        self.u_f = None
        self.t_0 = None
        self.t_f = None
        self.x_min = None
        self.x_max = None
        self.u_min = None
        self.u_max = None
        self.L = None
        self.E = None
        self.Q = None
        self.R = None
        self.S = None
        self.x_dict = None
        self.u_dict = None
        self.has_objective = None
        self.H = None
        self.y_dot = None
        self.Hu = None
        self.u_noc = None
        self.z_dot = None

        self.name = name
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


        if self.name is 'double_int':
            # double integrator
            self.x_dict = {'1': 's', '2': 'v'}
            self.u_dict = {'1': 'a'}

            self.x_dot = sp.Matrix([x2,
                                    u1])

            self.x_0 = np.array([0, 0])
            self.x_f = np.array([1, 0])
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.x_min[1] = 0
            self.x_max[1] = 0.65
#            self.u_min = -np.inf*np.ones(self.u_dim)
#            self.u_max = np.inf*np.ones(self.u_dim)
            self.u_min = -2*np.ones(self.u_dim)
            self.u_max = 2*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name is 'simple_pend':
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
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name in ['simple_pend_cart_pl', 'simple_pend_cart', 'pend_cart_pl', 'pend_cart']:
            # (simple) (partially linearized) pendulum on cart
            l1 = 0.5            # length of the pendulum
            s1 = l1/2           # center of mass of the pendulum
            m1 = 0.1            # mass of the pendulum
            J1 = 4/3*m1*l1**2   # moment of inertia of the pendulum
            m0 = 1.0            # mass of the cart
            g = 9.81            # gravitational acceleration

            self.x_dict = {'1': 's', '2': 'v', '3': 'theta', '4': 'omega'}
            self.u_dict = {'1': 'a'}

            self.x_dot = eval(load('examples/'+self.name+'.str')['x_dot_str'])

            self.x_0 = np.array([0, 0, np.pi, 0])
            self.x_f = np.array([0, 0, 0, 0])
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)
#            self.u_min = -70*np.ones(self.u_dim)
#            self.u_max = 70*np.ones(self.u_dim)

            self.Q = sp.diag(1,0,1,0)
            self.R = sp.diag(1)
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name in ['simple_dual_pend_cart_pl', 'simple_dual_pend_cart', 'dual_pend_cart_pl', 'dual_pend_cart']:
            # (simple) (partially linearized) dual pendulum on cart
            # length, center of mass, mass, and moment of inertia of the pendulums
            l1 = 0.7
            s1 = l1/2
            m1 = 0.7
            J1 = 4/3*m1*l1**2
            l2 = 0.5
            s2 = l1/2
            m2 = 0.5
            J2 = 4/3*m1*l1**2

            m0 = 1.0    # mass of the cart
            g = 9.81    # gravitational acceleration

            self.x_dict = {'1': 's', '2': 'v', '3': 'theta1', '4': 'omega1', '5': 'theta2', '6': 'omega2'}
            self.u_dict = {'1': 'a'}

            self.x_dot = eval(load('examples/'+self.name+'.str')['x_dot_str'])

            self.x_0 = np.array([0, 0, np.pi, 0, np.pi, 0])
            self.x_f = np.array([0, 0, 0, 0, 0, 0])
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
#            self.x_min[0] = -2
#            self.x_max[0] = 2
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)
#            self.u_min = -100*np.ones(self.u_dim)
#            self.u_max = 100*np.ones(self.u_dim)

            self.Q = sp.diag(1,0,1,0,1,0)
            self.R = sp.diag(1)
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name is 'vtol':
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

            self.Q = sp.diag(1,0,1,0,1,0)
            self.R = sp.diag(*np.ones(self.u_dim))
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name is 'ua_manipulator_pl':
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
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 1.8

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name is 'acrobot_pl':
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
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 2

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name is 'double_pend_cart_pl':
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

            self.x_0 = np.array([0, 0, np.pi, 0, np.pi, 0])
            self.x_f = np.array([0, 0, 0, 0, 0, 0])
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 3

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
#            self.x_min[0] = -3
#            self.x_max[0] = 3
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)
#            self.u_min = 0*np.ones(self.u_dim)
#            self.u_max = 0*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))
            self.S = sp.diag(*np.ones(self.x_dim))


        elif self.name is 'triple_pend_cart_pl':
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

            self.x_0 = np.array([0, 0, np.pi, 0, np.pi, 0, np.pi, 0])
            self.x_f = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#            self.u_0 = np.zeros(self.u_dim)
#            self.u_f = np.zeros(self.u_dim)
            self.t_0 = 0
            self.t_f = 3.5

            self.x_min = -np.inf*np.ones(self.x_dim)
            self.x_max = np.inf*np.ones(self.x_dim)
#            self.x_min[0] = -3
#            self.x_max[0] = 3
            self.u_min = -np.inf*np.ones(self.u_dim)
            self.u_max = np.inf*np.ones(self.u_dim)

            self.Q = sp.diag(*np.ones(self.x_dim))
            self.R = sp.diag(*np.ones(self.u_dim))
            self.S = sp.diag(*np.ones(self.x_dim))


        self.L = ((self.x.T - np.reshape(self.x_f, (1,self.x_dim)))*self.Q*(self.x - np.reshape(self.x_f, (self.x_dim,1)))
                        + self.u.T*self.R*self.u)
        self.E = (self.x.T - np.reshape(self.x_f, (1,self.x_dim)))*self.S*(self.x - np.reshape(self.x_f, (self.x_dim,1)))

        self.has_objective = (np.any(np.array(self.Q) != 0) or np.any(np.array(self.R) != 0))


    def setup_optbvp(self):
        self.H = self.L + self.y.T*self.x_dot
        self.y_dot = sp.simplify(-self.H.jacobian(self.x))
        self.Hu = sp.simplify(self.H.jacobian(self.u))

        u_noc = sp.solve(self.Hu, self.u)
        self.u_noc = sp.Matrix([val for (key, val) in u_noc.items()])
        self.z_dot = sp.Matrix([*self.x_dot, *self.y_dot]).subs([(self.u[i], self.u_noc[i]) for i in range(self.u_dim)])