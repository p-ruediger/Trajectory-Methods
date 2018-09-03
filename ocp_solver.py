# -*- coding: utf-8 -*-
'''
Created on Mon May 14 12:35:31 2018

@author: Patrick Rüdiger

Student project/thesis: Verfahrensvergleich zur Trajektorienplanung für dynamische Systeme (comparison of trajectory planning methods for dynamical systems)

Implementation of direct and indirect shooting and collocation methods for trajectory optimization with CasADi using CVODES and IPOPT

'''

#     Partially based on CasADi examples.
#     https://github.com/casadi/casadi/tree/master/docs/examples/python
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl, K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

import numpy as np
import sympy as sp
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ocp import ocp


# needed to convert SymPy expression to CasADi expression
ca_replacements = {'sin': 'ca.sin',
                   'cos': 'ca.cos',
                   'tan': 'ca.tan',
                   'sqrt': 'ca.sqrt',
                   'exp:': 'ca.exp',
                   'log:': 'ca.log',
                   'Abs:': 'ca.fabs',
                   'sign': 'ca.sign',
                   'x': 'self.x',
                   'u': 'self.u',
                   'y': 'self.y'}

def str_replace_all(string, replacements):
    for (key, val) in replacements.items():
        string = string.replace(key, val)
    return string



class ocp_solver(object): # general optimal control problem (ocp) solver class


    def __init__(self, ocp):
        self.h_sim = None       # simulation: step size
        self.tgrid_sim = None   # simulation: time grid
        self.u_opt = None       # optimal input trajectory
        self.x_opt = None       # optimal state trajectory
        self.nlp_solver = None  # CasADi solver object
        # performance statistics
        self.stats = {'success': None,          # IPOPT: local optimum found
                      'iters': None,            # IPOPT: number of iterations
                      'time': None,             # IPOPT: elapsed real time (wall time)
                      'time_per_iter': None,    # IPOPT: elapsed real time per iteration (mean)
                      'x_f_dev': None,          # deviation of final state (Euclidean norm)
                      'J': None}                # total cost

        self.ocp = ocp          # ocp to be solved

        self.x = ca.MX.sym('x', self.ocp.x_dim) # CasADi symbols for state
        self.u = ca.MX.sym('u', self.ocp.u_dim) # CasADi symbols for input


    def initialize(self, h_sim, N, M, u_ig, x_ig, y_ig):    # initialize method parameters
        self.int_tol = 1e-12                # simulation: tolerance of relative and absolute local discretization error of ode solver
        self.h_sim = h_sim                  # simulation: step size
        if u_ig != None: self.u_ig = u_ig   # initial guess for optimal input trajectory
        if x_ig != None: self.x_ig = x_ig   # initial guess for optimal state trajectory
        if y_ig != None: self.y_ig = y_ig   # initial guess for optimal co-state trajectory (indirect methods)
        if N != None: self.N = N            # number of (time) subintervals
        if M != None: self.M = M            # order of collocation method


    def plot_trajectory(self): # plots trajectory and shows performance statistics for testing purposes
        plt.rcParams.update(plt.rcParamsDefault)
        plt.figure(1)
        legend = []
        for i in range(self.ocp.x_dim):
            plt.plot(self.tgrid_sim, self.x_opt[i,:].T, '-')
            legend.append('x_'+ str(i+1) + ': ' + self.ocp.x_dict[str(i+1)])
        for i in range(self.ocp.u_dim):
            plt.plot(self.tgrid_sim[0:-1], self.u_opt[i,:].T, '-', drawstyle=self.u_type, linewidth=1.0)
            legend.append('u_'+ str(i+1) + ': ' + self.ocp.u_dict[str(i+1)])
        plt.xlabel('t')
        plt.legend(legend)
        plt.grid()
        plt.show()

        print(self.stats)



class ocp_i_solver(ocp_solver): # general class for indirect ocp solvers


    def __init__(self, ocp):
        self.y = None
        self.z = None
        self.z_dot = None
        self.u_opt_fcn = None
        self.ode = None
        self.simulator = None
        self.nlpsol_opts = None
        self.z_opt = None

        self.u_type = 'default'     # continuous input trajectory

        super(ocp_i_solver, self).__init__(ocp)

        if not self.ocp.has_objective: print('Error: No objective') # indirect methods require objective (cost functional)
        assert self.ocp.has_objective
        self.ocp.indirect_method() # determines ode and input parametrization based on necessary optimality conditions for indirect methods


    def initialize_ca(self): # initialize CasADi expressions and objects
        # set simulation time grid for ocp to be solved
        self.tgrid_sim = np.linspace(self.ocp.t_0, self.ocp.t_f, round((self.ocp.t_f - self.ocp.t_0)/self.h_sim)+1)

        # convert SymPy expression of state space model to CasADi expression
        self.x_dot = []
        for element in self.ocp.x_dot:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.x_dot.append(eval(element_str_replaced))
        self.x_dot = ca.vertcat(*self.x_dot)

        # convert SymPy expression of integral cost to CasADi expression
        self.L = []
        for element in self.ocp.L:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.L.append(eval(element_str_replaced))
        self.L = ca.vertcat(*self.L)

        self.y = ca.MX.sym('y', self.ocp.x_dim) # CasADi symbols for co-state

        # convert SymPy expression of optimal input parametrization based on necessary optimality conditions to CasADi expression
        self.u_opt = []
        for element in self.ocp.u_noc:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.u_opt.append(eval(element_str_replaced))
        self.u_opt = ca.vertcat(*self.u_opt)

        # consider input constraints
        if np.any(self.ocp.u_min > -np.inf*np.ones(self.ocp.u_dim)):
            self.u_opt = ca.fmax(self.u_opt, self.ocp.u_min)
        if np.any(self.ocp.u_max < np.inf*np.ones(self.ocp.u_dim)):
            self.u_opt = ca.fmin(self.u_opt, self.ocp.u_max)

        self.z = ca.vertcat(self.x, self.y) # collect CasADi symbols for state and co-state

        # convert SymPy expression of ode based on necessary optimality conditions to CasADi expression
        self.z_dot = []
        for element in self.ocp.z_dot:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.z_dot.append(eval(element_str_replaced))
        self.z_dot = ca.vertcat(*self.z_dot)

        # CasADi function of optimal input parametrization based on necessary optimality
        self.u_opt_fcn = ca.Function('u_opt_fcn', [self.z], [self.u_opt])

        # CasADi integrator objects to solve initial value problems using CVODES
        self.ode = {'x': self.z, 'ode': self.z_dot}                                 # ode based on necessary optimality conditions
        self.ode2 = {'x': self.x, 'p': self.u, 'ode': self.x_dot, 'quad': self.L}   # state space model and cost function
        # integrator object for simulation
        self.simulator = ca.integrator('simulator', 'cvodes', self.ode, {'grid': self.tgrid_sim, 'output_t0': True, 'abstol': self.int_tol, 'reltol': self.int_tol})
        # integrator object to calculate total cost
        self.quad = ca.integrator('quad', 'cvodes', self.ode2, {'tf': self.h_sim, 'abstol': self.int_tol, 'reltol': self.int_tol})

        # options for nonlinear program (nlp) solver IPOPT
        self.nlpsol_opts = {'nlpsol': 'ipopt',
                            'nlpsol_options': {'ipopt.max_iter': 1e3, 'ipopt.tol': 1e-6, 'ipopt.constr_viol_tol': 1e-4, 'ipopt.compl_inf_tol': 1e-4, 'ipopt.dual_inf_tol': 1,
                                               'ipopt.acceptable_tol': 1e-6, 'ipopt.acceptable_constr_viol_tol': 1e-4, 'ipopt.acceptable_compl_inf_tol': 1e-4,
                                               'ipopt.acceptable_dual_inf_tol': 1}}


    def get_trajectory(self):
        # determine optimal input and state trajectory using the solution of the ode based on necessary optimality conditions
        u_fcn = self.u_opt_fcn.map(len(self.tgrid_sim))
        self.u_opt = u_fcn(self.z_opt)[0:-1].full()
        self.x_opt = np.zeros((0,len(self.tgrid_sim)))
        for i in range(self.ocp.x_dim):
            self.x_opt = np.append(self.x_opt, self.z_opt[i,:])
        self.x_opt = np.reshape(self.x_opt, (self.ocp.x_dim,len(self.tgrid_sim)))

        # collect solution statistics
        stats = self.nlp_solver.stats()                                             # get IPOPT stats
        self.stats['success'] = stats['nlpsol']['success']                          # IPOPT: local optimum found
        self.stats['time'] = stats['nlpsol']['t_wall_nlpsol']                       # IPOPT: elapsed real time (wall time)
        self.stats['iters'] = stats['nlpsol']['iter_count']                         # IPOPT: number of iterations
        self.stats['time_per_iter'] = self.stats['time']/self.stats['iters']        # IPOPT: elapsed real time per iteration (mean)
        self.stats['x_f_dev'] = np.linalg.norm(self.x_opt[:,-1] - self.ocp.x_f)     # deviation of final state (Euclidean norm)
        # calculate total cost
        J = 0
        for i in range(len(self.tgrid_sim)-1):
            J += self.quad(x0=self.x_opt[:,i], p=self.u_opt[:,i])['qf'].full()[0][0]
        self.stats['J'] = J                                                         # total cost



class ocp_issm_solver(ocp_i_solver): # solver class for indirect single shooting method


    def __init__(self, ocp):
        super(ocp_issm_solver, self).__init__(ocp)


    def solve(self, h_sim, y_ig=None, plot=False):
        self.initialize(h_sim, None, None, None, None, y_ig)
        self.initialize_ca()

        # formulate root-finding problem with the single shooting method, i.e. (consecutive) solution of an initial value problem (ivp) via CasADi integrator object
        I = ca.integrator('I', 'cvodes', self.ode, {'t0': self.ocp.t_0, 'tf': self.ocp.t_f, 'abstol': self.int_tol, 'reltol': self.int_tol})
        y_0 = ca.MX.sym('y_0', self.ocp.x_dim)      # problem variables: intial co-state
        z_0 = ca.vertcat(self.ocp.x_0, y_0)         # collect initial state and co-state
        Z = I(x0=z_0)['xf']                         # implicit expression of the ivp solution via CasADi integrator object
        x_f = Z[0:self.ocp.x_dim] - self.ocp.x_f    # root-finding problem (nonlinear equation) to be solved (condition for final state)
        rfp = ca.Function('rfp', [y_0], [x_f])      # root-finding problem as CasADi function

        # solve the root-finding problem
        self.nlp_solver = ca.rootfinder('solver', 'nlpsol', rfp, self.nlpsol_opts)  # construct rootfinder object using IPOPT as solver
        if y_ig == None:                                                            # initialize problem variables with zeros or consider initial guess
            y_0_opt = np.array(self.nlp_solver(0).nonzeros())
        else:
            y_0_opt = np.array(self.nlp_solver(y_ig).nonzeros())

        # get the complete state and input trajectory via simulation
        self.z_opt = self.simulator(x0=np.concatenate((self.ocp.x_0, y_0_opt)))['xf']
        self.get_trajectory()
        if plot: self.plot_trajectory()



class ocp_imsm_solver(ocp_i_solver): # solver class for indirect multiple shooting method


    def __init__(self, ocp):
        super(ocp_imsm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, x_ig=None, y_ig=None, plot=False):
        self.initialize(h_sim, N, None, None, x_ig, y_ig)
        self.initialize_ca()

        # formulate root-finding problem with the multiple shooting method, i.e. (consecutive) solution of initial value problems (ivp) via CasADi integrator object
        I = ca.integrator('I', 'cvodes', self.ode, {'t0': self.ocp.t_0, 'tf': self.ocp.t_f/self.N, 'abstol': self.int_tol, 'reltol': self.int_tol})
        V = ca.MX.sym('V', (self.N+1)*2*self.ocp.x_dim) # problem variables: state and co-state at the beginning of the N subintervals and at the final time
        # collect the problem variables
        Z = []
        for i in range(self.N+1):
            Z.append(V[i*2*self.ocp.x_dim:(i+1)*2*self.ocp.x_dim])
        # collect the N+1 nonlinear equations of the root-finding problem
        G = []
        G.append(Z[0][0:self.ocp.x_dim] - self.ocp.x_0)         # condition for initial state
        for k in range(self.N):
            Z_end = I(x0=Z[k])['xf']                            # implicit expression of the ivp solution via CasADi integrator object
            G.append(Z_end - Z[k+1])                            # continuity condition
        G.append(Z[self.N][0:self.ocp.x_dim] - self.ocp.x_f)    # condition for final state
        rfp = ca.Function('rfp', [V], [ca.vertcat(*G)])         # root-finding problem as CasADi function

        # solve the root-finding problem
        self.nlp_solver = ca.rootfinder('solver', 'nlpsol', rfp, self.nlpsol_opts)  # construct rootfinder object using IPOPT as solver
        if np.any(x_ig == None) or np.any(y_ig == None):                            # initialize problem variables with zeros or consider initial guess
            z_opt = self.nlp_solver(0)
        else:
            z_opt = self.nlp_solver(np.ravel(np.concatenate([x_ig, y_ig]), 'F'))

        # get the complete state and input trajectory via simulation
        self.z_opt = self.simulator(x0=z_opt[0:2*self.ocp.x_dim])['xf']
        self.get_trajectory()
        if plot: self.plot_trajectory()



class ocp_icm_solver(ocp_i_solver): # solver class for indirect (orthogonal) collocation method


    def __init__(self, ocp):
        self.h_N = None     # (time) discretization step size

        super(ocp_icm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, M, x_ig=None, y_ig=None, plot=False):
        self.initialize(h_sim, N, M, None, x_ig, y_ig)
        self.initialize_ca()
        self.h_N = (self.ocp.t_f - self.ocp.t_0)/self.N                 # set discretization step size for ocp to be solved and N subintervals
        z_dot_fcn = ca.Function('z_dot_fcn', [self.z], [self.z_dot])    # ode based on necessary optimality conditions as CasADi function

        # setup collocation method using one interpolation polynomial (sum of M+1 Lagrange polynomials) on the interval [0,1]
        cp = np.array(ca.collocation_points(self.M, 'legendre'))    # M collocation points (roots of the shifted Legendre polynomial of degree M)
        c = np.array(np.append(0, cp))                              # M+1 interpolation points
        a = np.zeros((self.M+1,self.M+1))                           # derivatives of the M+1 Lagrange polynomials at each interpolation point (needed for collocation equations)
        b = np.zeros(self.M+1)                                      # values of the M+1 Lagrange polynomials at the end of the interval (needed for the continuity equation)
        # determine coefficients (a, b) by constructing M+1 Lagrange polynomials, i.e. one interpolation polynomial
        for j in range(self.M+1):
            # construct the Lagrange polynomial
            p = np.poly1d([1])
            for k in range(self.M+1):
                if k != j:
                    p *= np.poly1d([1, -c[k]]) / (c[j] - c[k])
            # evaluate the Lagrange polynomial at the end of the interval
            b[j] = p(1.0)
            # evaluate the derivative of the Lagrange polynomial at all collocation points
            p_dot = np.polyder(p)
            for r in range(self.M+1):
                a[j,r] = p_dot(c[r])

        # formulate root-finding problem with the collocation method
        V = ca.MX.sym('V', (self.N*(self.M+1)+1)*2*self.ocp.x_dim)  # problem variables: state and co-state at the M+1 interpolation points of the N subintervals and at the final time
        # collect the problem variables
        Z = []
        for i in range(self.N*(self.M+1)+1):
            Z.append(V[i*2*self.ocp.x_dim:(i+1)*2*self.ocp.x_dim])
        # collect the (M+1)N+1 nonlinear equations of the root-finding problem, i.e. collocation and continuity equations for all N subintervals
        # this equates to evaluating interpolation polynomial respectively their derivatives at the interpolations points using the previously determined coefficients (a, b)
        G = []
        G.append(Z[0][0:self.ocp.x_dim] - self.ocp.x_0)                     # condition for initial state
        for k in range(self.N):                                             # loop over subintervals
            Zk_end = b[0]*Z[k*(self.M+1)]                                   # sum over interpolation points to get the states at the end of the subinterval
            for j in range(1,self.M+1):                                     # loop over collocation points to setup collocation equations
                zp = a[0,j]*Z[k*(self.M+1)]/self.h_N                        # sum over interpolation points to get the state derivatives at the collocation point
                for r in range(self.M):                                     # "
                   zp += a[r+1,j]*Z[k*(self.M+1)+r+1]/self.h_N              # "
                z_dot_j = z_dot_fcn(Z[k*(self.M+1)+j])                      # state derivative at the collocation point as demanded by the ode based on optimality conditions
                G.append(z_dot_j - zp)                                      # add collocation equation to the root-finding problem
                Zk_end += b[j]*Z[k*(self.M+1)+j]                            # sum over interpolation points to get the states at the end of the subinterval
            G.append(Zk_end - Z[(k+1)*(self.M+1)])                          # add continuity equation to the root-finding problem
        G.append(Z[self.N*(self.M+1)][0:self.ocp.x_dim] - self.ocp.x_f)     # condition for final state
        rfp = ca.Function('rfp', [V], [ca.vertcat(*G)])                     # root-finding problem as CasADi function

        # solve the root-finding problem
        self.nlp_solver = ca.rootfinder('solver', 'nlpsol', rfp, self.nlpsol_opts)  # construct rootfinder object using IPOPT as solver
        if None in [x_ig, y_ig]:                                                    # initialize problem variables with zeros or consider initial guess
            z_opt = self.nlp_solver(0)
        else:
            z_opt = self.nlp_solver(np.ravel(np.concatenate([x_ig, y_ig]), 'F'))

        # get the complete state and input trajectory via simulation
        self.z_opt = self.simulator(x0=z_opt[0:2*self.ocp.x_dim])['xf']
        self.get_trajectory()
        if plot: self.plot_trajectory()



class ocp_d_solver(ocp_solver): # general class for direct ocp solvers


    def __init__(self, ocp):
        self.h_N = None             # (time) discretization step size
        self.tgrid_N = None         # discretization time grid

        self.u_type = 'steps-post'  # piecewise constant input trajectory

        super(ocp_d_solver, self).__init__(ocp)


    def initialize_ca(self): # initialize CasADi expressions and objects
        # set discretization step size and time grid for ocp to be solved
        self.h_N = (self.ocp.t_f - self.ocp.t_0)/self.N
        self.tgrid_N = np.linspace(self.ocp.t_0, self.ocp.t_f, self.N+1)
        if self.h_N > self.h_sim:   # ensures that input step does not occur within simulation step
            self.h_sim = self.h_N/round(self.h_N/self.h_sim)
        else:
            self.h_sim = self.h_N
        # set simulation time grid for ocp to be solved
        self.tgrid_sim = np.linspace(self.ocp.t_0, self.ocp.t_f, round((self.ocp.t_f - self.ocp.t_0)/self.h_sim)+1)

        # convert SymPy expression of state space model to CasADi expression
        self.x_dot = []
        for element in self.ocp.x_dot:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.x_dot.append(eval(element_str_replaced))
        self.x_dot = ca.vertcat(*self.x_dot)

        # convert SymPy expression of integral cost to CasADi expression
        self.L = []
        for element in self.ocp.L:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.L.append(eval(element_str_replaced))
        self.L = ca.vertcat(*self.L)

        # CasADi integrator objects to solve initial value problems using CVODES
        self.ode = {'x': self.x, 'p': self.u, 'ode': self.x_dot, 'quad': self.L}    # state space model and cost function
        # integrator object for shooting methods
        self.I = ca.integrator('I', 'cvodes', self.ode, {'tf': self.h_N, 'abstol': self.int_tol, 'reltol': self.int_tol})
        # integrator object for simulation
        self.simulator = ca.integrator('simulator', 'cvodes', self.ode, {'tf': self.h_sim, 'abstol': self.int_tol, 'reltol': self.int_tol})

        # empty nonlinear program (nlp)
        self.w = []     # vector of problem variables
        self.w0 = []    # initial guess for vector of problem variables
        self.lbw = []   # vector of lower bounds of problem variables
        self.ubw = []   # vector of upper bounds of problem variables
        self.J = 0      # cost functional
        self.g = []     # vector of general inequality constraints
        self.lbg = []   # vector of lower bounds of general inequality constraints
        self.ubg = []   # vector of upper bounds of general inequality constraints

        # options for nlp solver IPOPT
        self.ipopt_opts = {'ipopt': {'max_iter': 5e3, 'tol': 1e-6, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4, 'dual_inf_tol': 1,
                                     'acceptable_tol': 1e-6, 'acceptable_constr_viol_tol': 1e-4, 'acceptable_compl_inf_tol': 1e-4,
                                     'acceptable_dual_inf_tol': 1}}


    def get_trajectory(self):
        # get state trajectory and total cost via simulation
        self.x_opt = np.reshape(self.ocp.x_0, (self.ocp.x_dim,1))
        J = 0
        for i in range(len(self.tgrid_sim)-1):
            sim_i = self.simulator(x0=self.x_opt[:,-1], p=self.u_opt[:,i])
            self.x_opt = np.append(self.x_opt, np.reshape(np.array(sim_i['xf'].full()), (self.ocp.x_dim,1)), axis=1)
            J += sim_i['qf'].full()[0][0]

        # collect solution statistics
        self.stats['J'] = J                                                         # total cost
        stats = self.nlp_solver.stats()                                             # get IPOPT stats
        self.stats['success'] = stats['success']                                    # IPOPT: local optimum found
        self.stats['time'] = stats['t_wall_solver']                                 # IPOPT: elapsed real time (wall time)
        self.stats['iters'] = stats['iter_count']                                   # IPOPT: number of iterations
        self.stats['time_per_iter'] = self.stats['time']/self.stats['iters']        # IPOPT: elapsed real time per iteration (mean)
        self.stats['x_f_dev'] = np.linalg.norm(self.x_opt[:,-1] - self.ocp.x_f)     # deviation of final state (Euclidean norm)



class ocp_dssm_solver(ocp_d_solver): # solver class for direct single shooting method


    def __init__(self, ocp):
        super(ocp_dssm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, u_ig=None, plot=False):
        self.initialize(h_sim, N, None, u_ig, None, None)
        self.initialize_ca()

        # formulate nonlinear program (nlp) with the single shooting method, i.e. (consecutive) solution of initial value problems (ivp) via CasADi integrator object
        Xk = ca.MX(self.ocp.x_0)        # initial state
        for k in range(self.N):
            # add current input (constant within current subinterval) to problem variables
            Uk = ca.MX.sym('U_' + str(k), self.ocp.u_dim)
            self.w.append(Uk)
            # consider input constraints
            self.lbw.append(self.ocp.u_min)
            self.ubw.append(self.ocp.u_max)
            # initialize input with zeros or consider initial guess
            if u_ig == None:
                self.w0.append(np.zeros(self.ocp.u_dim))
            else:
                self.w0.append(self.u_ig[:,k])
            # solve current ivp
            Ik = self.I(x0=Xk, p=Uk)    # implicit expression of the ivp solution via CasADi integrator object
            Xk = Ik['xf']               # state at the end of the current subinterval
            self.J += Ik['qf']          # quadrature state (cost) at the end of the current subinterval
            # consider conditions for state at the end of the current subinterval
            self.g.append(Xk)
            if k == self.N-1:           # consider condition for final state
                self.lbg.append(self.ocp.x_f)
                self.ubg.append(self.ocp.x_f)
            else:                       # consider state constraints
                self.lbg.append(self.ocp.x_min)
                self.ubg.append(self.ocp.x_max)
        nlp = {'f': self.J, 'x': ca.vertcat(*self.w), 'g': ca.vertcat(*self.g)}     # nlp to be solved
        self.nlp_solver = ca.nlpsol('solver', 'ipopt', nlp, self.ipopt_opts)        # construct nlp solver object using IPOPT as solver

        # solve the nlp
        sol = self.nlp_solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                              lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))

        # get input trajectory from the nlp solution
        u_opt_N = np.reshape(sol['x'], (self.ocp.u_dim,self.N))
        # adjust input trajectory to simulation time grid
        self.u_opt = interp1d(self.tgrid_N[0:-1], u_opt_N, bounds_error=False, fill_value=u_opt_N[:,-1], kind='zero')(self.tgrid_sim[0:-1])

        # get state trajectory and total cost via simulation
        self.get_trajectory()
        if plot: self.plot_trajectory()


class ocp_dmsm_solver(ocp_d_solver): # solver class for direct multiple shooting method


    def __init__(self, ocp):
        super(ocp_dmsm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, u_ig=None, x_ig=None, plot=False):
        self.initialize(h_sim, N, None, u_ig, x_ig, None)
        self.initialize_ca()

        # formulate nonlinear program (nlp) with the multiple shooting method, i.e. (consecutive) solution of an initial value problems (ivp) via CasADi integrator object
        Xk = ca.MX(self.ocp.x_0)                # initial state
        for k in range(self.N):
            # add current input (constant within current subinterval) to problem variables
            Uk = ca.MX.sym('U_' + str(k), self.ocp.u_dim)
            self.w.append(Uk)
            # consider input constraints
            self.lbw.append(self.ocp.u_min)
            self.ubw.append(self.ocp.u_max)
            # initialize current input with zeros or consider initial guess
            if u_ig == None:
                self.w0.append(np.zeros(self.ocp.u_dim))
            else:
                self.w0.append(self.u_ig[:,k])
            # solve current ivp
            Ik = self.I(x0=Xk, p=Uk)            # implicit expression of the ivp solution via CasADi integrator object
            Xk_f = Ik['xf']                     # state at the end of the current subinterval
            self.J += Ik['qf']                  # quadrature state (cost) at the end of the current subinterval
            # consider conditions for state at the end of the current subinterval
            if k == self.N-1:                   # consider condition for final state
                Xk = ca.MX(self.ocp.x_f)
            else:
                # add state at the end of the current subinterval to problem variables
                Xk = ca.MX.sym('X_' + str(k+1), self.ocp.x_dim)
                self.w.append(Xk)
                # consider state constraints
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                # initialize state at the end of the current subinterval with zeros or consider initial guess
                if x_ig == None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k])
            # ensure continuity of state
            self.g.append(Xk_f - Xk)
            self.lbg.append(np.zeros(self.ocp.x_dim))
            self.ubg.append(np.zeros(self.ocp.x_dim))
        nlp = {'f': self.J, 'x': ca.vertcat(*self.w), 'g': ca.vertcat(*self.g)}     # nlp to be solved
        self.nlp_solver = ca.nlpsol('solver', 'ipopt', nlp, self.ipopt_opts)        # construct nlp solver object using IPOPT as solver

        # solve the nlp
        sol = self.nlp_solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                              lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))

        # get input trajectory from the nlp solution
        w_opt = sol['x'].full().flatten()
        u_opt_N = np.zeros((0,self.N))
        for i in range(self.ocp.u_dim):
            u_opt_N = np.append(u_opt_N, w_opt[i::self.ocp.u_dim+self.ocp.x_dim])
        u_opt_N = np.reshape(u_opt_N, (self.ocp.u_dim,self.N))
         # adjust input trajectory to simulation time grid
        self.u_opt = interp1d(self.tgrid_N[0:-1], u_opt_N, bounds_error=False, fill_value=u_opt_N[:,-1], kind='zero')(self.tgrid_sim[0:-1])

        # get state trajectory and total cost via simulation
        self.get_trajectory()
        if plot: self.plot_trajectory()



class ocp_dcm_solver(ocp_d_solver): # solver class for direct (orthogonal) collocation method


    def __init__(self, ocp):
        super(ocp_dcm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, M, u_ig=None, x_ig=None, plot=False):
        self.initialize(h_sim, N, M, u_ig, x_ig, None)
        self.initialize_ca()
        fq_fcn = ca.Function('f', [self.x, self.u], [self.x_dot, self.L], ['x', 'u'], ['xdot', 'L'])    # state space model and cost functional as CasADi function

        # setup collocation method using one interpolation polynomial (sum of M+1 Lagrange polynomials) on the interval [0,1]
        cp = np.array(ca.collocation_points(self.M, 'legendre'))    # M collocation points (roots of the shifted Legendre polynomial of degree M)
        c = np.array(np.append(0, cp))                              # M+1 interpolation points
        a = np.zeros((self.M+1,self.M+1))                           # derivatives of the M+1 Lagrange polynomials at each interpolation point (needed for collocation equations)
        b = np.zeros(self.M+1)                                      # values of the M+1 Lagrange polynomials at the end of the interval (needed for the continuity equation)
        d = np.zeros(self.M+1)                                      # integrals of M+1 Lagrange polynomials over the interval (needed to calculate the cost)
        # determine coefficients (a, b, d) by constructing M+1 Lagrange polynomials, i.e. one interpolation polynomial
        for j in range(self.M+1):
            # construct the Lagrange polynomial
            p = np.poly1d([1])
            for k in range(self.M+1):
                if k != j:
                    p *= np.poly1d([1, -c[k]]) / (c[j] - c[k])
            # evaluate the Lagrange polynomial at the end of the interval
            b[j] = p(1.0)
            # evaluate the derivative of the Lagrange polynomial at all collocation points
            p_dot = np.polyder(p)
            for r in range(self.M+1):
                a[j,r] = p_dot(c[r])
            # evaluate the integral of the polynomial over the interval
            p_int = np.polyint(p)
            d[j] = p_int(1.0)

        # formulate the nlp with the collocation method
        Xk = ca.MX(self.ocp.x_0)    # initial state
        for k in range(self.N):
            # add current input (constant within current subinterval) to problem variables
            Uk = ca.MX.sym('U_' + str(k), self.ocp.u_dim)
            self.w.append(Uk)
            # consider input constraints
            self.lbw.append(self.ocp.u_min)
            self.ubw.append(self.ocp.u_max)
            # initialize current input with zeros or consider initial guess
            if u_ig == None:
                self.w0.append(np.zeros(self.ocp.u_dim))
            else:
                self.w0.append(self.u_ig[:,k])
            # state at collocation points
            Xc = []
            for j in range(self.M):
                # add state at the current collocation point to problem variables
                Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), self.ocp.x_dim)
                Xc.append(Xkj)
                self.w.append(Xkj)
                # consider state constraints
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                # initialize state at the current collocation point with zeros or consider initial guess
                if x_ig == None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k*self.M+j+1])
            # loop over collocation points to setup collocation equations
            Xk_end = b[0]*Xk
            for j in range(1,self.M+1):
               # sum over interpolation points to get the state derivatives at the collocation point
               xp = a[0,j]*Xk/self.h_N
               for r in range(self.M):
                   xp += a[r+1,j]*Xc[r]/self.h_N
               # add collocation equation to the nlp
               fj, qj = fq_fcn(Xc[j-1], Uk)
               self.g.append(fj - xp)
               self.lbg.append(np.zeros(self.ocp.x_dim))
               self.ubg.append(np.zeros(self.ocp.x_dim))
               # sum over interpolation points to get the states at the end of the subinterval
               Xk_end += b[j]*Xc[j-1]
               # contribution to quadrature state (cost)
               self.J = self.J + d[j]*qj*self.h_N
            # state at the end of the current subinterval
            if k == self.N-1:   # consider condition for final state
                Xk = ca.MX(self.ocp.x_f)
            else:
                # add state at the end of the current subinterval to problem variables
                Xk = ca.MX.sym('X_' + str(k+1), self.ocp.x_dim)
                self.w.append(Xk)
                # consider state constraints
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                # initialize state at the end of the current subinterval with zeros or consider initial guess
                if x_ig == None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k*(self.M+1)])
            # add continuity equation to the nlp
            self.g.append(Xk_end - Xk)
            self.lbg.append(np.zeros(self.ocp.x_dim))
            self.ubg.append(np.zeros(self.ocp.x_dim))
        nlp = {'f': self.J, 'x': ca.vertcat(*self.w), 'g': ca.vertcat(*self.g)} # nlp to be solved
        self.nlp_solver = ca.nlpsol('solver', 'ipopt', nlp, self.ipopt_opts)    # construct nlp solver object using IPOPT as solver

        # solve the nlp
        sol = self.nlp_solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                              lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))

        # get input trajectory from the nlp solution
        w_opt = sol['x'].full().flatten()
        self.w_opt = w_opt
        u_opt_N = np.zeros((self.ocp.u_dim,0))
        for i in range(self.N):
            u_opt_N = np.append(u_opt_N, np.reshape(w_opt[(self.M+1)*self.ocp.x_dim*i + self.ocp.u_dim*i:(self.M+1)*self.ocp.x_dim*i + self.ocp.u_dim*(i+1)], (self.ocp.u_dim,1)), axis=1)
        u_opt_N = np.reshape(u_opt_N, (self.ocp.u_dim,self.N))
        self.u_opt_N = u_opt_N
        # adjust input trajectory to simulation time grid
        self.u_opt = interp1d(self.tgrid_N[0:-1], u_opt_N, bounds_error=False, fill_value=u_opt_N[:,-1], kind='zero')(self.tgrid_sim[0:-1])

        # get state trajectory and total cost via simulation
        self.get_trajectory()
        if plot: self.plot_trajectory()



class ocp_dcm2_solver(ocp_d_solver): # solver class for direct (orthogonal) collocation method with piecewise polynomial input parametrization


    def __init__(self, ocp):
        super(ocp_dcm2_solver, self).__init__(ocp)

        self.u_type = 'default' # continuous input trajectory


    def solve(self, h_sim, N, M, u_ig=None, x_ig=None, plot=False):
        self.initialize(h_sim, N, M, u_ig, x_ig, None)
        self.initialize_ca()
        fq_fcn = ca.Function('f', [self.x, self.u], [self.x_dot, self.L], ['x', 'u'], ['xdot', 'L'])    # state space model and cost functional as CasADi function

        # setup collocation method
        cp = np.array(ca.collocation_points(self.M, 'legendre'))    # M collocation points (roots of the shifted Legendre polynomial of degree M)
        # one interpolation polynomial (sum of M+1 Lagrange polynomials) on the interval [0,1] for state
        c = np.array(np.append(0, cp))                              # M+1 interpolation points
        a = np.zeros((self.M+1,self.M+1))                           # derivatives of the M+1 Lagrange polynomials at each interpolation point (needed for collocation equations)
        b = np.zeros(self.M+1)                                      # values of the M+1 Lagrange polynomials at the end of the interval (needed for the continuity equation)
        d = np.zeros(self.M+1)                                      # integrals of M+1 Lagrange polynomials over the interval (needed to calculate the cost)
        # determine coefficients (a, b, d) by constructing M+1 Lagrange polynomials, i.e. one interpolation polynomial
        for j in range(self.M+1):
            # construct the Lagrange polynomial
            p = np.poly1d([1])
            for k in range(self.M+1):
                if k != j:
                    p *= np.poly1d([1, -c[k]]) / (c[j] - c[k])
            # evaluate the Lagrange polynomial at the end of the interval
            b[j] = p(1.0)
            # evaluate the derivative of the Lagrange polynomial at all collocation points
            p_dot = np.polyder(p)
            for r in range(self.M+1):
                a[j,r] = p_dot(c[r])
            # evaluate the integral of the polynomial over the interval
            p_int = np.polyint(p)
            d[j] = p_int(1.0)
        # one interpolation polynomial (sum of M Lagrange polynomials) on the interval [0,1] for input
        bu = np.zeros(self.M)                                       # values of the M Lagrange polynomials at the end of the interval
        bu0 = np.zeros(self.M)                                      # values of the M Lagrange polynomials at the beginning of the interval
        # determine coefficients (bu, bu0) by constructing M Lagrange polynomials, i.e. one interpolation polynomial
        pu_dots = [] # derivatives of the Lagrange polynomials
        for j in range(self.M):
            # construct the Lagrange polynomial
            pu = np.poly1d([1])
            for k in range(self.M):
                if k != j:
                    pu *= np.poly1d([1, -cp[k]]) / (cp[j] - cp[k])
            # evaluate the Lagrange polynomial at the end and beginning of the interval
            bu[j] = pu(1.0)
            bu0[j] = pu(0.0)
            # determine derivative of the Lagrange polynomial
            pu_dots.append(np.polyder(pu))
        # insert piecewise polynomial input parametrization in state space model (needed for accurate simulation results)
        uc = []                     # symbolic variables for input at collocation points
        for j in range(M):
            uc.append(ca.MX.sym('uc_' + str(j+1), self.ocp.u_dim))
        c_t = ca.MX.sym('c')        # symbolic variable for point within interval [0,1]
        pc = ca.vertcat(*uc, c_t)   # collect variables (ode parameters)
        c_t_sp = sp.symbols('c_t')  # SymPy symbolic variable for point within interval [0,1] (in order to insert in numpy polynomials pu_dots)
        # determine ode for input according to polynomial parametrization
        u_dot = 0
        for j in range(self.M):
            u_dot += eval(str(pu_dots[j](c_t_sp)))*uc[j]/self.h_N
        # setup new state space model and corresponding CasADi integrator object
        self.xu = ca.vertcat(self.x, self.u)            # new state consists of system state and input
        self.xu_dot = ca.vertcat(self.x_dot, u_dot)     # new state space model
        self.ode2 = {'x': self.xu, 'p': pc, 'ode': self.xu_dot, 'quad': self.L}
        self.simulator2 = ca.integrator('simulator2', 'cvodes', self.ode2, {'tf': self.h_sim, 'abstol': self.int_tol, 'reltol': self.int_tol})

        # formulate the nlp with the collocation method
        Xk = ca.MX(self.ocp.x_0)    # initial state
        Uk_end = None               # input at the end of the last subinterval
        for k in range(self.N):
            # input at collocation points
            Uc = []
            for j in range(self.M):
                # add input at the current collocation point to problem variables
                Ukj = ca.MX.sym('U_' + str(k) + '_' + str(j), self.ocp.u_dim)
                Uc.append(Ukj)
                self.w.append(Ukj)
                # consider input constraints
                self.lbw.append(self.ocp.u_min)
                self.ubw.append(self.ocp.u_max)
                # initialize input at the current collocation point with zeros or consider initial guess
                if u_ig == None:
                    self.w0.append(np.zeros(self.ocp.u_dim))
                else:
                    self.w0.append(self.u_ig[:,k*self.M+j])
            # state at collocation points
            Xc = []
            for j in range(self.M):
                # add state at the current collocation point to problem variables
                Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), self.ocp.x_dim)
                Xc.append(Xkj)
                self.w.append(Xkj)
                # consider state constraints
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                # initialize state at the current collocation point with zeros or consider initial guess
                if x_ig == None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k*self.M+j+1])
            # loop over collocation points to setup collocation equations
            Xk_end = b[0]*Xk
            for j in range(1,self.M+1):
               # sum over interpolation points to get the state derivatives at the collocation point
               xp = a[0,j]*Xk/self.h_N
               for r in range(self.M):
                   xp += a[r+1,j]*Xc[r]/self.h_N
               # add collocation equation to the nlp
               fj, qj = fq_fcn(Xc[j-1], Uc[j-1])
               self.g.append(fj - xp)
               self.lbg.append(np.zeros(self.ocp.x_dim))
               self.ubg.append(np.zeros(self.ocp.x_dim))
               # sum over interpolation points to get the states at the end of the subinterval
               Xk_end += b[j]*Xc[j-1]
               # contribution to quadrature state (cost)
               self.J = self.J + d[j]*qj*self.h_N
            # state at the end of the current subinterval
            if k == self.N-1:   # consider condition for final state
                Xk = ca.MX(self.ocp.x_f)
            else:
                # add state at the end of the current subinterval to problem variables
                Xk = ca.MX.sym('X_' + str(k+1), self.ocp.x_dim)
                self.w.append(Xk)
                # consider state constraints
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                # initialize state at the end of the current subinterval with zeros or consider initial guess
                if x_ig == None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k*(self.M+1)])
            # add continuity equation to the nlp
            self.g.append(Xk_end - Xk)
            self.lbg.append(np.zeros(self.ocp.x_dim))
            self.ubg.append(np.zeros(self.ocp.x_dim))

            Uk_0 = 0
            for j in range(self.M):
                Uk_0 += bu0[j]*Uc[j]
            if k > 0:
                self.g.append(Uk_end - Uk_0)
                self.lbg.append(np.zeros(self.ocp.u_dim))
                self.ubg.append(np.zeros(self.ocp.u_dim))
            Uk_end = 0
            for j in range(self.M):
                Uk_end += bu[j]*Uc[j]
        self.w = ca.vertcat(*self.w)
        self.g = ca.vertcat(*self.g)
        nlp = {'f': self.J, 'x': self.w, 'g': self.g}                           # nlp to be solved
        self.nlp_solver = ca.nlpsol('solver', 'ipopt', nlp, self.ipopt_opts)    # construct nlp solver object using IPOPT as solver

        # solve the nlp
        sol = self.nlp_solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                              lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))

        # get input at collocation points from the nlp solution
        w_opt = sol['x'].full().flatten()
        self.w_opt = w_opt
        u_opt_NM = np.zeros((self.ocp.u_dim,0))
        for i in range(self.N):
            u_opt_NM = np.append(u_opt_NM, w_opt[(self.M+1)*self.ocp.x_dim*i + self.M*self.ocp.u_dim*i:(self.M+1)*self.ocp.x_dim*i + self.M*self.ocp.u_dim*(i+1)])
        u_opt_NM = np.reshape(u_opt_NM, (self.ocp.u_dim,self.N*self.M))

        # collect ode parameters (input at collocation points)
        self.p_opt = []
        for i in range(self.N):
            p_opt_i = np.zeros((self.ocp.u_dim*M,0))
            for j in range(self.M):
                p_opt_i = np.append(p_opt_i, u_opt_NM[:,i*self.M+j])
            self.p_opt.append(p_opt_i)

        # get state and input trajectory and total cost via simulation
        u_opt_0 = 0
        for j in range(self.M):
            u_opt_0 += bu0[j]*u_opt_NM[:,j]
        self.xu_opt = np.reshape(np.append(self.ocp.x_0, u_opt_0), (self.ocp.x_dim+self.ocp.u_dim,1))
        J = 0
        for i in range(len(self.tgrid_sim)-1):
            pc_opt_i = np.append(self.p_opt[int(i*self.h_sim/self.h_N)], (self.tgrid_sim[i] - self.tgrid_N[int(i*self.h_sim/self.h_N)])/self.h_N)
            if self.tgrid_sim[i] in self.tgrid_N:
                u_opt_0 = 0
                for j in range(self.M):
                    u_opt_0 += bu0[j]*u_opt_NM[:,np.where(self.tgrid_N==self.tgrid_sim[i])[0][0]*self.M+j]
                xu_opt_N = np.append(self.xu_opt[0:self.ocp.x_dim,-1], u_opt_0)
                sim_i = self.simulator2(x0=xu_opt_N, p=pc_opt_i)
            else:
                sim_i = self.simulator2(x0=self.xu_opt[:,-1], p=pc_opt_i)
            self.xu_opt = np.append(self.xu_opt, np.reshape(np.array(sim_i['xf'].full()), (self.ocp.x_dim+self.ocp.u_dim,1)), axis=1)
            J += sim_i['qf'].full()[0][0]
        self.x_opt = self.xu_opt[0:self.ocp.x_dim,:]
        self.u_opt = self.xu_opt[self.ocp.x_dim:self.ocp.x_dim+self.ocp.u_dim,0:-1]

        # collect solution statistics
        self.stats['J'] = J                                                         # total cost
        stats = self.nlp_solver.stats()                                             # get IPOPT stats
        self.stats['success'] = stats['success']                                    # IPOPT: local optimum found
        self.stats['time'] = stats['t_wall_solver']                                 # IPOPT: elapsed real time (wall time)
        self.stats['iters'] = stats['iter_count']                                   # IPOPT: number of iterations
        self.stats['time_per_iter'] = self.stats['time']/self.stats['iters']        # IPOPT: elapsed real time per iteration (mean)
        self.stats['x_f_dev'] = np.linalg.norm(self.x_opt[:,-1] - self.ocp.x_f)     # deviation of final state (Euclidean norm)

        if plot: self.plot_trajectory()



if __name__ == '__main__':


    problem_names = {'1': 'double_int', '2': 'pend', '3': 'pend_cart_pl', '4': 'pend_cart',
                     '5': 'dual_pend_cart_pl', '6': 'dual_pend_cart',
                     '7': 'vtol', '8': 'ua_manipulator_pl', '9': 'acrobot_pl',
                     '10': 'double_pend_cart_pl', '11': 'triple_pend_cart_pl'}


    ocp1 = ocp(problem_names['11'])
#    ocp1 = ocp(problem_names['5'], False)
#    ocp1 = ocp(problem_names['5'], False, 1.5)

    h_sim = 0.001
    N = 400
    M = 4

    tgrid_sim = np.linspace(ocp1.t_0, ocp1.t_f, round((ocp1.t_f - ocp1.t_0)/h_sim)+1)
    tgrid_N = np.linspace(ocp1.t_0, ocp1.t_f, N+1)
    tgrid_NM = np.linspace(ocp1.t_0, ocp1.t_f, N*(M+1))

#    dssm_solver = ocp_dssm_solver(ocp1)
#    dssm_solver.solve(h_sim, N, None, True)

#    dmsm_solver = ocp_dmsm_solver(ocp1)
#    dmsm_solver.solve(h_sim, N, None, None, True)

    dcm_solver = ocp_dcm_solver(ocp1)
    dcm_solver.solve(h_sim, N, M, None, None, True)

#    dcm2_solver = ocp_dcm2_solver(ocp1)
#    dcm2_solver.solve(h_sim, N, M, None, None, True)

#    x_opt = dcm_solver.x_opt
#    x_ig = interp1d(tgrid_sim, x_opt, kind='zero')(tgrid_NM)
#    x_ig = interp1d(tgrid_sim, x_opt, kind='zero')(tgrid_N)
#    y_ig = np.zeros(x_ig.shape)
#    u_opt = dcm_solver.u_opt
#    u_ig = interp1d(tgrid_sim[0:-1], u_opt, kind='zero')(tgrid_N[0:-1])

#    dcm_solver = ocp_dcm_solver(ocp1)
#    dcm_solver.solve(h_sim, N, M, False, False, u_ig, x_ig, True)


#    issm_solver = ocp_issm_solver(ocp1)
#    issm_solver.solve(h_sim, None, True)

#    imsm_solver = ocp_imsm_solver(ocp1)
#    imsm_solver.solve(h_sim, N, x_ig, y_ig)
#    imsm_solver.solve(h_sim, N, None, None, True)

#    icm_solver = ocp_icm_solver(ocp1)
#    icm_solver.solve(h_sim, N, M, None, None, True)
