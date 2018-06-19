# -*- coding: utf-8 -*-
'''
Created on Mon May 14 12:35:31 2018

@author: Patrick Rüdiger
'''

#     Partially based on CasADi examples.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
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
import symbtools as st
import casadi as ca
#from plotter import plot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ocp import ocp



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



class ocp_solver(object):


    def __init__(self, ocp):
        self.h_sim = None
        self.tgrid_sim = None
        self.u_opt = None
        self.x_opt = None

        self.ocp = ocp

        self.x = ca.MX.sym('x', self.ocp.x_dim)
        self.u = ca.MX.sym('u', self.ocp.u_dim)


    def initialize(self, h_sim, N, M, u_ig, x_ig, y_ig):
        self.int_tol = 1e-10
        self.h_sim = h_sim
        if u_ig is not None: self.u_ig = u_ig
        if x_ig is not None: self.x_ig = x_ig
        if y_ig is not None: self.y_ig = y_ig
        if N is not None: self.N = N
        if M is not None: self.M = M

        # Time grid for simulation
        self.tgrid_sim = np.linspace(self.ocp.t_0, self.ocp.t_f, round((self.ocp.t_f - self.ocp.t_0)/self.h_sim)+1)


    def plot_trajectory(self, step_u=False):
        plt.figure(1)
        legend = []
        for i in range(self.ocp.x_dim):
            plt.plot(self.tgrid_sim, self.x_opt[i,:].T, '-')
            legend.append('x_'+ str(i+1) + ': ' + self.ocp.x_dict[str(i+1)])
        for i in range(self.ocp.u_dim):
            if not step_u:
                plt.plot(self.tgrid_sim[0:-1], self.u_opt[i,:].T, '-')
            else:
                plt.step(self.tgrid_sim[0:-1], self.u_opt[i,:].T, '-', where='post')
            legend.append('u_'+ str(i+1) + ': ' + self.ocp.u_dict[str(i+1)])
        plt.xlabel('t')
        plt.legend(legend)
        plt.grid()
        plt.show()



class ocp_i_solver(ocp_solver):


    def __init__(self, ocp):
        self.y = None
        self.u_opt = None
        self.z = None
        self.z_dot = None
        self.u_opt_fcn = None
        self.ode = None
        self.simulator = None
        self.nlpsol_opts = None
        self.z_opt = None

        super(ocp_i_solver, self).__init__(ocp)

        if not self.ocp.has_objective: print('Error: No objective')
        assert self.ocp.has_objective
        self.ocp.setup_optbvp()


    def initialize_ca(self):
        self.y = ca.MX.sym('y', self.ocp.x_dim)

        self.u_opt = []
        for element in self.ocp.u_noc:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.u_opt.append(eval(element_str_replaced))
        self.u_opt = ca.vertcat(*self.u_opt)

        if np.any(self.ocp.u_min > -np.inf*np.ones(self.ocp.u_dim)):
            self.u_opt = ca.fmax(self.u_opt, self.ocp.u_min)
        if np.any(self.ocp.u_max < np.inf*np.ones(self.ocp.u_dim)):
            self.u_opt = ca.fmin(self.u_opt, self.ocp.u_max)

        self.z = ca.vertcat(self.x, self.y)

        self.z_dot = []
        for element in self.ocp.z_dot:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.z_dot.append(eval(element_str_replaced))
        self.z_dot = ca.vertcat(*self.z_dot)

        self.u_opt_fcn = ca.Function('u_opt_fcn', [self.z], [self.u_opt])

        self.ode = {'x': self.z, 'ode': self.z_dot}

        # Simulator to get optimal state and control trajectories
        self.simulator = ca.integrator('simulator', 'cvodes', self.ode, {'grid':self.tgrid_sim,'output_t0':True, 'abstol': self.int_tol, 'reltol': self.int_tol})

        self.nlpsol_opts = {'nlpsol': 'ipopt',
                            'nlpsol_options': {'ipopt.hessian_approximation': 'limited-memory', 'ipopt.max_iter': 1e3, 'ipopt.tol': 1e-8, 'ipopt.constr_viol_tol': 1e-4, 'ipopt.compl_inf_tol': 1e-4, 'ipopt.dual_inf_tol': 1,
                                               'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_constr_viol_tol': 1e-4, 'ipopt.acceptable_compl_inf_tol': 1e-4,
                                               'ipopt.acceptable_dual_inf_tol': 1}}


    def get_trajectory(self):
        u_fcn = self.u_opt_fcn.map(len(self.tgrid_sim))
        self.u_opt = u_fcn(self.z_opt)[0:-1]
        self.x_opt = np.zeros((0,len(self.tgrid_sim)))
        for i in range(self.ocp.x_dim):
            self.x_opt = np.append(self.x_opt, self.z_opt[i,:])
        self.x_opt = np.reshape(self.x_opt, (self.ocp.x_dim,len(self.tgrid_sim)))



class ocp_issm_solver(ocp_i_solver):


    def __init__(self, ocp):
        super(ocp_issm_solver, self).__init__(ocp)


    def solve(self, h_sim, y_ig=None):
        self.initialize(h_sim, None, None, None, None, y_ig)
        self.initialize_ca()

        # indirect single shooting method

        # Create an integrator (CVodes)
        I = ca.integrator('I', 'cvodes', self.ode, {'t0': self.ocp.t_0, 'tf': self.ocp.t_f, 'abstol': self.int_tol, 'reltol': self.int_tol})

        y_0 = ca.MX.sym('y_0', self.ocp.x_dim)
        z_0 = ca.vertcat(self.ocp.x_0, y_0)

        Z = I(x0=z_0)['xf']
        x_f = Z[0:self.ocp.x_dim] - self.ocp.x_f

        # Formulate root-finding problem
        rfp = ca.Function('rfp', [y_0], [x_f])

        # Allocate an implict solver
        solver = ca.rootfinder('solver', 'nlpsol', rfp, self.nlpsol_opts)

        # Solve the problem
        if y_ig is None:
            y_0_opt = np.array(solver(0).nonzeros())
        else:
            y_0_opt = np.array(solver(y_ig).nonzeros())

        # Simulate to get the state trajectory
        self.z_opt = self.simulator(x0=np.concatenate((self.ocp.x_0, y_0_opt)))['xf']

        self.get_trajectory()
        self.plot_trajectory()



class ocp_imsm_solver(ocp_i_solver):


    def __init__(self, ocp):
        super(ocp_imsm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, x_ig=None, y_ig=None):
        self.initialize(h_sim, N, None, None, x_ig, y_ig)
        self.initialize_ca()

        # indirect multiple shooting method

        # Create an integrator (CVodes)
        I = ca.integrator('I', 'cvodes', self.ode, {'t0': self.ocp.t_0, 'tf': self.ocp.t_f/self.N, 'abstol': self.int_tol, 'reltol': self.int_tol})

        # Variables in the root finding problem
        V = ca.MX.sym('V', (self.N+1)*2*self.ocp.x_dim)

        Z = []
        for i in range(self.N+1):
            Z.append(V[i*2*self.ocp.x_dim:(i+1)*2*self.ocp.x_dim])

        # Formulate the root finding problem
        G = []
        G.append(Z[0][0:self.ocp.x_dim] - self.ocp.x_0)
        for k in range(self.N):
            Z_end = I(x0=Z[k])['xf']
            G.append(Z_end - Z[k+1])
        G.append(Z[self.N][0:self.ocp.x_dim] - self.ocp.x_f)

        rfp = ca.Function('rfp', [V], [ca.vertcat(*G)])

        # Allocate a solver
        solver = ca.rootfinder('solver', 'nlpsol', rfp, self.nlpsol_opts)

        # Solve the problem
        if np.any(x_ig == None) or np.any(y_ig == None):
            z_opt = solver(0)
        else:
            z_opt = solver(np.ravel(np.concatenate([x_ig, y_ig]), 'F'))

        # Simulate to get the trajectories
        self.z_opt = self.simulator(x0=z_opt[0:2*self.ocp.x_dim])['xf']

        self.get_trajectory()
        self.plot_trajectory()



class ocp_icm_solver(ocp_i_solver):


    def __init__(self, ocp):
        self.h_N = None

        super(ocp_icm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, M, x_ig=None, y_ig=None):
        self.initialize(h_sim, N, M, None, x_ig, y_ig)
        self.initialize_ca()
        self.h_N = (self.ocp.t_f - self.ocp.t_0)/self.N

        # indirect collocation method

        z_dot_fcn = ca.Function('z_dot_fcn', [self.z], [self.z_dot])

        cp = np.array(ca.collocation_points(self.M, 'legendre'))
        c = np.array(np.append(0, cp)) # 'radau' or 'legendre'

        # Coefficients of the collocation equation
        a = np.zeros((self.M+1,self.M+1))

        # Coefficients of the continuity equation
        b = np.zeros(self.M+1)

        # Construct polynomial basis
        for j in range(self.M+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for k in range(self.M+1):
                if k != j:
                    p *= np.poly1d([1, -c[k]]) / (c[j] - c[k])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            b[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            p_dot = np.polyder(p)
            for r in range(self.M+1):
                a[j,r] = p_dot(c[r])

        # Variables in the root finding problem
        V = ca.MX.sym('V', (self.N*(self.M+1)+1)*2*self.ocp.x_dim)

        Z = []
        for i in range(self.N*(self.M+1)+1):
            Z.append(V[i*2*self.ocp.x_dim:(i+1)*2*self.ocp.x_dim])

        # Formulate the root finding problem
        G = []
        G.append(Z[0][0:self.ocp.x_dim] - self.ocp.x_0)
        for k in range(self.N):
            # Loop over collocation points
            Zk_end = b[0]*Z[k*(self.M+1)]
            for j in range(1,self.M+1):
               # Expression for the state derivative at the collocation point
               zp = a[0,j]*Z[k*(self.M+1)]
               for r in range(self.M):
                   zp += a[r+1,j]*Z[k*(self.M+1)+r+1]

               z_dot_j = z_dot_fcn(Z[k*(self.M+1)+j])

               # Append collocation equations
               G.append(self.h_N*z_dot_j - zp)

               # Add contribution to the end state
               Zk_end += b[j]*Z[k*(self.M+1)+j]

            G.append(Zk_end - Z[(k+1)*(self.M+1)])

        G.append(Z[self.N*(self.M+1)][0:self.ocp.x_dim] - self.ocp.x_f)

        rfp = ca.Function('rfp', [V], [ca.vertcat(*G)])

        # Allocate a solver
        solver = ca.rootfinder('solver', 'nlpsol', rfp, self.nlpsol_opts)

        # Solve the problem
        if None in [x_ig, y_ig]:
            z_opt = solver(0)
        else:
            z_opt = solver(np.ravel(np.concatenate([x_ig, y_ig]), 'F'))

        # Simulate to get the trajectories
        self.z_opt = self.simulator(x0=z_opt[0:2*self.ocp.x_dim])['xf']

        self.get_trajectory()
        self.plot_trajectory()



class ocp_d_solver(ocp_solver):


    def __init__(self, ocp):
        self.h_N = None
        self.tgrid_N = None

        super(ocp_d_solver, self).__init__(ocp)


    def initialize_ca(self):
        self.h_N = (self.ocp.t_f - self.ocp.t_0)/self.N
        self.tgrid_N = np.linspace(self.ocp.t_0, self.ocp.t_f, self.N+1)

        self.x_dot = []
        for element in self.ocp.x_dot:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.x_dot.append(eval(element_str_replaced))
        self.x_dot = ca.vertcat(*self.x_dot)

        self.L = []
        for element in self.ocp.L:
            element_str_replaced = str_replace_all(str(element), ca_replacements)
            self.L.append(eval(element_str_replaced))
        self.L = ca.vertcat(*self.L)

        # CVODES from the SUNDIALS suite
        self.dae = {'x': self.x, 'p': self.u, 'ode': self.x_dot, 'quad': self.L}
        self.I = ca.integrator('I', 'cvodes', self.dae, {'tf': self.h_N, 'abstol': self.int_tol, 'reltol': self.int_tol})

        self.ode = {'x': self.x, 'p': self.u, 'ode': self.x_dot}
        self.simulator = ca.integrator('simulator', 'cvodes', self.ode, {'tf': self.h_sim, 'abstol': self.int_tol, 'reltol': self.int_tol})

        # Start with an empty NLP
        self.w = []
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.J = 0
        self.g = []
        self.lbg = []
        self.ubg = []

        self.ipopt_opts = {'ipopt': {'max_iter': 1e3, 'tol': 1e-8, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4, 'dual_inf_tol': 1,
                                     'acceptable_tol': 1e-4, 'acceptable_constr_viol_tol': 1e-4, 'acceptable_compl_inf_tol': 1e-4,
                                     'acceptable_dual_inf_tol': 1}}


    def get_trajectory(self):
        self.x_opt = np.reshape(self.ocp.x_0, (self.ocp.x_dim,1))
        for i in range(len(self.tgrid_sim)-1):
            x_opt_i = self.simulator(x0=self.x_opt[:,-1], p=self.u_opt[:,i])
            self.x_opt = np.append(self.x_opt, np.reshape(np.array(x_opt_i['xf'].full()), (self.ocp.x_dim,1)), axis=1)



class ocp_dssm_solver(ocp_d_solver):


    def __init__(self, ocp):
        super(ocp_dssm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, u_ig=None):
        self.initialize(h_sim, N, None, u_ig, None, None)
        self.initialize_ca()

        # direct single shooting method

        # Formulate the NLP
        Xk = ca.MX(self.ocp.x_0)
        for k in range(self.N):
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), self.ocp.u_dim)
            self.w.append(Uk)
            self.lbw.append(self.ocp.u_min)
            self.ubw.append(self.ocp.u_max)
            if u_ig is None:
                self.w0.append(np.zeros(self.ocp.u_dim))
            else:
                self.w0.append(self.u_ig[:,k])

            # Integrate till the end of the interval
            Ik = self.I(x0=Xk, p=Uk)
            Xk = Ik['xf']
            self.J += Ik['qf']

            # Add inequality constraint
            self.g.append(Xk)
            if k == self.N-1:
                self.lbg.append(self.ocp.x_f)
                self.ubg.append(self.ocp.x_f)
            else:
                self.lbg.append(self.ocp.x_min)
                self.ubg.append(self.ocp.x_max)

        # Create an NLP solver
        prob = {'f': self.J, 'x': ca.vertcat(*self.w), 'g': ca.vertcat(*self.g)}
        solver = ca.nlpsol('solver', 'ipopt', prob, self.ipopt_opts)

        # Solve the NLP
        sol = solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                     lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))
        u_opt_N = np.reshape(sol['x'], (self.ocp.u_dim,self.N))
        self.u_opt = interp1d(self.tgrid_N[0:-1], u_opt_N, bounds_error=False, fill_value=u_opt_N[:,-1], kind='zero')(self.tgrid_sim[0:-1])

        self.get_trajectory()
        self.plot_trajectory(True)


class ocp_dmsm_solver(ocp_d_solver):


    def __init__(self, ocp):
        super(ocp_dmsm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, u_ig=None, x_ig=None):
        self.initialize(h_sim, N, None, u_ig, x_ig, None)
        self.initialize_ca()

        # direct multiple shooting method

        # Formulate the NLP
        Xk = ca.MX(self.ocp.x_0)
        for k in range(self.N):
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), self.ocp.u_dim)
            self.w.append(Uk)
            self.lbw.append(self.ocp.u_min)
            self.ubw.append(self.ocp.u_max)
            if u_ig is None:
                self.w0.append(np.zeros(self.ocp.u_dim))
            else:
                self.w0.append(self.u_ig[:,k])

            # Integrate till the end of the interval
            Ik = self.I(x0=Xk, p=Uk)
            Xk_f = Ik['xf']
            self.J += Ik['qf']

            # Add inequality constraint
            if k == self.N-1:
                Xk = ca.MX(self.ocp.x_f)
            else:
                Xk = ca.MX.sym('X_' + str(k+1), self.ocp.x_dim)
                self.w.append(Xk)
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                if x_ig is None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k])
            self.g.append(Xk_f - Xk)
            self.lbg.append(np.zeros(self.ocp.x_dim))
            self.ubg.append(np.zeros(self.ocp.x_dim))

        # Create an NLP solver
        prob = {'f': self.J, 'x': ca.vertcat(*self.w), 'g': ca.vertcat(*self.g)}
        solver = ca.nlpsol('solver', 'ipopt', prob, self.ipopt_opts)

        # Solve the NLP
        sol = solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                     lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))
        w_opt = sol['x'].full().flatten()
        u_opt_N = np.zeros((0,self.N))
        for i in range(self.ocp.u_dim):
            u_opt_N = np.append(u_opt_N, w_opt[i::self.ocp.u_dim+self.ocp.x_dim])
        u_opt_N = np.reshape(u_opt_N, (self.ocp.u_dim,self.N))

        self.u_opt = interp1d(self.tgrid_N[0:-1], u_opt_N, bounds_error=False, fill_value=u_opt_N[:,-1], kind='zero')(self.tgrid_sim[0:-1])

        self.get_trajectory()
        self.plot_trajectory(True)



class ocp_dcm_solver(ocp_d_solver):


    def __init__(self, ocp):
        super(ocp_dcm_solver, self).__init__(ocp)


    def solve(self, h_sim, N, M, u_polynomial, u_continuous=False, u_ig=None, x_ig=None):
        self.initialize(h_sim, N, M, u_ig, x_ig, None)
        self.initialize_ca()

        # direct collocation method

        # Continuous time dynamics
        fL_fcn = ca.Function('f', [self.x, self.u], [self.x_dot, self.L], ['x', 'u'], ['xdot', 'L'])

        cp = np.array(ca.collocation_points(self.M, 'legendre'))
        c = np.array(np.append(0, cp)) # 'radau' or 'legendre'

        # Coefficients of the collocation equation
        a = np.zeros((self.M+1,self.M+1))

        # Coefficients of the continuity equation
        b = np.zeros(self.M+1)
        bu = np.zeros(self.M)
        bu0 = np.zeros(self.M)

        # Coefficients of the quadrature function
        d = np.zeros(self.M+1)

        # Construct polynomial basis
        for j in range(self.M+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for k in range(self.M+1):
                if k != j:
                    p *= np.poly1d([1, -c[k]]) / (c[j] - c[k])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            b[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            p_dot = np.polyder(p)
            for r in range(self.M+1):
                a[j,r] = p_dot(c[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            p_int = np.polyint(p)
            d[j] = p_int(1.0)

        if u_polynomial:
            Pu = []
            for j in range(self.M):
                pu = np.poly1d([1])
                for k in range(self.M):
                    if k != j:
                        pu *= np.poly1d([1, -cp[k]]) / (cp[j] - cp[k])
                Pu.append(pu)

                bu[j] = pu(1.0)
                bu0[j] = pu(0.0)

        # Formulate the NLP
        Xk = ca.MX(self.ocp.x_0)
        Uk_end = None
        for k in range(self.N):
            # State and control at collocation points
            if u_polynomial:
                Uc = []
                for j in range(self.M):
                    Ukj = ca.MX.sym('U_' + str(k) + '_' + str(j), self.ocp.u_dim)
                    Uc.append(Ukj)
                    self.w.append(Ukj)
                    self.lbw.append(self.ocp.u_min)
                    self.ubw.append(self.ocp.u_max)
                    if u_ig is None:
                        self.w0.append(np.zeros(self.ocp.u_dim))
                    else:
                        self.w0.append(self.u_ig[:,k*self.M+j])
            else:
                Uk = ca.MX.sym('U_' + str(k), self.ocp.u_dim)
                self.w.append(Uk)
                self.lbw.append(self.ocp.u_min)
                self.ubw.append(self.ocp.u_max)
                if u_ig is None:
                    self.w0.append(np.zeros(self.ocp.u_dim))
                else:
                    self.w0.append(self.u_ig[:,k])

            Xc = []
            for j in range(self.M):
                Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), self.ocp.x_dim)
                Xc.append(Xkj)
                self.w.append(Xkj)
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                if x_ig is None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k*self.M+j+1])

            # Loop over collocation points
            Xk_end = b[0]*Xk
            for j in range(1,self.M+1):
               # Expression for the state derivative at the collocation point
               xp = a[0,j]*Xk
               for r in range(self.M):
                   xp += a[r+1,j]*Xc[r]

               # Append collocation equations
               if u_polynomial:
                   fj, qj = fL_fcn(Xc[j-1], Uc[j-1])
               else:
                   fj, qj = fL_fcn(Xc[j-1], Uk)
               self.g.append(self.h_N*fj - xp)
               self.lbg.append(np.zeros(self.ocp.x_dim))
               self.ubg.append(np.zeros(self.ocp.x_dim))

               # Add contribution to the end state
               Xk_end += b[j]*Xc[j-1]

               # Add contribution to quadrature function
               self.J = self.J + d[j]*qj*self.h_N

            # New NLP variable for state at end of interval
            if k == self.N-1:
                Xk = ca.MX(self.ocp.x_f)

            else:
                Xk = ca.MX.sym('X_' + str(k+1), self.ocp.x_dim)
                self.w.append(Xk)
                self.lbw.append(self.ocp.x_min)
                self.ubw.append(self.ocp.x_max)
                if x_ig is None:
                    self.w0.append(np.zeros(self.ocp.x_dim))
                else:
                    self.w0.append(self.x_ig[:,k*(self.M+1)])
            # Add equality constraint
            self.g.append(Xk_end - Xk)
            self.lbg.append(np.zeros(self.ocp.x_dim))
            self.ubg.append(np.zeros(self.ocp.x_dim))

            if u_polynomial and u_continuous:
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

        # Create an NLP solver
        prob = {'f': self.J, 'x': self.w, 'g': self.g}
        solver = ca.nlpsol('solver', 'ipopt', prob, self.ipopt_opts)

        # Solve the NLP
        sol = solver(x0=np.concatenate(self.w0), lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                     lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg))
        w_opt = sol['x'].full().flatten()
        self.w_opt = w_opt

        if u_polynomial:
            u_opt_N = np.zeros((self.ocp.u_dim,0))
            for i in range(self.N):
                u_opt_N = np.append(u_opt_N, w_opt[(self.M+1)*self.ocp.x_dim*i + self.M*self.ocp.u_dim*i:(self.M+1)*self.ocp.x_dim*i + self.M*self.ocp.u_dim*(i+1)])
            u_opt_N = np.reshape(u_opt_N, (self.ocp.u_dim,self.N*self.M))

            t = sp.symbols('t')
            self.u_opt = np.zeros((self.ocp.u_dim,0))
            for i in range(self.N):
                u_t_i = 0
                for j in range(self.M):
                    u_t_i += Pu[j]((t - self.tgrid_N[i])/self.h_N)*u_opt_N[:,i*self.M+j]
                u_t_i_fcn = st.expr_to_func(t, u_t_i)
                self.u_opt = np.append(self.u_opt, [u_t_i_fcn(element) for element in self.tgrid_sim[int(i*round(self.h_N/self.h_sim)):int((i+1)*round(self.h_N/self.h_sim))]])
            self.u_opt = np.reshape(self.u_opt, (self.ocp.u_dim,len(self.tgrid_sim)-1))

            self.get_trajectory()
            self.plot_trajectory()

        else:
            u_opt_N = np.zeros((self.ocp.u_dim,0))
            for i in range(self.N):
                u_opt_N = np.append(u_opt_N, np.reshape(w_opt[(self.M+1)*self.ocp.x_dim*i + self.ocp.u_dim*i:(self.M+1)*self.ocp.x_dim*i + self.ocp.u_dim*(i+1)], (self.ocp.u_dim,1)), axis=1)
            u_opt_N = np.reshape(u_opt_N, (self.ocp.u_dim,self.N))

            self.u_opt = interp1d(self.tgrid_N[0:-1], u_opt_N, bounds_error=False, fill_value=u_opt_N[:,-1], kind='zero')(self.tgrid_sim[0:-1])

            self.get_trajectory()
            self.plot_trajectory(True)



if __name__ is '__main__':


    problem_names = {'1': 'double_int', '2': 'simple_pend', '3': 'simple_pend_cart_pl', '4': 'simple_pend_cart',
                     '5': 'pend_cart_pl', '6': 'pend_cart', '7': 'simple_dual_pend_cart_pl', '8': 'simple_dual_pend_cart',
                     '9': 'dual_pend_cart_pl', '10': 'dual_pend_cart', '11': 'vtol', '12': 'ua_manipulator_pl', '13': 'acrobot_pl',
                     '14': 'double_pend_cart_pl', '15': 'triple_pend_cart_pl'}

    problem = problem_names['1']
    ocp1 = ocp(problem)

    h_sim = 0.001
    N = 100
    M = 4

    tgrid_sim = np.linspace(ocp1.t_0, ocp1.t_f, round((ocp1.t_f - ocp1.t_0)/h_sim)+1)
    tgrid_N = np.linspace(ocp1.t_0, ocp1.t_f, N+1)

#    dssm_solver = ocp_dssm_solver(ocp1)
#    dssm_solver.solve(h_sim, N)

#    dmsm_solver = ocp_dmsm_solver(ocp1)
#    dmsm_solver.solve(h_sim, N)

    dcm_solver = ocp_dcm_solver(ocp1)
    dcm_solver.solve(h_sim, N, M, False, False)

#    issm_solver = ocp_issm_solver(ocp1)
#    issm_solver.solve(h_sim)

#    imsm_solver = ocp_imsm_solver(ocp1)
#    x_opt = dcm_solver.x_opt
#    x_ig = interp1d(tgrid_sim, x_opt, kind='zero')(tgrid_N)
#    y_ig = np.zeros(x_ig.shape)
#    imsm_solver.solve(h_sim, N, x_ig, y_ig)
#    imsm_solver.solve(h_sim, N)

#    icm_solver = ocp_icm_solver(ocp1)
#    icm_solver.solve(h_sim, N, M)