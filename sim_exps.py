# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:33:11 2018

@author: Patrick
"""

import numpy as np
from ocp import ocp
from ocp_solver import ocp_dssm_solver, ocp_dmsm_solver, ocp_dcm_solver, ocp_issm_solver, ocp_imsm_solver, ocp_icm_solver
from plotter import plot as plt
from joblib import dump, load, hash
import os
from scipy.interpolate import interp1d


def solve(method, prob, mparams, u_ig=None, x_ig=None):

    h_sim = 0.001

    if method is 'dssm':
        solver = ocp_dssm_solver(prob)
        solver.solve(h_sim, mparams[method]['N'], u_ig)
    elif method is 'dmsm':
        solver = ocp_dmsm_solver(prob)
        solver.solve(h_sim, mparams[method]['N'], u_ig, x_ig)
    elif method in ['dcm', 'dcm8', 'dcm9']:
        solver = ocp_dcm_solver(prob)
        solver.solve(h_sim, mparams[method]['N'], mparams[method]['M'], False, False, u_ig, x_ig)
    elif method is 'dcmc':
        h_sim = 0.00001
        solver = ocp_dcm_solver(prob)
        solver.solve(h_sim, mparams[method]['N'], mparams[method]['M'], True, True, u_ig, x_ig)
    elif method is 'issm':
        solver = ocp_issm_solver(prob)
        solver.solve(h_sim)
    elif method is 'imsm':
        solver = ocp_imsm_solver(prob)
        solver.solve(h_sim, mparams[method]['N'], x_ig)
    elif method is 'icm':
        solver = ocp_icm_solver(prob)
        solver.solve(h_sim, mparams[method]['N'], mparams[method]['M'], x_ig)

    sol = {'tgrid_sim': solver.tgrid_sim, 'x_opt': solver.x_opt, 'u_opt': solver.u_opt,
           'u_type': solver.u_type, 'stats': solver.stats, 'prob': solver.ocp, 'method': method}

    return sol


def solve_problems(exp, problems, methods, mparams, u_ig=None, x_ig=None):

    try:
        os.mkdir('exps')
    except Exception:
        pass

    hsh = hash([problems, methods, mparams], 'sha1')

    file_names = []
    for problem in problems:
        for method in methods:
            stats_file = open('exps/' + exp + '_' + hsh + '.txt', 'a')
            names_file = open('exps/' + exp + '_' + hsh + '_names.txt', 'a')
            sol = solve(method, problem, mparams, u_ig, x_ig)
            base_name = (problem.name + '_' + str(problem.has_objective) + '_' + str(problem.c) + '_'
                         + method + '_' + str(mparams[method]['N']) + '_' + str(mparams[method]['M']))
            stats_file.write(base_name + ': ' + str(sol['stats']) + '\n')
            file_names.append(base_name + '_' + hsh + '.dat')
            names_file.write(file_names[-1] + '\n')
            dump(sol, 'exps/' + file_names[-1])
            stats_file.close()
            names_file.close()

    if exp is 'exp7':
        return sol


def plot_results(sols, file_names, exp):

    font_size = 12
    fig_width = 8/2.54
    fig_width2 = 16/2.54
    fig_size = [fig_width, fig_width*1.3]
    fig_size2 = [fig_width2, fig_width2*0.4]
    fig_size3 = [fig_width, fig_width*1.3*1.4]
    fig_size4 = [fig_width, fig_width*1.3*1.8]


    labels = {'t':              r'$t\mathrm{\left/s\right.}$',
              's':              r'$s\mathrm{\left/m\right.}$',
              'v':              r'$v\mathrm{\left/\frac{m}{s}\right.}$',
#              'theta':          r'$\theta\mathrm{\left/rad\right.}$',
#              'omega':          r'$\omega\mathrm{\left/\frac{rad}{s}\right.}$',
              'theta':          r'$\theta\mathrm{\left/{}^\circ\right.}$',
              'theta1':        r'$\theta_1\mathrm{\left/{}^\circ\right.}$',
              'theta2':        r'$\theta_2\mathrm{\left/{}^\circ\right.}$',
              'theta3':        r'$\theta_3\mathrm{\left/{}^\circ\right.}$',
              'omega':          r'$\omega\mathrm{\left/\frac{{}^\circ}{s}\right.}$',
              'omega1':        r'$\omega_1\mathrm{\left/\frac{{}^\circ}{s}\right.}$',
              'omega2':        r'$\omega_2\mathrm{\left/\frac{{}^\circ}{s}\right.}$',
              'omega3':        r'$\omega_3\mathrm{\left/\frac{{}^\circ}{s}\right.}$',
              'a':              r'$a\mathrm{\left/\frac{m}{s^2}\right.}$',
#              'alpha':          r'$\alpha\mathrm{\left/\frac{rad}{s^2}\right.}$',
              'alpha':          r'$\alpha\mathrm{\left/\frac{{}^\circ}{s^2}\right.}$',
              'F':              r'$F\mathrm{\left/Nm\right.}$',
              'time':           r'$T\mathrm{\left/s\right.}$',
              'time_per_iter':  r'$T_\mathrm{Iter}\mathrm{\left/s\right.}$',
              'x_f_dev':        r'$e_\mathrm{f}$',
              'J':              r'$J$',
              'u_abs_max':      r'$a_\mathrm{max}\mathrm{\left/\frac{m}{s^2}\right.}$',
              't_f':            r'$t_\mathrm{f}\mathrm{\left/s\right.}$',
              's_1':            r'$s_1\mathrm{\left/m\right.}$'}

    lpad = 20
    lw = 1.0

    sols_method = {}
    fns = {}
    methods = []
    for i in range(len(sols)):
        methods.append(sols[i]['method'])
        sols_method.update({sols[i]['method']: sols[i]})
        fns.update({sols[i]['method']: file_names[i]})

    if exp in ['exp1', 'exp2_1', 'exp2_2', 'exp3_1', 'exp3_2', 'exp6_1', 'exp6_2']:
        for method in methods:
            sol = sols_method[method]

            plot = plt(sol['tgrid_sim'])
            for i in range(sol['prob'].x_dim):
                if sol['prob'].x_dict[str(i+1)] in ['theta', 'omega', 'theta1', 'omega1', 'theta2', 'omega2', 'theta3', 'omega3', 'alpha']:
                    plot.add_subplot([sol['x_opt'][i,:]/np.pi*180], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
                else:
                    plot.add_subplot([sol['x_opt'][i,:]], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
            plot.create_plot(fns[method] + '_' + 'x', labels['t'], None, None, lpad, fig_size, font_size)

            plot = plt(sol['tgrid_sim'][0:-1])
            for i in range(sol['prob'].u_dim):
                plot.add_subplot([sol['u_opt'][i,:]], labels[sol['prob'].u_dict[str(i+1)]], [None], ['b'], ['-'], [sol['u_type']], [lw])
            plot.create_plot(fns[method] + '_' + 'u', labels['t'], None, None, lpad, fig_size, font_size)

    if exp is 'exp1':
        plot = plt(sols['dssm']['tgrid_sim'][0:-1])
        for i in range(sols['dssm']['prob'].u_dim):
            plot.add_subplot([sols['dssm']['u_opt'][i,:], sols['issm']['u_opt'][i,:]],
                             labels[sols['dssm']['prob'].u_dict[str(i+1)]],
                             ['DESV', 'IESV'],  ['b', 'r'], ['-', '-'],
                             [sols['dssm']['u_type'], sols['issm']['u_type']], [1.5, lw])
        plot.create_plot(fns['dssm'] + '_' + 'issm' + '_' + 'u', labels['t'], None, None, lpad, fig_size2, font_size)

        plot = plt(sols['dmsm']['tgrid_sim'][0:-1])
        for i in range(sols['dmsm']['prob'].u_dim):
            plot.add_subplot([sols['dmsm']['u_opt'][i,:], sols['imsm']['u_opt'][i,:]],
                             labels[sols['dmsm']['prob'].u_dict[str(i+1)]],
                             ['DMSV, DKKV', 'IMSV, IKKV, DKKV*'], ['b', 'r'], ['-', '-'],
                             [sols['dmsm']['u_type'], sols['imsm']['u_type']], [1.5, lw])
        plot.create_plot(fns['dmsm'] + '_' + 'imsm' + '_' + 'u', labels['t'], None, None, lpad, fig_size2, font_size)

    if exp in ['exp2_1', 'exp2_2', 'exp3_1', 'exp3_2', 'exp6_1', 'exp6_2']:
        for method in methods:
            sol = sols_method[method]

            plot = plt(sol['tgrid_sim'][0:-1])
            for i in range(sol['prob'].x_dim):
                if sol['prob'].x_dict[str(i+1)] in ['theta', 'omega', 'theta1', 'omega1', 'theta2', 'omega2', 'theta3', 'omega3', 'alpha']:
                    plot.add_subplot([sol['x_opt'][i,0:-1]/np.pi*180], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
                else:
                    plot.add_subplot([sol['x_opt'][i,0:-1]], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
            for i in range(sol['prob'].u_dim):
                plot.add_subplot([sol['u_opt'][i,:]], labels[sol['prob'].u_dict[str(i+1)]], [None], ['b'], ['-'], [sol['u_type']], [lw])
            plot.create_plot(fns[method] + '_' + 'x' + '_' + 'u', labels['t'], None, None, lpad, fig_size, font_size)


    if exp in ['exp4_1', 'exp4_2', 'exp5_1', 'exp5_2', 'exp5_3']:
        stats_c = {'u_abs_max': []}
        for (key, val) in sols[0]['stats'].items():
            stats_c.update({key:[]})

        iters_solved = np.array([])
        T_solved = np.array([])
        T_iter_solved = np.array([])
        x_f_dev_solved = np.array([])
        for i in range(len(sols)):
            sol = sols[i]
            for (key, val) in sol['stats'].items():
                stats_c[key].append(val)
            stats_c['u_abs_max'].append(np.max(np.abs(sol['u_opt'])))
            if sol['stats']['success'] and sol['stats']['x_f_dev'] <= 1e-3:
                iters_solved = np.append(iters_solved, sol['stats']['iters'])
                T_solved = np.append(T_solved, sol['stats']['time'])
                T_iter_solved = np.append(T_iter_solved, sol['stats']['time_per_iter'])
                x_f_dev_solved = np.append(x_f_dev_solved, sol['stats']['x_f_dev'])

        print('iters_solved_mean: ' + str(np.mean(iters_solved)))
        print('T_solved_mean: '+ str(np.mean(T_solved)))
        print('T_iter_solved_mean: '+ str(np.mean(T_iter_solved)))
        print('x_f_dev_solved_mean: '+ str(np.mean(x_f_dev_solved)))

        if exp in ['exp4_1', 'exp4_2', 'exp5_1', 'exp5_2']:
            plot_stats = ['time', 'time_per_iter', 'x_f_dev', 'J', 'u_abs_max']
        elif exp is 'exp5_3':
            plot_stats = ['time', 'time_per_iter', 'x_f_dev', 'u_abs_max']
        if exp in ['exp4_1', 'exp4_2']:
            c = np.round(np.arange(0.2, 5.001, 0.1), 1)
        elif exp in ['exp5_1', 'exp5_2', 'exp5_3']:
            c = np.round(np.arange(0.05, 2.001, 0.05), 2)
        print('argmin(J): '+ str(c[np.argmin(stats_c['J'])]))
        print('Z: '+ str(len(T_solved)/len(c)))
        plot = plt(c)
        ymins = []
        ymaxs = []
        for stat in plot_stats:
            plot.add_subplot([stats_c[stat]], labels[stat], [None], ['b'], ['-'], ['steps-post'], [lw])
            if stat is 'J':
#                ymins.append(-100)
                ymins.append(None)
                ymaxs.append(None)
            else:
                ymins.append(None)
                ymaxs.append(None)
        if exp in ['exp4_1', 'exp4_2']:
            plot.create_plot(exp, labels['t_f'], ymins, ymaxs, lpad, fig_size, font_size)
        elif exp in ['exp5_1', 'exp5_2', 'exp5_3']:
            plot.create_plot(exp, labels['s_1'], None, ymaxs, lpad, fig_size, font_size)


    if exp in ['exp5_1', 'exp5_2', 'exp5_3']:
        plot_sols = []
        plot_cs = [0.25, 0.55]
        for i in np.array([np.where(c==plot_cs[0])[0][0], np.where(c==plot_cs[1])[0][0]]):
            plot_sols.append(sols[i])

        plot = plt(plot_sols[0]['tgrid_sim'][0:-1])
        for i in range(plot_sols[0]['prob'].x_dim):
            if plot_sols[0]['prob'].x_dict[str(i+1)] in ['theta', 'omega', 'theta1', 'omega1', 'theta2', 'omega2', 'alpha']:
                plot.add_subplot([plot_sols[0]['x_opt'][i,0:-1]/np.pi*180, plot_sols[1]['x_opt'][i,0:-1]/np.pi*180],
                                 labels[plot_sols[0]['prob'].x_dict[str(i+1)]],
                                 [None, None], ['b', 'r'], ['-', '-'],
                                 ['default', 'default'], [lw, lw])
            else:
                plot.add_subplot([plot_sols[0]['x_opt'][i,0:-1], plot_sols[1]['x_opt'][i,0:-1]],
                                 labels[plot_sols[0]['prob'].x_dict[str(i+1)]],
                                 [None, None], ['b', 'r'], ['-', '-'],
                                 ['default', 'default'], [lw, lw])
        for i in range(plot_sols[0]['prob'].u_dim):
            plot.add_subplot([plot_sols[0]['u_opt'][i,:], plot_sols[1]['u_opt'][i,:]],
                             labels[plot_sols[0]['prob'].u_dict[str(i+1)]],
                             [None, None],  ['b', 'r'], ['-', '-'],
                             [plot_sols[0]['u_type'], plot_sols[1]['u_type']], [lw, lw])
        plot.create_plot(exp + '_x_u_' + str(plot_cs[0]) + '_' + str(plot_cs[1]), labels['t'], None, None, lpad, fig_size3, font_size)


    if exp is 'exp7':
        for method in methods:
            sol = sols_method[method]

        plot = plt(sol['tgrid_sim'][0:-1])
        for i in range(2):
            if sol['prob'].x_dict[str(i+1)] in ['theta', 'omega', 'theta1', 'omega1', 'theta2', 'omega2', 'theta3', 'omega3', 'alpha']:
                plot.add_subplot([sol['x_opt'][i,0:-1]/np.pi*180], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
            else:
                plot.add_subplot([sol['x_opt'][i,0:-1]], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
        for i in range(sol['prob'].u_dim):
            plot.add_subplot([sol['u_opt'][i,:]], labels[sol['prob'].u_dict[str(i+1)]], [None], ['b'], ['-'], [sol['u_type']], [lw])
        plot.create_plot(exp + '_x_u', labels['t'], None, None, lpad, fig_size, font_size)

        ymins = [None, None, None, -500, None, None]
        ymaxs = [None, None, None, 500, None, None]
        plot = plt(sol['tgrid_sim'])
        for i in range(2,sol['prob'].x_dim):
            if sol['prob'].x_dict[str(i+1)] in ['theta', 'omega', 'theta1', 'omega1', 'theta2', 'omega2', 'theta3', 'omega3', 'alpha']:
                plot.add_subplot([sol['x_opt'][i,:]/np.pi*180], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
            else:
                plot.add_subplot([sol['x_opt'][i,:]], labels[sol['prob'].x_dict[str(i+1)]], [None], ['b'], ['-'], ['default'], [lw])
        plot.create_plot(exp + '_x', labels['t'], ymins, ymaxs, lpad, fig_size, font_size)


def run_exp(exp):

    problem_names = {'1': 'double_int', '2': 'pend', '3': 'pend_cart_pl', '4': 'pend_cart',
                     '5': 'dual_pend_cart_pl', '6': 'dual_pend_cart',
                     '7': 'vtol', '8': 'ua_manipulator_pl', '9': 'acrobot_pl',
                     '10': 'double_pend_cart_pl', '11': 'triple_pend_cart_pl'}

    dmethods = ['dssm', 'dmsm', 'dcm']
    imethods = ['issm', 'imsm', 'icm']
    methods = [*dmethods, *imethods]

    mparams = {'dssm': {'N': 100, 'M': None}, 'dmsm': {'N': 100, 'M': None}, 'dcm': {'N': 100, 'M': 4}, 'dcmc': {'N': 10, 'M': 4},
               'dcm8': {'N': 100, 'M': 8}, 'dcm9': {'N': 100, 'M': 9},
               'issm': {'N': None, 'M': None}, 'imsm': {'N': 15, 'M': None}, 'icm': {'N': 9, 'M': 9}}

    if exp in ['exp4_1', 'exp4_2']:
        mparams.update({'dmsm': {'N': 500, 'M': None}, 'dcm': {'N': 500, 'M': 4}})
    elif exp in ['exp5_1', 'exp5_2', 'exp5_3']:
        mparams.update({'dmsm': {'N': 200, 'M': None}, 'dcm': {'N': 200, 'M': 4}})


    if exp is 'exp1':
        problems = [ocp(problem_names['3'], True, None)]
        solve_problems(exp, problems, [*methods, 'dcmc'], mparams)

    elif exp is 'exp2_1':
        problems = [ocp(problem_names['3'], True, None)]
        solve_problems(exp, problems, ['dmsm', 'dcm', 'dcm8'], mparams)

    elif exp is 'exp2_2':
        problems = [ocp(problem_names['3'], False, None)]
        solve_problems(exp, problems, ['dmsm', 'dcm', 'dcm8'], mparams)

    elif exp is 'exp3_1':
        problems = [ocp(problem_names['4'], True, None)]
        solve_problems(exp, problems, ['dmsm', 'dcm', 'dcm9'], mparams)

    elif exp is 'exp3_2':
        problems = [ocp(problem_names['4'], False, None)]
        solve_problems(exp, problems, ['dmsm', 'dcm', 'dcm9'], mparams)

    elif exp in ['exp4_1', 'exp4_2']:
        problems = []
        c = np.round(np.arange(0.2, 5.001, 0.1), 1)
        for i in range(len(c)):
            problems.append(ocp(problem_names['3'], True, c[i]))
        if exp is 'exp4_1':
            solve_problems(exp, problems, ['dcm'], mparams)
        elif exp is 'exp4_2':
            solve_problems(exp, problems, ['dmsm'], mparams)

    elif exp in ['exp5_1', 'exp5_2', 'exp5_3']:
        problems = []
        has_objective = exp is not 'exp5_3'
        c = np.round(np.arange(0.05, 2.001, 0.05), 2)
        for i in range(len(c)):
            problems.append(ocp(problem_names['5'], has_objective, c[i]))
        if exp is 'exp5_1':
            solve_problems(exp, problems, ['dcm'], mparams)
        elif exp is 'exp5_2':
            solve_problems(exp, problems, ['dmsm'], mparams)
        elif exp is 'exp5_3':
            solve_problems(exp, problems, ['dcm'], mparams)

    elif exp is 'exp6_1':
        problems = [ocp(problem_names['9'], True, None)]
        solve_problems(exp, problems, ['dmsm', 'dcm'], mparams)

    elif exp is 'exp6_2':
        problems = [ocp(problem_names['9'], False, None)]
        solve_problems(exp, problems, ['dmsm', 'dcm'], mparams)

    if exp is 'exp7':
        mparams.update({'dcm': {'N': 400, 'M': 4}})
        problems = [ocp(problem_names['11'], True, None)]
        sol = solve_problems(exp, problems, ['dcm'], mparams)

#        mparams.update({'dcm': {'N': 200, 'M': 4}})
#        tgrid_NM = np.linspace(problems[0].t_0, problems[0].t_f, mparams['dcm']['N']*(mparams['dcm']['M']+1))
#        x_ig = interp1d(sol['tgrid_sim'], sol['x_opt'], kind='zero')(tgrid_NM)
#        tgrid_N = np.linspace(problems[0].t_0, problems[0].t_f, mparams['dcm']['N']+1)
#        u_ig = interp1d(sol['tgrid_sim'][0:-1], sol['u_opt'], kind='zero')(tgrid_N[0:-1])
#        solve_problems(exp, problems, ['dcm'], mparams, u_ig, x_ig)
#
#        mparams.update({'dcm': {'N': 400, 'M': 4}})
#        tgrid_NM = np.linspace(problems[0].t_0, problems[0].t_f, mparams['dcm']['N']*(mparams['dcm']['M']+1))
#        x_ig = interp1d(sol['tgrid_sim'], sol['x_opt'], kind='zero')(tgrid_NM)
#        tgrid_N = np.linspace(problems[0].t_0, problems[0].t_f, mparams['dcm']['N']+1)
#        u_ig = interp1d(sol['tgrid_sim'][0:-1], sol['u_opt'], kind='zero')(tgrid_N[0:-1])
#        solve_problems(exp, problems, ['dcm'], mparams, u_ig, x_ig)


if __name__ is '__main__':

    exp = 'exp5_3'
    mode = '2'


    if mode is '1':
        run_exp(exp)

    elif mode is '2':

        exp_name = 'exp5_3_d06574557ed8cd9a6900696e4e29fb72e5a195fb'

        file = open('exps/' + exp_name + '_names.txt', 'r')
        file_names = file.read().splitlines()
        file.close()

        sols = []
        for file_name in file_names:
            sols.append(load('exps/' + file_name))

        plot_results(sols, file_names, exp)
