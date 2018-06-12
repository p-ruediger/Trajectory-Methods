# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:38:54 2017

@author: Patrick Rüdiger
"""

import os
import numpy as np
from joblib import dump, load
from matplotlib import pyplot as plt

class plot(object):

    def __init__(self, xarray):

        self.subplots = []
        self.xarray = xarray


    def add_subplot(self, yarrays, ylabel, labels, lpad, colors, linestyles, drawstyles):

        subplot = {'yarrays': yarrays, 'ylabel': ylabel, 'labels': labels, 'lpad': lpad, 'colors': colors,
                   'linestyles': linestyles, 'drawstyles': drawstyles}
        self.subplots.append(subplot)


    def append_subplot(self, yarray, label, lpad, color, linestyles, drawstyles):

        self.subplots[-1]['yarrays'].append(yarray)
        self.subplots[-1]['labels'].append(label)
        self.subplots[-1]['colors'].append(color)
        self.subplots[-1]['linestyles'].append(linestyles)
        self.subplots[-1]['drawstyles'].append(drawstyles)


    def save_plot(self, filename):

        try:
            os.mkdir('plots')
        except Exception:
            pass
        plot = {'xarray': self.xarray, 'subplots': self.subplots}
        dump(plot, 'plots/'+filename)


    def load_plotfile(self, filename):

        plot = load('plots/'+filename)
        self.xarray = plot['xarray']
        self.subplots = plot['subplots']

    def append_plotfile(self, filename):

        subplots = []
        plot = load('plots/'+filename)
        subplots = plot['subplots']
        for i in range(len(self.subplots)): subplots.append(self.subplots[i])
        self.subplots = subplots
        self.save_plot(filename)


    def create_plot(self, filename, xlabel, ymin, ymax, fig_size, font_size):

        params = {'backend': 'pdf',
                  'axes.labelsize': font_size,
                  'font.size': font_size,
                  'legend.fontsize': font_size,
                  'xtick.labelsize': font_size,
                  'ytick.labelsize': font_size,
                  'text.usetex': True,
                  'figure.figsize': fig_size}

        plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath} \usepackage[utf8]{inputenc} \usepackage[ngerman]{babel} \usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage[babel=true]{microtype}']
        plt.rcParams.update(params)

        try:
            os.mkdir('plots')
        except Exception:
            pass

        plt.figure(filename)
        axs = []
        for i in range(len(self.subplots)):
            axs.append(plt.subplot(int(str(len(self.subplots))+'1'+str(i+1))))
            for j in range(len(self.subplots[i]['yarrays'])):
                plt.plot(self.xarray, self.subplots[i]['yarrays'][j], label=self.subplots[i]['labels'][j], color=self.subplots[i]['colors'][j],
                         linestyle=self.subplots[i]['linestyles'][j], drawstyle=self.subplots[i]['drawstyles'][j])
            if self.subplots[i]['labels'][j] is not None: plt.legend(loc=0)
            plt.ylabel(self.subplots[i]['ylabel'], rotation=0, labelpad=self.subplots[i]['lpad'])
            plt.margins(x=0.02)
#            plt.margins(tight=True)
            plt.ylim(ymin=ymin, ymax=ymax)
            plt.grid(True)
            plt.setp(axs[i].get_xticklabels(), visible=(i==len(self.subplots)-1))

        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.show()
        plt.savefig('./plots/'+filename+'.pdf')
#        plt.savefig('./plots/'+filename+'.png')


class grid_plot(object):

    def create_grid_plot(self, x, y, z, cmap, z_min, z_max, xlabel, ylabel, zlabel, fig_size, font_size):

        params = {'backend': 'pdf',
          'axes.labelsize': font_size,
          'font.size': font_size,
          'legend.fontsize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'text.usetex': True,
          'figure.figsize': fig_size}

        plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath} \usepackage[utf8]{inputenc} \usepackage[ngerman]{babel} \usepackage{lmodern} \usepackage[T1]{fontenc} \usepackage[babel=true]{microtype}']
        plt.rcParams.update(params)

        fig, ax = plt.subplots()
        p = ax.pcolor(x, y, z, cmap=plt.cm.tab20c, vmin=z_min, vmax=z_max)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel, rotation=0, labelpad=30)
        cb = fig.colorbar(p)
        cb.ax.set_ylabel(zlabel, rotation=0, labelpad=30)


if __name__ == '__main__':

    p = plot()
    x = np.arange(100)
    y = np.arange(100)*2
    p.add_subplot([x, y], 'xy', ['x','y'], ['C1','C2'])
    p.add_subplot([x, y], 'xy', ['x','y'], ['C3','C4'])
    p.save_plot('xy')
    p.create_plot('xy', 't')

#    p = plot()
#    p.load_plotfile('xy')
#    x = np.arange(100)
#    y = np.arange(100)*2
#    p.append_subplot(x, 'z', 'g')
#    p.create_plot('xy', 't')
#
#    p = plot()
#    x = np.arange(100)
#    y = np.arange(100)*2
#    p.add_subplot([x, y], 'xy', ['x','y'], ['r','b'])
#    p.append_plotfile('xy')
#    p.create_plot('xy', 't')
