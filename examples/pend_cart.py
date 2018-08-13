# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:28:22 2018

@author: Patrick
"""

# based on http://nbviewer.jupyter.org/github/cknoll/beispiele/blob/master/zweifachpendel_nq2_np2_ruled_manif.ipynb

import sympy as sp
import symbtools as st
import symbtools.modeltools as mt
from joblib import dump


Np = 1
Nq = 1
n = Np + Nq
pp = st.symb_vector("p1:{0}".format(Np+1))
qq = st.symb_vector("q1:{0}".format(Nq+1))

ttheta = st.row_stack(pp, qq)
tthetad = st.time_deriv(ttheta, ttheta)
tthetadd = st.time_deriv(ttheta, ttheta, order=2)
st.make_global(ttheta, tthetad, tthetadd)

params = sp.symbols('l1, s1, m1, m0, g')
st.make_global(params)

tau1 = sp.Symbol("tau1")

#Einheitsvektoren

ex = sp.Matrix([1,0])
ey = sp.Matrix([0,1])

# Koordinaten der Schwerpunkte und Gelenke
S0 = ex*q1 # Schwerpunkt Wagen
G0 = S0 # Gelenk zwischen Wagen und Pendel 1
# Schwerpunkte des Pendels (Pendel zeigt für kleine Winkel nach oben)
S1 = G0 + mt.Rz(p1)*ey*s1

# Zeitableitungen der Schwerpunktskoordinaten
Sd0, Sd1 = st.col_split(st.time_deriv(st.col_stack(S0, S1), ttheta))

# Energie
T_rot = 0
T_trans = (m0*Sd0.T*Sd0 + m1*Sd1.T*Sd1)/2

T = T_rot + T_trans[0]

V = m1*g*S1[1]

mod = mt.generate_symbolic_model(T, V, ttheta, [0, tau1])

# Zustandsraummodell
mod.calc_state_eq(simplify=True)
x_dot = mod.f + mod.g*tau1

# Zustandsraummodell, partiell linearisiert
mod.calc_coll_part_lin_state_eq(simplify=True)
x_dot_pl = mod.ff + mod.gg*qddot1

# Zustandsdefinition anpassen und ZRM speichern
replacements = {'Matrix': 'sp.Matrix',
                'sin': 'sp.sin',
                'cos': 'sp.cos',
                'q1': 'x1',
                'qdot1': 'x2',
                'qddot1': 'u1',
                'p1': 'x3',
                'pdot1': 'x4',
                'tau1': 'u1'}

def str_replace_all(string, replacements):
    for (key, val) in replacements.items():
        string = string.replace(key, val)
    return string

x_dot = sp.Matrix([x_dot[1], x_dot[3], x_dot[0], x_dot[2]])
x_dot_str = str_replace_all(str(x_dot), replacements)
dump({'x_dot_str': x_dot_str}, 'examples/pend_cart.str')

x_dot_pl = sp.Matrix([x_dot_pl[1], x_dot_pl[3], x_dot_pl[0], x_dot_pl[2]])
x_dot_pl_str = str_replace_all(str(x_dot_pl), replacements)
dump({'x_dot_str': x_dot_pl_str}, 'examples/pend_cart_pl.str')