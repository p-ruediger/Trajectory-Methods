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


Np = 3
Nq = 1
n = Np + Nq
pp = st.symb_vector("p1:{0}".format(Np+1))
qq = st.symb_vector("q1:{0}".format(Nq+1))

ttheta = st.row_stack(pp, qq)
tthetad = st.time_deriv(ttheta, ttheta)
tthetadd = st.time_deriv(ttheta, ttheta, order=2)
st.make_global(ttheta, tthetad, tthetadd)

params = sp.symbols('l1, l2, l3, s1, s2, s3, m1, m2, m3, J1, J2, J3, m0, g')
st.make_global(params)

tau1 = sp.Symbol("tau1")

#Einheitsvektoren

ex = sp.Matrix([1,0])
ey = sp.Matrix([0,1])

# Koordinaten der Schwerpunkte und Gelenke
S0 = ex*q1 # Schwerpunkt Wagen
G0 = S0 # Gelenk zwischen Wagen und Pendel 1
G1 = G0 + mt.Rz(p1)*ey*l1 # Gelenk zwischen Pendel 1 und 2
G2 = G1 + mt.Rz(p1+p2)*ey*l2 # Gelenk zwischen Pendel 2 und 3
# Schwerpunkte der Pendel (Pendel zeigen für kleine Winkel nach oben; Pendelwinkel relativ)
S1 = G0 + mt.Rz(p1)*ey*s1
S2 = G1 + mt.Rz(p1+p2)*ey*s2
S3 = G2 + mt.Rz(p1+p2+p3)*ey*s3

# Zeitableitungen der Schwerpunktskoordinaten
Sd0, Sd1, Sd2, Sd3 = st.col_split(st.time_deriv(st.col_stack(S0, S1, S2, S3), ttheta))

# Energie
T_rot = (J1*pdot1**2)/2 + (J2*(pdot1 + pdot2)**2)/2 + (J3*(pdot1 + pdot2 + pdot3)**2)/2
T_trans = (m0*Sd0.T*Sd0 + m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3)/2

T = T_rot + T_trans[0]

V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

mod = mt.generate_symbolic_model(T, V, ttheta, [0, 0, 0, tau1])

# Zustandsraummodell, partiell linearisiert
mod.calc_coll_part_lin_state_eq(simplify=True)
x_dot = mod.ff + mod.gg*qddot1

# Zustandsdefinition anpassen und ZRM speichern
replacements = {'Matrix': 'sp.Matrix',
                'sin': 'sp.sin',
                'cos': 'sp.cos',
                'q1': 'x1',
                'qdot1': 'x2',
                'qddot1': 'u1',
                'p1': 'x3',
                'pdot1': 'x4',
                'p2': 'x5',
                'pdot2': 'x6',
                'p3': 'x7',
                'pdot3': 'x8'}

def str_replace_all(string, replacements):
    for (key, val) in replacements.items():
        string = string.replace(key, val)
    return string

x_dot = sp.Matrix([x_dot[3], x_dot[7], x_dot[0], x_dot[4], x_dot[1], x_dot[5], x_dot[2], x_dot[6]])
x_dot_str = str_replace_all(str(x_dot), replacements)
dump({'x_dot_str': x_dot_str}, 'examples/triple_pend_cart_pl.str')