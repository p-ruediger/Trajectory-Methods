# Trajectory Methods
Student project/thesis: Comparison of trajectory planning methods for dynamical systems

Implementation of direct and indirect shooting and collocation methods for trajectory optimization with [CasADi](https://web.casadi.org/) using [CVODES](https://computation.llnl.gov/projects/sundials/cvodes) and [IPOPT](https://projects.coin-or.org/Ipopt)

## [Implemented methods](ocp_solver.py)
- direct single shooting method (dssm) with piecewise constant input signal parametrization (N subintervals)
- direct multiple shooting method (dmsm) with piecewise constant input signal parametrization (N subintervals)
- direct (orthogonal) collocation method (dcm) with piecewise constant input signal parametrization (N subintervals with M collocation points)
- dcm with piecewise polynomial input signal parametrization (N subintervals with M collocation points)
- indirect single shooting method (issm)
- indirect multiple shooting method (imsm) (N subintervals)
- indirect (orthogonal) collocation method (icm) (N subintervals with M collocation points)

All collocation methods use the roots of the shifted Legendre polynomial of degree M as collocation points (orthogonal collocation).

## [Example Systems](ocp.py)
- double integrator
- pendulum
- single pendulum on cart
- dual pendulum on cart
- double pendulum on cart
- triple pendulum on cart
- vertical take-off and landing aircraft (VTOL)
- underactuated manipulator
- acrobot

## Dependencies
Python 3.6 with the packages
- NumPy
- SciPy
- SymPy
- CasADi
- matplotlib
- joblib

The packages may be installed all at once using the command
```
pip install numpy scipy sympy casadi matplotlib joblib
```
