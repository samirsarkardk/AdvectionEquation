# AdvectionEquation
================================================
config.py
================================================
#
# GLOBAL CONFIGURATION
#
# PDE:
#
#       u_t + C*u_x = 0
#
# Initial condition:
#
#       u(x,0) = sin(x)
#
# Periodic boundary condition:
#
#       u(0,t) = u(2*pi,t)
#
# Domain:
#
#       0 <= x <= 2*pi
#       0 <= t <= 1
#
# Exact solution:
#
#       u(x,t) = sin(x - C*t)
#
# with C = 20.
#
# ============================================================