from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *
w = 3
a = 2+3*w
# J = a**2
# #𝐽=(2+3𝑤)2
# print(f"a = {a}, J = {J}")

# a_epsilon = a + 0.001       # a epsilon 11.001
# J_epsilon = a_epsilon**2    # J_epsilon 121.02200099999999
# k = (J_epsilon - J)/0.001   # difference divided by epsilon
# print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

# sw,sJ,sa = symbols('w,J,a')
# sJ = sa**2
# sJ.subs([(sa,a)])
# dJ_da = diff(sJ, sa)
w_epsilon = w + 0.001       # a  plus a small value, epsilon
a_epsilon = 2 + 3*w_epsilon # a_epsilon 11.003
k = (a_epsilon - a)/0.001   # difference divided by epsilon
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")