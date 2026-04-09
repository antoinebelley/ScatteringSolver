# EXAMPLE ON HOW TO RUN THE CODE
# Plots radial function u(r) to compare with Fig.6 of arxiv2105.05415

import matplotlib.pyplot as plt
from solve_scattering import solve_scattering_problem, solve_scattering_problem_double_exp, get_Rp, find_optimized_C, get_Rp_de, spherical_j0
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

p_in = 25
Tz = 1
RegulatorPower = 2
pmax = 50


plt.figure(figsize=(16, 8))
plt.subplot(111)
ax = plt.gca()


for i in range(1):
  RegulatorLambda = 2
  print(RegulatorLambda,flush=True)
  npmesh = 20
  Nmesh = 2*50+100
  pmom, psi, delta,_ = solve_scattering_problem(p_in, Tz,RegulatorLambda, RegulatorPower, pmax=pmax, Nmesh=Nmesh, npmesh=npmesh, ur_norm=False, norm_bound=[500,1000])
  r = np.linspace(0,1000,100000)
  wf = get_Rp(r,pmom,psi)
  ax.plot(r,wf)
  ax.plot(r, spherical_j0(p_in/197.3*r+delta))

plt.ylim(-1,1)
plt.show()