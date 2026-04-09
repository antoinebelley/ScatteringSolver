from solve_scattering import solve_scattering_problem, solve_scattering_problem_double_exp#, find_optimized_C
import amplitudes as amp
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import PchipInterpolator,Akima1DInterpolator
# from scipy.integrate import quad

ga= amp.ga
hc = amp.hc
fpi = amp.fpi

p_in = 25
p_out = 30
tz_in = 1
tz_out = -1
RegulatorPower = 3


# Nmesh = 500
npmesh = 500


C2AS_momentum = []
C2AS_coord = []

Al_momentum = []
Al_coord = []


Lambda = [2,2.5,3,3.5,4,5,6,7,8,9,10]
# Cs = [-1.050249651647136, -0.717996521251586, -0.552641685752669,-0.452423997399695]
# for i in range(len(Lambda)):
for i in range(11):
  print(f"Lambda = {Lambda[i]}", flush=True)
  print(f"--------------------\n")
  RegulatorLambda = Lambda[i]
  # Nmesh = int(2*npmesh+500*RegulatorLambda)
  Nmesh = 2*npmesh+1100
  # Nmesh = 2*npmesh+1
  pmax = 20
  # # # C= Cs[i]
  pbra, wfbra, deltabra,C = solve_scattering_problem(p_out, tz_out, RegulatorLambda, RegulatorPower, pmax=pmax, Nmesh=Nmesh, npmesh = npmesh, ur_norm=False, norm_bound=[500,1000])
  pket, wfket, deltaket,_ = solve_scattering_problem(p_in, tz_in, RegulatorLambda, RegulatorPower,C=C,  pmax=pmax, Nmesh=Nmesh, npmesh = npmesh, ur_norm=False, norm_bound=[500,1000])
  # pbra, wfbra, deltabra,C = solve_scattering_problem_double_exp(p_out, tz_out, RegulatorLambda, RegulatorPower, pmax=pmax, level1=8, level2=8, level3=12, ur_norm=True, norm_bound=[500,1000])
  # pket, wfket, deltaket,_ = solve_scattering_problem_double_exp(p_in, tz_in, RegulatorLambda, RegulatorPower,C=C,  pmax=pmax,level1=8, level2=8, level3=12, ur_norm=True, norm_bound=[500,1000])
  wfbra = wfbra
  wfket = wfket
  #Compute amplitude directly in momentum space
  F = np.zeros((pbra.size, pket.size))
  amp.calc_fermi_matrix(pbra*hc, pket*hc, F)
  fermi = (wfket@F)@wfbra

  gt = 3*ga*ga*fermi

  T = np.zeros((pbra.size, pket.size))
  amp.calc_t_matrix(pbra*hc, pket*hc, T)
  t = (wfbra@T)@wfket


  CT = np.zeros((pbra.size, pket.size))
  CT = amp.calc_contact_matrix(pbra*hc, pket*hc, RegulatorLambda, RegulatorPower, CT)
  ct = (wfbra@CT)@wfket

  # print("VALUES IN MOMENTUM SPACE")
  # print("========================")
  # print(f'F  = {fermi:.5e} MeV-2')
  # print(f'GT = {gt:.5e} MeV-2')
  # print(f'T  = {t:.5e} MeV-2')
  # print(f'CT = {ct:.5e} MeV-2')
  C2As = ct*C*C
  C2AS_momentum.append(C2As)
  total = (fermi+gt+t)
  Al_momentum.append(total)
  # print(f'Along = {total:.5e} MeV-2')
  # print(f'C^2*As = {C2As:.5e} MeV-2')
  # print(f'gvv = {(0.0195+total)/(2*ct)}')
  # print(f'c1tilde = {-4*np.pi*(0.0195+total)/(2*C2As)}\n')

  # #Transform the wf to coordinate space and compute the amplitudes in coordinate space
  # fermi = amp.ampfr(pket, pbra, wfket, wfbra, deltabra, deltaket, RegulatorLambda, RegulatorPower, rcrit = 1000)
  # gt = 3*ga*ga*fermi
  # t = amp.amptr(pket, pbra, wfket, wfbra, RegulatorLambda, RegulatorPower, RegulatorLambda, RegulatorPower,rcrit=15)
  # # #This part is extremely slow and cannot be jitted. 
  # ct = amp.ampshortr(pket, pbra, wfket, wfbra, deltabra, deltaket, RegulatorLambda, RegulatorPower)

  # print("VALUES IN COORDINATE SPACE")
  # print("==========================")
  # print(f'F  = {fermi:.5e} MeV-2')
  # print(f'GT = {gt:.5e} MeV-2')
  # print(f'T  = {t:.5e} MeV-2')
  # print(f'CT = {ct:.5e} MeV-2')
  # C2As = ct*C*C
  # C2AS_coord.append(C2As)
  # total = fermi+gt+t
  # Al_coord.append(total)
  # print(f'Along = {total:.5e} MeV-2')
  # print(f'C^2*As = {C2As:.5e} MeV-2')
  # print(f'c1tilde = {-4*np.pi*(0.0195+total)/(2*C2As)}\n\n\n')
  # plt.plot(pket,wfket, marker='.')
  # plt.show()
  with np.printoptions(threshold=np.inf):
    print(C2AS_momentum,flush=True)
    print(Al_momentum,flush=True)
    print(np.array(Al_momentum)/np.array(C2AS_momentum),flush=True)


