"""
Code to solve the scattering problem with the test potential 
prensented in arxiv2105.05415 in order to fit gvv for the contact term.
Solves Lippmann-Schwinger on a momentum mesh using K-Matrix fomalism.

Code is speedup with the use of Numba (JIT compiler for python).
Unfortunately, numba is not yet compatible with scipy functions
so it cannot be used to speed up the normalization of the wavefunctions.
In theory, this could be done by calling the package numba_special but
I can't get it to build on my computer.

Author: Antoine Belley
Date: Nov. 2021

"""

import numpy as np
# from numpy.polynomial.legendre import leggauss
from scipy.special import roots_legendre
from scipy.optimize import minimize_scalar
from numba import njit
from DoubleExponentialIntegral import double_exp_p_w


#Physical constants required
ga = 1.27 # Unitless
fpi = 92.2 # MeV
hc = 197.3269804 # MeV * fm
m_pi = 138.039 # MeV
m_proton = 938.27208816 # MeV
m_neutron = 939.56542052 # MeV
m_nucleon = (m_proton+m_neutron)/2 # MeV


@njit
def spherical_j0(z):
  """
  Spherical bessel function of degree 0.

  Paramters:
    - z (float): Argument of the bessel function

  Returns:
    - j0 (float): Bessel function evaluated at z/
  """
  return np.sin(z)/z

@njit
def regulator_nonlocal(p, pp, RegulatorLambda, RegulatorPower):
  """
  Non-local regulator. 

  Parameters:
    - p (float): Mometum of incomming state in fm-1
    - pp (float): Momentum of outgoing state in fm-1
    - RegulatorLambda (float): Cutoff for regulator in fm-1
    - RegulatorPower (int): Power of regulator

  Returns:
   - Regulator (float): Regulator at momentum point p, pp
  """
  return np.exp(-np.power(p/RegulatorLambda,2*RegulatorPower)) * np.exp(-np.power(pp/RegulatorLambda,2*RegulatorPower))



@njit
def contact(p,pp,C, RegulatorLambda, RegulatorPower):
  """
  Computes the test potential V(p,pp) given in eq.18 of arxiv2105.05415.
  In contains only the contact and one pion exchange terms.
  LEC C need to be matched to reproduce the proton-neutron 
  scattering length of anp = -23.74 fm. Interaction is
  regulated through a non-local regulator.

  Paramters:
    - p (float): Momentum of incoming state in fm-1
    - pp (float): Momentum of outgoing state in fm-1
    - C (float): Dimensionless LEC of contact term
    - RegulatorLambda (float): Cutoff for regulator in MeV
    - RegulatorPower (int): Power of regulator

  Returns:
    - V (float): Potential at momentum point p,pp
  """
  prefact = 1/(2*np.pi*np.pi)*ga*ga/(4*fpi*fpi)
  contact = C
  regulator = regulator_nonlocal(p,pp,RegulatorLambda,RegulatorPower)
  return prefact*contact*regulator

@njit
def OPE(p,pp, RegulatorLambda, RegulatorPower):
  """
  Computes the test potential V(p,pp) given in eq.18 of arxiv2105.05415.
  In contains only the contact and one pion exchange terms.
  LEC C need to be matched to reproduce the proton-neutron 
  scattering length of anp = -23.74 fm. Interaction is
  regulated through a non-local regulator.

  Paramters:
    - p (float): Momentum of incoming state in fm-1
    - pp (float): Momentum of outgoing state in fm-1
    - C (float): Dimensionless LEC of contact term
    - RegulatorLambda (float): Cutoff for regulator in MeV
    - RegulatorPower (int): Power of regulator

  Returns:
    - V (float): Potential at momentum point p,pp
  """
  prefact = 1/(2*np.pi*np.pi)*ga*ga/(4*fpi*fpi)
  mpi2 = m_pi*m_pi/hc/hc
  ppp4 = 4*p*pp
  if ppp4 == 0 or mpi2/ppp4 == np.inf:
    OPE = mpi2/(mpi2+(p-pp)**2)
  else:
    OPE = (mpi2/ppp4) * np.log( 1 + (ppp4 / (mpi2 + (p-pp)**2)))
  regulator = regulator_nonlocal(p,pp,RegulatorLambda,RegulatorPower)
  return prefact*OPE*regulator


@njit
def test_potential(p,pp,C, RegulatorLambda, RegulatorPower):
  """
  Computes the test potential V(p,pp) given in eq.18 of arxiv2105.05415.
  In contains only the contact and one pion exchange terms.
  LEC C need to be matched to reproduce the proton-neutron 
  scattering length of anp = -23.74 fm. Interaction is
  regulated through a non-local regulator.

  Paramters:
    - p (float): Momentum of incoming state in fm-1
    - pp (float): Momentum of outgoing state in fm-1
    - C (float): Dimensionless LEC of contact term
    - RegulatorLambda (float): Cutoff for regulator in MeV
    - RegulatorPower (int): Power of regulator

  Returns:
    - V (float): Potential at momentum point p,pp
  """
  ct = contact(p,pp,C, RegulatorLambda, RegulatorPower)
  ope  = OPE(p,pp, RegulatorLambda, RegulatorPower)
  return ct-ope


@njit
def calc_potential_mat(psibra, psiket, q0, C, V,RegulatorLambda, RegulatorPower):
  """
  Computes the matrix containing the potential on all the mesh points.

  Parameters:
    - psibra (ndarray): Mesh points for the momentum of th bra state in fm^-1
    - psiket (ndarray): Mesh points for the momentum of th bra state in fm^-1
    - q0 (float): Incoming momentum in MeV^-1
    - C (float): Dimensionless LEC of contact term
    - V (ndarray): Array into which we want to write the potential values.
                      Must be zero array.
                      Must have dim(V) = (dim(psi_mom)+1, dim(psi_mom)+1)
    - RegulatorLambda (float): Cutoff for regulator in MeV
    - RegulatorPower (int): Power of regulator

  Returns:
    None
  """
  n = psibra.size
  m = psiket.size
  for i in range(n):
    for j in range(i,m):
      pot = test_potential(psibra[i], psiket[j], C, RegulatorLambda, RegulatorPower)
      V[i,j] = pot
      V[j,i] = pot
    V[i,-1] = test_potential(psibra[i], q0, C, RegulatorLambda, RegulatorPower)
  for j in range(n):
    V[-1,j] = test_potential(q0, psiket[j], C, RegulatorLambda, RegulatorPower)
  V[-1,-1] = test_potential(q0, q0, C, RegulatorLambda, RegulatorPower)

@njit
def propagator_integrand(q, q0):
  """
  Integrand of the propagator which takes the form
  2*mu*hc*q^2/(q0^2-q^2)

  Parameters:
    - mu (float): Reduced mass of the state in MeV
    - q (float): Momentum point in fm-1
    - q0 (float): Incoming momentum in fm^-1

  Returns:
    - Integrand (float): Integrand of the propagator in MeV^2
  """
  return q*q/(q0*q0-q*q)

@njit
def calc_propagator_mat(psi_mom, psi_weights, mu, q0, G):
  """
  Matrix containing the propagator on all the mesh points.

  Parameters:
    - psi_mom (ndarray): Mesh points for the momentum in fm^-1
    - psi_weights (ndarray): Weights of each momentum mesh points
    - mu (float): Reduced mass of the state in MeV
    - q0 (float): Incoming momentum in MeV^-1
    - G (ndarray): Array into which we want to write the propagator values.
                      Must be zero array.
                      Must have dim(G) = (dim(psi_mom)+1, dim(psi_mom)+1)

  Returns:
    None
  """
  n = psi_mom.size
  q02 = q0*q0
  for i in range(n):
    qi = psi_mom[i]
    wi = psi_weights[i]
    propi = propagator_integrand(qi,q0)
    G[i,i] = 2*mu*hc*wi*propi
    G[n,n] = G[n,n]-2*mu*hc*wi*q02/(q02-qi*qi)

def gauss_legendre(a, b, n):
  """
  Computes the n mesh points and associated weight for the gauss legendre
  quadrature in given interval.

  Parameters:
    - a (float): Lower bound of the interval
    - b (float): Upper bound of the interval
    - n (int): Number of mesh points to compute in the inteval

  Returns:
    - p (ndarray): Array containing the momentum points in the interval
    - w (ndarray): Array containing the weight associated to the momentum points
                   in p
  """
  p, w = roots_legendre(n)
  w = (b-a)/2*w
  p = (b-a)/2*p+(a+b)/2
  return p,w

@njit()
def Kmatrix(p_in, pmom,wmom, mu, C, RegulatorLambda, RegulatorPower):
  """
  Computes the K-matrix required in solving the Lippmann-Schweinger equation.

  Parameters:
    - p_in (float): Input momentum in MeV
    - pmom (ndarray): Momentum mesh in fm^-1
    - wmom (ndarray): Weight associated to the momentum mesh
    - mu (float): Reduced mass of the particles MeV
    - C (float): LECs for the contact term of the interaction. In
                 this particular form of the potential, it is unitless.
    - RegulatorLambda (float): Regulator cutoff for the interaction in fm^-1
    - RegulatorPower (int): Power of the regulator

  Returns:
    - K (ndarray): K-matrix in MeV^-2
  """
  k0 = p_in/hc
  m = pmom.size+1
  V = np.zeros((m,m))
  G = np.zeros((m,m))
  E = np.eye(m)
  calc_propagator_mat(pmom, wmom, mu, k0, G)
  calc_potential_mat(pmom, pmom, k0, C, V, RegulatorLambda, RegulatorPower)
  # print(G)
  E = E-(V@G)
  # print(E)
  K = np.linalg.inv(E)@V
  return K



@njit
def phaseshift(p_in, K11, mu):
  """
  Computes the phaseshift of the scatterinf wavefunction

  Paramters:
    - p_in (float): Input momentum in MeV
    - K (ndarray): K-matrix in MeV^-2
    - mu (float): Reduced mass of the particles in MeV

  Returns:
    - delta (float): Phaseshift of the scattering wavefunctions
  """
  delta = np.arctan(-np.pi*p_in*mu*K11)
  return delta

@njit
def scattering_length(p_in, K11, mu):
  """
  This is really -1/(k0 cot(delta(k0))) and so it gives the scattering
  lenght in the limit of k0 going to 0.

  Paramters:
    - p_in (float): Input momentum in MeV
    - K (ndarray): K-matrix in MeV^-2
    - mu (float): Reduced mass of the particles in MeV

  Returns:
    -  -1/(k0 cot(delta(k0))) (float): Scattering lenght in fm given that we
                                       take the limit of k0 goes to 0

  """
  delta = phaseshift(p_in, K11, mu)
  return - hc/p_in * np.tan(delta)

def reduced_mass(Tz):
  """
  Reduced mass of the incoming state.

  Parameters: 
    - Tz (int): Isospin of the state. Must be either -1,0,1

  Returns:
    - mu (float): Reduced mass of the incoming state in MeV
  """
  mu = 0
  if Tz == -1: mu = m_proton/2
  if Tz == 0: mu = m_proton*m_neutron/(m_proton+m_neutron)
  if Tz == 1: mu = m_neutron/2
  return mu

def find_optimized_C(RegulatorLambda, RegulatorPower, pmax = 50, Nmesh = 1000, npmesh = 50):
  """
  Find to optimal value of the contact LEC of the test potential in order
  for the test potential to reproduce the proton-neutron scattering lenght,
  i.e. a_np = -23.74 fm. 

  Paramters:
    - RegulatorLambda (float): Regulator cutoff for the interaction in fm^-1
    - RegulatorPower (int): Power of the regulator
    - pmax (float): Maximal momentum value used in fm^-1
    - Nmesh (int): Number of mesh points

  Returns:
    - C (float): LECs for the contact term of the interaction. In
                 this particular form of the potential, it is unitless.
  """
  p_in = 1e-2
  pmom, wmom, = initialize_meshpoints(p_in,pmax,RegulatorLambda,Nmesh,npmesh)
  mu = reduced_mass(0)
  def func(C):
    K = Kmatrix(p_in, pmom,wmom ,mu, C, RegulatorLambda, RegulatorPower)
    return np.abs(-23.74 - scattering_length(p_in,K[-1,-1],mu))
  res = minimize_scalar(func)
  return res.x

def initialize_meshpoints(p_in, pmax, RegulatorLambda, NMesh, npmesh = 50):
  """
  Initializes the momentum mesh and weights. The mesh is splitted so that we have
  npmesh points from 0 to p_in/hc, npmesh points from p_in/hc to 2*p_in/hc and Nmesh - 2*npmesh
  from 2*p_in/hc to pmax.

  Parameters:
    - p_in (float): Input momentum in MeV
    - pmax (float): Maximal momentum value used in fm^-1
    - Nmesh (int): Number of mesh points
    - npmesh (int): NUmber of mesh points in the region 0 to p

  Returns:
    - pmom (ndarray): Momentum mesh in fm^-1
    - wmom (ndarray): Weight associated to the momentum mesh
  """
  k0 = p_in/hc
  extra_region = 0
  if pmax > 2*RegulatorLambda: extra_region=33
  pmom = np.zeros(NMesh+extra_region)
  wmom = np.zeros(NMesh+extra_region)
  pmom[:npmesh], wmom[:npmesh] = gauss_legendre(0,k0,npmesh)
  pmom[npmesh:2*npmesh], wmom[npmesh:2*npmesh] = gauss_legendre(k0, 2*k0,npmesh)
  if pmax<2*RegulatorLambda:
    pmom[2*npmesh:], wmom[2*npmesh:] = gauss_legendre(2*k0, pmax, NMesh-2*npmesh)
  else:
     pmom[2*npmesh:-extra_region], wmom[2*npmesh:-extra_region] = gauss_legendre(2*k0, 2*RegulatorLambda, NMesh-2*npmesh)
     pmom[-extra_region:], wmom[-extra_region:] = gauss_legendre(2*RegulatorLambda,pmax, extra_region)
  return pmom, wmom

@njit
def momentum_wavefunctions(p_in, psi, pmom,  wmom, K, mu):
  """
  Computes the unormalized scattering wavefunctions in momentum space by using K-fomatlism
  and writes the values into the array psi.
  The wavefucntions at p =/= p_in/hc are given by psi(p) = 2*mu*p^2*hc*w*K(p,k0)/(k0^2-p^2)
  and psi(k0) = 1 + 2*mu*k0^2*hc*K(k0,k0)*Sum (w[i]/(k0^2-p[i^2])).

  Parameters:
    - p_in (float): Input momentum in MeV
    - psi (ndarray): Zero array of size len(pmom)+1 into which we will 
                     write the value of the wavefunction.
    - pmom (ndarray): Momentum mesh in fm^-1
    - wmom (ndarray): Weight associated to the momentum mesh

  Returns:
    None

  """
  k0 = p_in/hc
  k2 = k0*k0
  pp = 0
  for i in range(pmom.size):
    p2 = pmom[i]*pmom[i]
    psi[i] =  2*mu*p2*wmom[i]*hc/(k2-p2)*K[i,-1]
    pp +=  wmom[i]/ (k2 - p2)
  pp *= -2*mu*k2*hc*K[-1,-1]
  psi[-1] = 1 - pp



@njit()
def normalize_wavefunction(p_in, psi, pmom, delta, norm_range, ur_norm = False):
  """
  Normalize the wavefunctions so that the match the assymptotic form in
  coordinate space of j0(k0*r+delta). To do so, we fit the coordinate 
  wavefunction to the value of its assymptotic form over the value of norm_range. 

  Parameters:
    - p_in (float): Input momentum in MeV
    - psi (ndarray): Unormalized values of the wavefunctions.
    - pmom (ndarray): Momentum mesh in fm^-1
    - delta (float): Phaseshift of the scattering wavefunction
    - norm_range (ndarray): Array containning the values of r for which
                            we want to do the fit. Needs to be high enough 
                            for the wavefunctions to follow its assymptotic
                            form but small enough to have small numerical
                            errros. From 200 to 400 seems to be a reasonable 
                            range.

  Returns:
    - pmom (ndarray): Momentum mesh in fm^-1. The value at k0 as been added.
                      The momentum values are in increasing order.
    - psi (ndarray): Normalized values of the wavefunctions. The index are sorted
                     match the index of pmom.

  """
  k0 = p_in/hc
  m = pmom.size
  norm_n = norm_range.size
  prefact = np.sqrt(2/np.pi)
  num = 0
  denum = 0
  if ur_norm==True:
    for j in range(norm_n):
      wf_norm = 0
      for i in range(m):
        wf_norm += norm_range[j]*psi[i]*prefact*spherical_j0(norm_range[j]*pmom[i])
      num += np.sin(k0*norm_range[j]+delta)/(k0)*wf_norm
      denum += wf_norm*wf_norm
  else:
    for j in range(norm_n):
      wf_norm = 0
      for i in range(m):
        wf_norm += psi[i]*prefact*spherical_j0(norm_range[j]*pmom[i])
      num += spherical_j0(k0*norm_range[j]+delta)*wf_norm
      denum += wf_norm*wf_norm
  wf_norm = num/denum
  # print(wf_norm)
  psi = psi*wf_norm
  return psi

def solve_scattering_problem(p_in, Tz, RegulatorLambda, RegulatorPower, C=None, pmax=20, Nmesh=1000, norm_bound=[200,400], norm_n = 100000,npmesh=50, print_info = False, ur_norm=False):
  """
  Solves the scattering problem for the test potential and gives back the momentum wavefunctions.

  Parameters:
    - p_in (float): Input momentum in MeV
    - Tz (int): Isospin of the state. Must be either -1,0,1
    - RegulatorLambda (float): Cutoff for regulator in fm-1
    - RegulatorPower (int): Power of regulator
    - pmax (float): Maximal momentum value used in fm^-1
    - Nmesh (int): Number of mesh points
    - C (float): Value of the dimensionless contact LEC. 
    - norm_bound (ndarray): Bounds of the interval onto which we want
                            to fit to normalize the wavefunctions
    - norm_n (int): Number of points to consider during the fit
                    to normalize the wavefunctions
    - print_info (bool): Prints additional infos such as the value of C,
                         phaseshift, etc.

  """
  pmom,wmom = initialize_meshpoints(p_in,pmax,RegulatorLambda,Nmesh,npmesh)
  mu = reduced_mass(Tz)
  if C == None:
    C = find_optimized_C(RegulatorLambda,RegulatorPower, Nmesh=Nmesh, pmax=pmax, npmesh=npmesh)
  if print_info == True:
    fac = ga*ga/(4*fpi*fpi)
    print(f'C = {C}')
    print(f'Cp = {fac*C-fac} MeV^-2')
  K = Kmatrix(p_in, pmom,wmom, mu, C, RegulatorLambda, RegulatorPower)
  psi = np.zeros(pmom.size+1)
  momentum_wavefunctions(p_in,psi,pmom,wmom,K,mu)
  rnorm = np.linspace(norm_bound[0],norm_bound[1],norm_n)
  delta =  phaseshift(p_in,K[-1,-1],mu)
  if print_info == True:
    print(f'k0 = {p_in/hc}')
    print(f'delta(k0) = {delta}')
    print(f'-1/(k0 cot(delta(k0))) = {scattering_length(p_in, K[-1,-1], mu)}')
  pmom = np.insert(pmom, npmesh, p_in/hc)
  psi[npmesh:] =  np.roll(psi[npmesh:],1)
  # print(psi)
  psi = normalize_wavefunction(p_in,psi,pmom,delta,rnorm, ur_norm=ur_norm)
  return pmom, psi, delta,C


def get_Rp(r, p, wf):
  """
  Computes the coordinate wavefunction from the momentum space one.

  Paramters:
    - r (float or ndarray): Value(s) of r at which we want the wavefunction
    - p (ndarray): Momentum mesh of the momentum wavefunction
    - wf (ndarray): Values of the momentum wavefunction evaluated at the
                    momentum mesh values

  Returns:
    - f (float or ndarray): Coordinate wavefunction evaluated at the value(s)
                            of r.
  """
  try:
      f = np.sum(wf*spherical_j0(r*p))*np.sqrt(2/np.pi)
  except:
      f = r.copy()
      for i in range(len(r)):
          f[i] = np.sum(wf*spherical_j0(r[i]*p)*np.sqrt(2/np.pi))
  return f




##Functions to solve the problem using double exponential integration techinques.
#TODO: Fix comment/add references for double exponential techniques.
def initialize_meshpoints_double_exp(p_in, level1, level2, level3, RegulatorLambda, pmax):
  """
  Initializes the momentum mesh and weights. The mesh is splitted so that we have
  npmesh points from 0 to p_in/hc, npmesh points from p_in/hc to 2*p_in/hc and Nmesh - 2*npmesh
  from 2*p_in/hc to pmax.

  Parameters:
    - p_in (float): Input momentum in MeV

  Returns:
    - pmom (ndarray): Momentum mesh in fm^-1
    - wmom (ndarray): Weight associated to the momentum mesh
  """
  k0 = p_in/hc

  extra_region = 0
  if pmax>2*RegulatorLambda: extra_region=33
  pmom = np.zeros(2**level1+2**level2+2**level3+3+extra_region)
  k0pluspmom = np.zeros(2**level1+2**level2+2**level3+3+extra_region)
  k0minuspmom = np.zeros(2**level1+2**level2+2**level3+3+extra_region)
  wmom = np.zeros(2**level1+2**level2+2**level3+3+extra_region)

  h1, arg1,t1 = double_exp_p_w(0,k0,level=level1, tmax=6.112)
  pmom[:2**level1+1] =  k0/2*np.exp(arg1)/np.cosh(arg1)
  k0pluspmom[:2**level1+1] = k0/2*np.exp(arg1)/np.cosh(arg1)+k0
  k0minuspmom[:2**level1+1] = k0/2/(np.exp(arg1)*np.cosh(arg1))
  wmom[:2**level1+1] = h1*k0/2*np.pi/2*np.cosh(t1)/np.cosh(arg1)**2

  h2,arg2,t2 = double_exp_p_w(k0,pmax,level=level2, tmax=6.112)
  pmom[2**level1+1:2**level1+2**level2+2]  =  k0/2*np.exp(arg2)/np.cosh(arg2)+k0
  k0pluspmom[2**level1+1:2**level1+2**level2+2]  = k0/2*np.exp(arg2)/np.cosh(arg2)+2*k0
  k0minuspmom[2**level1+1:2**level1+2**level2+2]  = -k0/2*np.exp(arg2)/np.cosh(arg2)
  wmom[2**level1+1:2**level1+2**level2+2]  = h2*k0/2*np.pi/2*np.cosh(t2)/np.cosh(arg2)**2

  if pmax < RegulatorLambda:
    pmom[2**level1+2**level2+2:], wmom[2**level1+2**level2+2:] = gauss_legendre(2*k0,2*RegulatorLambda, 2**level3+1)
    k0pluspmom[2**level1+2**level2+2:] = k0+pmom[2**level1+2**level2+2:]
    k0minuspmom[2**level1+2**level2+2:] = k0-pmom[2**level1+2**level2+2:]

  else:
    pmom[2**level1+2**level2+2:-extra_region], wmom[2**level1+2**level2+2:-extra_region] = gauss_legendre(2*k0,2*RegulatorLambda, 2**level3+1)
    k0pluspmom[2**level1+2**level2+2:-extra_region] = k0+pmom[2**level1+2**level2+2:-extra_region]
    k0minuspmom[2**level1+2**level2+2:-extra_region] = k0-pmom[2**level1+2**level2+2:-extra_region]

    pmom[-extra_region:], wmom[-extra_region:] = gauss_legendre(2*RegulatorLambda,pmax, extra_region)
    k0pluspmom[-extra_region:] = k0+pmom[-extra_region:]
    k0minuspmom[-extra_region:] = k0-pmom[-extra_region:]

  return pmom, wmom, k0pluspmom, k0minuspmom

@njit
def propagator_integrand_double_exp(q, kpq, kmq, w):
  """
  Integrand of the propagator which takes the form
  2*mu*hc*q^2/(q0^2-q^2)

  Parameters:
    - mu (float): Reduced mass of the state in MeV
    - q (float): Momentum point in fm-1
    - q0 (float): Incoming momentum in fm^-1

  Returns:
    - Integrand (float): Integrand of the propagator in MeV^2
  """
  # qqoverkpq = q*q/kpq
  return q*q*w/kmq/kpq

@njit
def calc_propagator_mat_double_exp(psi_mom, psi_weights,k0pluspmom, k0minuspmom, mu, q0, G):
  """
  Matrix containing the propagator on all the mesh points.

  Parameters:
    - psi_mom (ndarray): Mesh points for the momentum in fm^-1
    - psi_weights (ndarray): Weights of each momentum mesh points
    - mu (float): Reduced mass of the state in MeV
    - q0 (float): Incoming momentum in MeV^-1
    - G (ndarray): Array into which we want to write the propagator values.
                      Must be zero array.
                      Must have dim(G) = (dim(psi_mom)+1, dim(psi_mom)+1)

  Returns:
    None
  """
  # total = 0
  n = psi_mom.size
  q02 = q0*q0
  for i in range(n):
    qi = psi_mom[i]
    wi = psi_weights[i]
    kpqi = k0pluspmom[i]
    kmqi = k0minuspmom[i]
    propi = propagator_integrand_double_exp(qi, kpqi, kmqi, wi)
    G[i,i] = 2*mu*hc*propi
    # print(qi,kpqi*kmqi,wi, propi)
    G[n,n] = G[n,n]-2*mu*hc*q02*wi/kmqi/kpqi
  #   total += propi
  #   if i ==32 or i==65:
  #     print(total)
  # print(total)

@njit()
def Kmatrix_double_exp(p_in, pmom,wmom,k0pluspmom, k0minuspmom, mu, C, RegulatorLambda, RegulatorPower):
  """
  Computes the K-matrix required in solving the Lippmann-Schweinger equation.

  Parameters:
    - p_in (float): Input momentum in MeV
    - pmom (ndarray): Momentum mesh in fm^-1
    - wmom (ndarray): Weight associated to the momentum mesh
    - mu (float): Reduced mass of the particles MeV
    - C (float): LECs for the contact term of the interaction. In
                 this particular form of the potential, it is unitless.
    - RegulatorLambda (float): Regulator cutoff for the interaction in fm^-1
    - RegulatorPower (int): Power of the regulator

  Returns:
    - K (ndarray): K-matrix in MeV^-2
  """
  k0 = p_in/hc
  m = pmom.size+1
  V = np.zeros((m,m))
  G = np.zeros((m,m))
  E = np.eye(m)
  calc_propagator_mat_double_exp(pmom, wmom, k0pluspmom, k0minuspmom, mu, k0, G)
  calc_potential_mat(pmom, pmom, k0, C, V, RegulatorLambda, RegulatorPower)
  # print(G)
  E = E-(V@G)
  # print(E)
  K = np.linalg.inv(E)@V
  return K

def find_optimized_C_double_exp(RegulatorLambda, RegulatorPower, level1, level2, level3, pmax):
  """
  Find to optimal value of the contact LEC of the test potential in order
  for the test potential to reproduce the proton-neutron scattering lenght,
  i.e. a_np = -23.74 fm. 

  Paramters:
    - RegulatorLambda (float): Regulator cutoff for the interaction in fm^-1
    - RegulatorPower (int): Power of the regulator
    - pmax (float): Maximal momentum value used in fm^-1
    - Nmesh (int): Number of mesh points

  Returns:
    - C (float): LECs for the contact term of the interaction. In
                 this particular form of the potential, it is unitless.
  """
  p_in = 1e-2
  pmom, wmom, k0pluspmom, k0minuspmom = initialize_meshpoints_double_exp(p_in, level1, level2, level3,RegulatorLambda,pmax)
  mu = reduced_mass(0)
  def func(C):
    K = Kmatrix_double_exp(p_in, pmom,wmom, k0pluspmom, k0minuspmom,mu, C, RegulatorLambda, RegulatorPower)
    return np.abs(-23.74 - scattering_length(p_in,K[-1,-1],mu))
  res = minimize_scalar(func)
  return res.x

@njit
def momentum_wavefunctions_double_exp(p_in,psi, pmom,  wmom, k0pluspmom, k0minuspmom, K, mu):
  """
  Computes the unormalized scattering wavefunctions in momentum space by using K-fomatlism
  and writes the values into the array psi.
  The wavefucntions at p =/= p_in/hc are given by psi(p) = 2*mu*p^2*hc*w*K(p,k0)/(k0^2-p^2)
  and psi(k0) = 1 + 2*mu*k0^2*hc*K(k0,k0)*Sum (w[i]/(k0^2-p[i^2])).

  Parameters:
    - p_in (float): Input momentum in MeV
    - psi (ndarray): Zero array of size len(pmom)+1 into which we will 
                     write the value of the wavefunction.
    - pmom (ndarray): Momentum mesh in fm^-1
    - wmom (ndarray): Weight associated to the momentum mesh

  Returns:
    None

  """
  k0 = p_in/hc
  k2 = k0*k0
  pp = 0
  for i in range(pmom.size):
    qqoverkpq = pmom[i]*pmom[i]/k0pluspmom[i]
    wmomoverkmq = wmom[i]/k0minuspmom[i]
    psi[i] =  2*mu*hc*qqoverkpq*wmomoverkmq*K[i,-1]
    pp +=  wmomoverkmq/k0pluspmom[i]
  pp *= -2*mu*k2*hc*K[-1,-1]
  psi[-1] = 1 - pp


@njit()
def normalize_wavefunction_exp(p_in, psi, pmom, delta, norm_range, k0pluspmom, k0minuspmom):
  """
  Normalize the wavefunctions so that the match the assymptotic form in
  coordinate space of j0(k0*r+delta). To do so, we fit the coordinate 
  wavefunction to the value of its assymptotic form over the value of norm_range. 

  Parameters:
    - p_in (float): Input momentum in MeV
    - psi (ndarray): Unormalized values of the wavefunctions.
    - pmom (ndarray): Momentum mesh in fm^-1
    - delta (float): Phaseshift of the scattering wavefunction
    - norm_range (ndarray): Array containning the values of r for which
                            we want to do the fit. Needs to be high enough 
                            for the wavefunctions to follow its assymptotic
                            form but small enough to have small numerical
                            errros. From 200 to 400 seems to be a reasonable 
                            range.

  Returns:
    - pmom (ndarray): Momentum mesh in fm^-1. The value at k0 as been added.
                      The momentum values are in increasing order.
    - psi (ndarray): Normalized values of the wavefunctions. The index are sorted
                     match the index of pmom.

  """
  k0 = p_in/hc
  m = pmom.size
  norm_n = norm_range.size
  prefact = np.sqrt(2/np.pi)
  num = 0
  denum = 0
  for j in range(norm_n):
    wf_norm = 0
    for i in range(m):
      wf_norm += psi[i]*prefact*spherical_j0(norm_range[j]*pmom[i])
      # wf_norm += psi[i]*prefact*(np.sin(k0pluspmom[i]*norm_range[j])-np.sin(k0minuspmom[i]*norm_range[j]))/(2*np.cos(k0*norm_range[j])*pmom[i]*norm_range[j])
    num += spherical_j0(k0*norm_range[j]+delta)*wf_norm
    # num += np.sin(k0*norm_range[j]+delta)/(k0*norm_range[j])*wf_norm
    denum += wf_norm*wf_norm
  wf_norm = num/denum
  # print(wf_norm)
  psi = psi*wf_norm
  return psi

def solve_scattering_problem_double_exp(p_in, Tz, RegulatorLambda, RegulatorPower, C=None, level1=4, level2=4, level3=4, pmax=10, norm_bound=[200,400], norm_n = 100000, print_info = False, ur_norm=False):
  """
  Solves the scattering problem for the test potential and gives back the momentum wavefunctions.

  Parameters:
    - p_in (float): Input momentum in MeV
    - Tz (int): Isospin of the state. Must be either -1,0,1
    - RegulatorLambda (float): Cutoff for regulator in fm-1
    - RegulatorPower (int): Power of regulator
    - pmax (float): Maximal momentum value used in fm^-1
    - Nmesh (int): Number of mesh points
    - C (float): Value of the dimensionless contact LEC. 
    - norm_bound (ndarray): Bounds of the interval onto which we want
                            to fit to normalize the wavefunctions
    - norm_n (int): Number of points to consider during the fit
                    to normalize the wavefunctions
    - print_info (bool): Prints additional infos such as the value of C,
                         phaseshift, etc.

  """
  pmom,wmom, k0pluspmom, k0minuspmom = initialize_meshpoints_double_exp(p_in, level1, level2, level3,RegulatorLambda, pmax)
  mu = reduced_mass(Tz)
  if C == None:
    C = find_optimized_C_double_exp(RegulatorLambda,RegulatorPower, level1, level2, level3, pmax)
  if print_info == True:
    fac = ga*ga/(4*fpi*fpi)
    print(f'C = {C}')
    print(f'Cp = {fac*C-fac} MeV^-2')
  K = Kmatrix_double_exp(p_in, pmom,wmom,k0pluspmom,k0minuspmom, mu, C, RegulatorLambda, RegulatorPower)
  psi = np.zeros(pmom.size+1)
  momentum_wavefunctions_double_exp(p_in,psi,pmom,wmom,k0pluspmom,k0minuspmom,K,mu)
  rnorm = np.linspace(norm_bound[0],norm_bound[1],norm_n)
  delta =  phaseshift(p_in,K[-1,-1],mu)
  if print_info == True:
    print(f'k0 = {p_in/hc}')
    print(f'delta(k0) = {delta}')
    print(f'-1/(k0 cot(delta(k0))) = {scattering_length(p_in, K[-1,-1], mu)}')
  pmom = np.insert(pmom, 2**level1+1, p_in/hc)
  k0pluspmom = np.insert(k0pluspmom, 2**level1+1, 2*p_in/hc)
  k0minuspmom = np.insert(k0minuspmom, 2**level1+1, 0)
  psi[2**level1+1:] =  np.roll(psi[2**level1+1:],1)
  # print(psi)
  psi = normalize_wavefunction(p_in,psi,pmom,delta,rnorm, ur_norm = False)
  return pmom, psi, delta, C

def get_Rp_de( p_in, r, p, wf, k0pluspmom, k0minuspmom):
  """
  Computes the coordinate wavefunction from the momentum space one.

  Paramters:
    - r (float or ndarray): Value(s) of r at which we want the wavefunction
    - p (ndarray): Momentum mesh of the momentum wavefunction
    - wf (ndarray): Values of the momentum wavefunction evaluated at the
                    momentum mesh values

  Returns:
    - f (float or ndarray): Coordinate wavefunction evaluated at the value(s)
                            of r.
  """
  k0 = p_in/hc
  try:
      f = np.sum(wf*(np.sin(k0pluspmom*r)-np.sin(k0minuspmom*r))/(2*np.cos(k0*r)*p*r))*np.sqrt(2/np.pi)
  except:
      f = r.copy()
      for i in range(len(r)):
          f[i] = np.sum(wf*(np.sin(k0pluspmom*r[i])-np.sin(k0minuspmom*r[i]))/(2*np.cos(k0*r[i])*p*r[i]))*np.sqrt(2/np.pi)
  return f

  