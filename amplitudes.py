import numpy as np
from scipy.special import  sici, gamma
from scipy.integrate import quad
from solve_scattering import spherical_j0, regulator_nonlocal
from numba import njit
from mpmath import hyper


#Physical constants required
ga = 1.27 # Unitless
fpi = 92.2 # MeV
hc = 197.3269804 # MeV * fm
m_pi = 138.039 # MeV
m_proton = 938.27208816 # MeV
m_neutron = 939.56542052 # MeV
m_nucleon = (m_proton+m_neutron)/2 # MeV

#FUNCTIONS TO COMPUTE THE AMPLITUDE IN R SPACE
@njit
def get_Rp(r, p, wf):
  f = np.sum(wf*spherical_j0(r*p)*np.sqrt(2/np.pi))
  return f

@njit
def get_Rp_asymptotic(r, p, deltap):
  f = spherical_j0(r*p/hc+deltap)
  return f

@njit
def get_up(r, p, wf):
  if r == 0:
    f = 0 
  else:
    f = get_Rp(r,p,wf)*r
  return f

@njit
def get_up_asymptotic(r,p, deltap):
  return get_Rp_asymptotic(r,p,deltap)*r

@njit
def overlap_assymptotic_u(r, p, pp, deltap, deltapp):
  return 0.5/p/pp*(np.cos(((p-pp)*r+deltap-deltapp))-np.cos((p+pp)*r+deltap+deltapp))

@njit
def get_overlap_R(r, pket, pbra, wfket, wfbra):
  f = np.sum(wfket*spherical_j0(r*pket)*np.sqrt(2/np.pi))*np.sum(wfbra*spherical_j0(r*pbra)*np.sqrt(2/np.pi))
  return f

@njit
def get_u_overlap(r, pket, pbra, wfket, wfbra):
  f = np.sum(wfket*spherical_j0(r*pket)*np.sqrt(2/np.pi))*np.sum(wfbra*spherical_j0(r*pbra)*np.sqrt(2/np.pi)/r/r)
  return f

@njit 
def factor_fermi():
  return -1
@njit
def factor_gt():
  return -3*ga*ga
@njit
def potential_tensor_timesr(r):
  return ga*ga*(np.exp(-m_pi/hc*r)*(1+m_pi/hc*r/2))

def ampfr(pket, pbra, wfket, wfbra, deltap, deltapp,RegulatorLambda, RegulatorPower, p=25, pp=30, rcrit=400):
  # @njit
  def integrand_num(r):
    return r*get_overlap_R(r, pket, pbra, wfket, wfbra)*fourrier_reg(r,RegulatorLambda,RegulatorPower)
  def integrand_assymptotic(r):
    return overlap_assymptotic_u(r, p,pp, deltap, deltapp)/r
  assymptotic_analytic_integral = 0.5/p/pp/hc/hc*(np.pi*np.cos(deltapp)*np.sin(deltap)
                                           -np.cos(deltapp-deltap)*sici((pp-p)*rcrit)[1]
                                           +np.sin(deltapp-deltap)*sici((pp-p)*rcrit)[0]
                                           +np.cos(deltapp+deltap)*sici((pp+p)*rcrit)[1]
                                           -np.sin(deltapp+deltap)*sici((pp+p)*rcrit)[0])
  amplitude = quad(integrand_num,0,rcrit, limit=1000)
  return factor_fermi()*amplitude[0]/hc/hc + assymptotic_analytic_integral

def ampgtr(pket, pbra, wfket, wfbra, deltap, deltapp, p=25, pp=30, rcrit=200):
  return -factor_gt()*ampfr(pket, pbra, wfket, wfbra, deltap, deltapp, p, pp, rcrit)

def amptr(pket, pbra, wfket, wfbra, RegulatorLambda, RegulatorPower, rcrit=400):
  # @njit
  def integrand_num(r):
    return r*get_overlap_R(r, pket, pbra, wfket, wfbra)*potential_tensor_timesr(r)*fourrier_reg(r,RegulatorLambda,RegulatorPower)
  amplitude = quad(integrand_num,0,rcrit,limit=1000)[0]/hc/hc
  return amplitude

def fourrier_reg(r, RegulatorLambda, RegulatorPower):
  prefact = 1/(2*np.pi*np.pi)
  result = 0
  if RegulatorPower == 2:
    if r <20./RegulatorLambda:
      Larg = np.power(RegulatorLambda*r,4)/256
      result = 1/3 * RegulatorLambda**3*gamma(7/4)*hyper([],[1/2,5/4],Larg) - 1/24 * RegulatorLambda**5 * r**2 *gamma(5/4)*hyper([],[3/2,7/4],Larg)
    else:
      result = 0 
  else:
    @njit
    def integrand(q):
      return q*q*np.exp(-np.power(q/RegulatorLambda,2*RegulatorPower))*spherical_j0(q*r)
    result = quad(integrand, 0,np.inf)[0]
  result *= prefact
  # result = float(result)
  # print(f'r={r}, result = {result}')
  return result

def ampshortr(pket, pbra, wfket, wfbra, deltap, deltapp, RegulatorLambda, RegulatorPower, p=25, pp=30):
  prefact_contact = (m_nucleon*ga*ga/4/fpi/fpi)*(m_nucleon*ga*ga/4/fpi/fpi)
  def ampshort_r(r):
    f = r*r*fourrier_reg(r, RegulatorLambda, RegulatorPower)*get_Rp(r, pket, wfket)
    return f
  def ampshort_rp(r):
    f = r*r*fourrier_reg(r,RegulatorLambda, RegulatorPower)*get_Rp(r, pbra, wfbra)
    return f
  ampshort = 4*np.pi*prefact_contact*quad(ampshort_r, 0, 20/RegulatorLambda,limit=1000)[0]*quad(ampshort_rp, 0, 20/RegulatorLambda, limit=1000)[0]
  return ampshort




#FUNCTION TO COMPUTE THE AMPLITUDE IN MOMENTUM SPACE

@njit
def fermi_potential_p(p,pp):
  # if (p==0 or pp==0 or p==pp):
  #   return 0
  ppp = p*pp
  pmpp2 = (p-pp)**2
  if pmpp2 ==0 or 2/np.pi/pmpp2 == np.inf or 4*ppp/pmpp2==np.inf:
    potential = 0
  elif ppp ==  0 or 1/ppp/2/np.pi == np.inf:
    potential = -2/np.pi/pmpp2
  else:
    prefac = -1/(ppp)/2/np.pi
    potential = np.log(1+4*ppp/pmpp2)
    potential *= prefac
  # print(p,pp,potential)
  return potential

  

@njit
def calc_fermi_matrix(pbra, pket, F):
  n = pbra.size
  m = pket.size
  for i in range(n):
    for j in range(i,m):
      F[i,j] = fermi_potential_p(pbra[i], pket[j])
      F[j,i] = fermi_potential_p(pbra[j], pket[i])


@njit
def gt_potential_p(p,pp):
  return 3*ga*ga*fermi_potential_p(p,pp)

@njit
def calc_gt_matrix(pbra, pket , GT):
  n = pbra.size
  m = pket.size
  for i in range(n):
    for j in range(i,m):
      GT[i,j] = gt_potential_p(pbra[i], pket[j])
      GT[j,i] = gt_potential_p(pbra[j], pket[i])

@njit
def tensor_potential_p(p,pp):
  prefac = -ga*ga/np.pi
  mp2 = m_pi*m_pi
  term1 = -2*mp2/ ( (mp2+(p-pp)**2) * (mp2+(p+pp)**2))
  if p*pp ==0:
    term2 = -2//(mp2+(p+pp)**2)
  else:
    log = np.log(1-4*p*pp/(mp2+(p+pp)**2))
    term2 = log/(2*p*pp)
  return prefac*(term1+term2)
@njit
def calc_t_matrix(pbra, pket , T):
  n = pbra.size
  m = pket.size
  for i in range(n):
    for j in range(m):
      T[i,j] = tensor_potential_p(pbra[i], pket[j])
      T[j,i] = tensor_potential_p(pbra[j], pket[i])

@njit
def contact_potential_p(p,pp, RegulatorLambda, RegulatorPower):
  return regulator_nonlocal(p/hc,pp/hc,RegulatorLambda,RegulatorPower)
  # return 1

@njit
def calc_contact_matrix(pbra, pket, RegulatorLambda, RegulatorPower, CT):
  prefact_contact = (m_nucleon*ga*ga/4/fpi/fpi)*(m_nucleon*ga*ga/4/fpi/fpi)
  n = pbra.size
  m = pket.size
  for i in range(n):
    for j in range(i,m):
      CT[i,j] = contact_potential_p(pbra[i], pket[j],RegulatorLambda, RegulatorPower)
      CT[j,i] = contact_potential_p(pket[j], pbra[i],RegulatorLambda, RegulatorPower)
  CT = prefact_contact*CT/2/np.pi/np.pi
  return CT
