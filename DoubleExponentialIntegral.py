import numpy as np
from math import fsum
import  matplotlib.pyplot as plt
from inspect import signature

def integrate_tanh_sinh(func, a,b, max_step=2**12, tol = 1e-6, tmax=6.112):
  level = 0
  n = 1
  err = 1
  bam = (b-a)/2
  bap = (b+a)/2
  x=[bap]
  w=[np.pi/2]
  hn = [0]
  running_sum = []
  while err>tol and n<max_step:
    h = tmax/n
    for i in range(1,n+1,2):
      if not h*i in hn:
        arg = np.pi/2*np.sinh(i*h)
        hn.append(h*i)
        hn.append(-h*i)
        x.append(a+bam/(np.exp(-arg)*np.cosh(arg)))
        x.append(a+bam/(np.exp(arg)*np.cosh(-arg)))
        w.append(np.pi/2*np.cosh(i*h)/np.cosh(arg)**2)
        w.append(np.pi/2*np.cosh(-i*h)/np.cosh(-arg)**2)
    values = [w[i]*func(x[i]) for i in range(len(x))]
    running_sum.append(h*fsum(values))
    err = error_estimate(values, running_sum, tol)
    n *=2
    level +=1
  print(x)
  plt.scatter(x, values)
  plt.show()
  return bam*running_sum[-1], err, level

def integrate_tanh_sinh_factorized(func, a,b, max_step=2**12, tol = 1e-6, tmax=6.112, div="b"):
  if div == "b":
    div = b
  level = 0
  n = 1
  err = 1
  bam = (b-a)/2
  bap = (b+a)/2
  x=[bap]
  oneplusx=[div+bap]
  oneminusx=[div-bap]
  w=[np.pi/2]
  hn = [0]
  running_sum = []
  while err>tol and n<=max_step:
    h = tmax/n
    for i in range(1,n+1,2):
      if not h*i in hn:
        arg = np.pi/2*np.sinh(i*h)
        hn.append(h*i)
        hn.append(-h*i)
        x.append(a+bam/(np.exp(-arg)*np.cosh(arg)))
        x.append(a+bam/(np.exp(arg)*np.cosh(-arg)))
        oneminusx.append(div-b+bam/(np.exp(arg)*np.cosh(arg)))
        oneminusx.append(div-b+bam*np.exp(arg)/(np.cosh(-arg)))
        oneplusx.append(div+a+bam/(np.exp(-arg)*np.cosh(arg)))
        oneplusx.append(div+a+bam/(np.exp(arg)*np.cosh(-arg)))
        w.append(np.pi/2*np.cosh(i*h)/np.cosh(arg)**2)
        w.append(np.pi/2*np.cosh(-i*h)/np.cosh(-arg)**2)
    values = [w[i]*func(x[i],oneplusx[i], oneminusx[i]) for i in range(len(oneplusx))]
    running_sum.append(h*fsum(values))
    err = error_estimate(values, running_sum, tol)
    n *=2
    level +=1
  print(x)
  print(oneplusx)
  print(oneminusx)
  print(values)
  plt.scatter(hn,values)
  plt.show()
  return bam*running_sum[-1], err, level-1


def integrate_exp_sinh(func, a, max_step=2**12, tol = 1e-6, tmax = 6.112):
  level = 0
  n = 1
  err = 1000
  x = [1+a]
  w = [np.pi/2]
  hn = [0]
  running_sum = []
  while err>tol and n<=max_step:
    h = tmax/n
    for i in range(1,n+1,2):
      t = h*i
      if not t in hn:
        arg = np.pi/2*np.sinh(t)
        hn.append(t)
        hn.append(-t)
        x.append(a+np.exp(arg))
        x.append(a+np.exp(-arg))
        w.append(np.pi/2*np.cosh(t)*np.exp(arg))
        w.append(np.pi/2*np.cosh(-t)*np.exp(-arg))
    values = [w[i]*func(x[i]) for i in range(len(x))]
    running_sum.append(h*fsum(values))
    err = error_estimate(values, running_sum, tol)
    n *=2
    level +=1
  print(hn)
  print(x)
  print(values)
  plt.scatter(hn,values)
  plt.show()
  return running_sum[-1], err, level-1


def integrate_sinh_sinh(func, max_step=2**12, tol=1e-6, tmax = 6.112):
  level = 0
  n = 1
  err = 1000
  x = [0]
  w = [np.pi/2]
  hn = [0]
  running_sum = []
  while err>tol and n<=max_step:
    h = tmax/n
    for i in range(1,n+1,2):
      t = h*i 
      if not t in hn:
        hn.append(t)
        hn.append(-t)
        arg = np.pi/2*np.sinh(t)
        x.append(np.sinh(arg))
        x.append(np.sinh(-arg))
        w.append(np.pi/2*np.cosh(t)*np.cosh(arg))
        w.append(np.pi/2*np.cosh(-t)*np.cosh(-arg))
    values = [w[i]*func(x[i]) for i in range(len(x))]
    running_sum.append(h*fsum(values))
    err = error_estimate(values, running_sum, tol)
    n *=2
    level += 1
  return running_sum[-1], err, level-1


def error_estimate(values, running_sum, tol):
  if len(running_sum)<3:
    return 1
  elif running_sum[-1]==running_sum[-2] and running_sum[-1]!=np.inf:
    return 0
  else:
    d1 = np.log10(np.abs(running_sum[-1]-running_sum[-2]))
    d2 = np.log10(np.abs(running_sum[-1]-running_sum[-3]))
    d3 = np.log10(tol*np.absolute(np.array(values)).max())
    d4 = np.log10(np.array([np.absolute(values[1]), np.absolute(values[2])]).max())
    d = np.array([d1*d1/d2,2*d1,d3,d4]).max()
    return 10**d

def integrate_double_exp(func:callable, a:float, b:float,max_step=2**12, tol = 1e-6, tmax=6.112):
  if a == -np.inf and b == np.inf:
    return integrate_sinh_sinh(func, max_step=max_step, tol=tol, tmax=tmax)
  elif a == -np.inf:
    return integrate_exp_sinh(func, -b, max_step=max_step, tol=tol, tmax=tmax)
  elif b == np.inf:
    return integrate_exp_sinh(func, a, max_step=max_step, tol=tol, tmax=tmax)
  else:
    sig = signature(func)
    params = sig.parameters
    if len(params) == 3:
      return  integrate_tanh_sinh_factorized(func,a,b,max_step=max_step, tol=tol, tmax=tmax)
    elif len(params) == 1:
      return integrate_tanh_sinh(func,a,b,max_step=max_step, tol=tol, tmax=tmax)
    else:
      print("Error: Ivalid number of arguments for func.")
      exit(1)

def double_exp_p_w(a,b,level=5,tmax=6.112, shift=1):
  h = tmax/2**(level-1)
  t = np.arange(-tmax, tmax+h, h)
  arg = np.pi/2*np.sinh(t)
  # # bap = (b+a)/2
  # bam = (b-a)/2
  # if a == -np.inf and b == np.inf:
  #   p = np.sinh(arg)
  #   w = np.pi/2*np.cosh(t)*np.cosh(arg)
  # elif a == -np.inf:
  #   p = -b+np.exp(arg)
  #   w = np.pi/2*np.cosh(t)*np.exp(arg)
  # elif b == np.inf:
  #   p = a+np.exp(arg)
  #   w = np.pi/2*np.cosh(t)*np.exp(arg)
  #   oneplusp = p+shift
  #   oneminusp = p-shift
  # else:
  #   p = b/2/(np.exp(-arg)*np.cosh(arg))+a/2/(np.exp(arg)*np.cosh(arg))
  #   w = bam*np.pi/2*np.cosh(t)/np.cosh(arg)**2
  #   oneminusp = bam/(np.exp(arg)*np.cosh(arg)) 
  #   oneplusp = b +b/2*np.exp(arg)/np.cosh(arg)+a/2/(np.exp(arg)*np.cosh(arg))
  # print(oneplusp)
  return h,arg,t

def func(x):
  return 1/(1+x**2)

# print(integrate_tanh_sinh(func, -1,1))
# double_exp_p_w(-1,1, level = 4)