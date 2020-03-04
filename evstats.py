import numpy as np

# get coefficient of variation squared; (< 1 means rhythmic; 1=Poisson, > 1 for bursty)
def getCV2 (isi): return (np.std(isi)/np.mean(isi))**2

# get local variation
# < 1 for regular/rhythmic, == 1 for Poisson, > 1 for bursty
# based on Shinomoto et al 2005
def getLV (isi):
  s = 0.0
  if len(isi) < 2: return 0.0
  for i in range(len(isi)-2):
    n = (isi[i]-isi[i+1])**2
    d = (isi[i]+isi[i+1])**2
    if d > 0.:
      s += 3.0*n/d
  return s / (len(isi)-1)

# get the fano factor; llevent is list of list of events
def getFF (lcount):
  #lcount = [len(levent) for levent in llevent]
  avg = np.mean(lcount)
  if avg > 0. and len(lcount) > 1:
    return np.std(lcount)**2 / avg
  return 0.0
