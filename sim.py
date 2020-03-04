import numpy as np
from pylab import *
import scipy.signal as sps

ion()

#
class PoissProc ():
  def __init__ (self, seed, rate):
    self.seed = seed
    self.rate = rate
    self.prng = np.random.RandomState(self.seed) # sets seeds for random num generator
  # based on cdf for exp wait time distribution from unif [0, 1)
  # returns in ms based on rate in Hz
  def twait (self, rate):
    return -1000. * np.log(1. - self.prng.rand()) / rate
  # new external pois designation
  def eventtimes (self, start, stop, gaptime=0.0):
    leventt = []
    if self.rate > 0.:
      t_gen = start - self.rate * 2 # start before start to remove artifact of lower event rate at start of period
      while t_gen < stop:
        t_gen += self.twait(self.rate) + gaptime # move forward by the wait time plus any gap time
        if t_gen >= start and t_gen < stop: # make sure event time is within the specified interval
          # vals are guaranteed to be monotonically increasing, no need to sort
          leventt.append(t_gen)
    return np.array(leventt)

# get a triangle
def gettriang (sampr, tdur):
  t = np.linspace(0,tdur,int(sampr*tdur/1e3))
  y = list((sps.sawtooth(2 * np.pi * t / tdur, width=1)+1)/2)
  y.reverse()
  return t,np.array(y)

#
def placetriang (sampr, sigdur, triangdur, eventt, seed=0, noiseamp=0.0, usevoss=False):
  sz = int(sampr*sigdur/1e3)
  if usevoss: # pink noise
    sig = voss(sz,amp=noiseamp,seed=seed)
  else: # white noise
    np.random.seed(seed)      
    sig = noiseamp*np.random.randn(sz)  
  ttriang, ytriang = gettriang(sampr, triangdur)
  sztriang = int(sampr*triangdur/1e3)
  for t in eventt:
    tdx = int(sampr*t/1e3)
    print(sig[tdx:tdx+sztriang].shape, tdx,tdx+sztriang, sztriang,ytriang.shape)
    sig0 = sig[tdx:tdx+sztriang]
    sig[tdx:tdx+sztriang] += ytriang[0:len(sig0)]
  return np.linspace(0,sigdur,sz),sig

#
def voss (outsz, amp=1.0, nsrc=16, seed=0):
  """Generates pink noise using the Voss-McCartney algorithm.
  outsz: number of values to generate
  nsrc: number of random sources to add
  returns: NumPy array
  """
  import pandas as pd
  np.random.seed(seed)  
  array = np.empty((outsz, nsrc))
  array.fill(np.nan)
  array[0, :] = np.random.random(nsrc)
  array[:, 0] = np.random.random(outsz)    
  # the total number of changes is outsz
  n = outsz
  cols = np.random.geometric(0.5, n)
  cols[cols >= nsrc] = 0
  rows = np.random.randint(outsz, size=n)
  array[rows, cols] = np.random.random(n)
  df = pd.DataFrame(array)
  df.fillna(method='ffill', axis=0, inplace=True)
  total = df.sum(axis=1)
  out = total.values
  out = out - mean(out) # min(out)
  mx = std(out) # max(out)
  if mx == 0.0: mx=1.0  
  out = out / mx
  out = amp * out
  return out

#
def makeburstysig (sampr, sigdur, burstfreq, burstdur, burstamp=1, noiseamp=1, seed=0, eventt=[1000,4000], smooth=True, raiseamp=0.25,\
                   usevoss=False,usegauss=False,bgsig=None):
  if bgsig is not None:
    sig = np.array([x for x in bgsig])
  elif usevoss: # pink noise
    sz = int(sampr*sigdur/1e3)
    sig = voss(sz,amp=noiseamp,seed=seed)
  else: # white noise
    sz = int(sampr*sigdur/1e3)    
    np.random.seed(seed)      
    sig = noiseamp*np.random.randn(sz)
  # Parameters for simulated signal
  burstsamp = int(burstdur*sampr) # number of samples
  # Design burst kernel
  burstkernelt = np.linspace(0,burstdur,burstsamp)
  burstkernel = burstamp * np.sin(burstkernelt*2*np.pi*burstfreq)    
  if smooth:
    if usegauss:
      wg = sps.gaussian(len(burstkernel),std=int(len(burstkernel)*0.3)) + raiseamp # raise to avoid zeroing out first/last cycles
      burstkernel = np.multiply(burstkernel,wg)
    else:
      wb = blackman(len(burstkernel)) + raiseamp # raise to avoid zeroing out first/last cycles
      burstkernel = np.multiply(burstkernel,wb) 
  print(burstkernel.shape,burstsamp)
  # Generate random signal with bursts
  times = np.linspace(0,sigdur,len(sig))
  for t in eventt:
    idx = int(sampr * t / 1e3)
    sig[idx:idx+burstsamp] += burstkernel
  return times, sig

#
def rmse (a1, a2):
  # return root mean squared error between a1, a2; assumes same lengths, sampling rates
  len1,len2 = len(a1),len(a2)
  sz = min(len1,len2)
  return np.sqrt(((a1[0:sz] - a2[0:sz]) ** 2).mean())
