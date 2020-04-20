# Most of the code in cyc.py is based on Cole & Voytek's "Cycle-by-cycle analysis of neural oscillations"
# implementation by Sam Neymotin (samuel.neymotin@nki.rfmh.org)
import scipy.signal as sps
from evstats import getCV2
from numpy import diff
import numpy as np
from pylab import *
from collections import OrderedDict

def index2ms (idx, sampr): return 1e3*idx/sampr
def ms2index (ms, sampr): return int(sampr*ms/1e3)

#
def getmidts (sig, leftidx, rightidx):
  leftamp = sig[leftidx]
  rightamp = sig[rightidx]
  midamp = (leftamp + rightamp) / 2.0
  idx = leftidx
  if leftamp > rightamp:
    for idx in range(leftidx,rightidx+1,1):
      if sig[idx] <= midamp:
        break
  else:
    for idx in range(rightidx,leftidx,-1):
      if sig[idx] <= midamp:
        break
  return idx

#
def calcwidth (n, left, right, sampr):
  lwidth = []
  for i in range(n):
    if i in left and i in right:
      lwidth.append(index2ms(right[i] - left[i],sampr))
    else:
      lwidth.append(0) # invalid
  return lwidth

#      
def getcyclefeatures (sig, sampr, maxF, hthresh = None):
  d = ms2index(1e3/maxF,sampr) # minimum distance between peaks and troughs (in samples)
  if hthresh is None: hthresh = min(sig) # default is to find any maxima, regardless of amplitude
  peaks_positive, peak_prop = sps.find_peaks(sig, height = hthresh, threshold = None, distance=d)
  peaks_negative, trough_prop = sps.find_peaks(-sig, height = hthresh, threshold = None, distance=d)
  peakh = peak_prop['peak_heights'] # peak heights
  troughh = -trough_prop['peak_heights'] # trough heights
  peakt = [index2ms(x,sampr) for x in peaks_positive] # time of peaks
  trought = [index2ms(x,sampr) for x in peaks_negative] # time of troughs
  interpeakt = diff(peakt) # time between peaks
  intertrought = diff(trought) # time between troughs
  npk,ntrgh = len(peaks_positive),len(peaks_negative)
  decayt,decayh,decayslope = [],[],[] # decay time,height,slope
  riset,riseh,riseslope = [],[],[] # rise time,height,slope
  rdsym = [] # rise-decay symmetry
  amp = [] # amplitude of a cycle (average of two consecutive peak heights)
  midts = []
  pktrghsym = []
  i,j=0,0
  dmidpkleft,dmidpkright = {},{}
  dmidtrghleft,dmidtrghright = {},{}
  while i < npk and j < ntrgh:
    if peakt[i] < trought[j]:
      m = getmidts(sig,peaks_positive[i],peaks_negative[j])
      midts.append( m )
      dmidpkright[i] = dmidtrghleft[j] = m
      i += 1
    else:
      m = getmidts(sig,peaks_negative[j],peaks_positive[i])
      midts.append( m )
      dmidpkleft[i] = dmidtrghright[j] = m      
      j += 1
  pkw = calcwidth(npk, dmidpkleft, dmidpkright, sampr)
  trghw = calcwidth(ntrgh, dmidtrghleft, dmidtrghright, sampr)
  i,j=0,0
  while i < npk and j < ntrgh:
    if peakt[i] < trought[j]:
      decayt.append( trought[j] - peakt[i] )
      decayh.append( peakh[i] - troughh[j] )
      decayslope.append( decayh[-1] / decayt[-1] )
      if pkw[i] > 0. and trghw[j] > 0.:
        pktrghsym.append( pkw[i] / (pkw[i] + trghw[j]) )
      i += 1
    else:
      riset.append( peakt[i] - trought[j] )
      riseh.append( peakh[i] - troughh[j] )
      riseslope.append( riseh[-1] / riset[-1] )
      if len(decayt) > 0 and (riset[-1] + decayt[-1]) > 0.:
        rdsym.append(riset[-1] / (riset[-1] + decayt[-1]))
      j += 1
  peakF = [1e3/x for x in interpeakt if x > 0.]
  troughF = [1e3/x for x in intertrought if x > 0.]
  if len(peakh) > 1:
    amp = [(x+y)/2.0 for x,y in zip(peakh,peakh[1:])]
  elif len(peakh) > 0:
    amp = peakh
  return OrderedDict({'peakidx':peaks_positive,'peakh':np.array(peakh),'peakt':np.array(peakt),'interpeakt':np.array(interpeakt),\
           'troughidx':peaks_negative,'troughh':np.array(troughh),'trought':np.array(trought),'intertrought':np.array(intertrought),\
           'decayt':np.array(decayt), 'decayh':np.array(decayh), 'decayslope':np.array(decayslope),\
           'riset':np.array(riset), 'riseh':np.array(riseh), 'riseslope':np.array(riseslope),\
           'rdsym':np.array(rdsym),'peakF':peakF,'troughF':troughF,\
           'npeak':len(peakh),'ntrough':len(troughh),\
          'peakCV2':getCV2(interpeakt),'troughCV2':getCV2(intertrought),
          'amp':np.array(amp),'midts':midts,'peaktroughsym':np.array(pktrghsym),'peakw':np.array(pkw),'troughw':np.array(trghw)})

#
def getcyclekeys ():
  return ['peakidx','peakh','peakt','interpeakt',\
           'troughidx','troughh','trought','intertrought',\
           'decayt', 'decayh', 'decayslope',\
           'riset', 'riseh', 'riseslope',\
           'rdsym','peakF','troughF',\
           'npeak','ntrough',\
          'peakCV2','troughCV2',
          'amp','midts','peaktroughsym','peakw','troughw']  

#
def drawcyclefeatures (sig, sampr, maxF = None, dprop = None):
  if dprop is None:
    dprop = getcyclefeatures(sig, sampr, maxF)
  tsig = np.linspace(0,(1e3/sampr)*len(sig),len(sig))
  plot(tsig,sig,'k')
  plot([tsig[i] for i in dprop['peakidx']], [sig[i] for i in dprop['peakidx']],'ro')
  plot([tsig[i] for i in dprop['troughidx']], [sig[i] for i in dprop['troughidx']],'go')
  plot([tsig[x] for x in dprop['midts']], [sig[x] for x in dprop['midts']], 'bo')

  
