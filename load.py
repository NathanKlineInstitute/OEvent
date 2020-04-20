from pylab import *
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion  
from scipy.interpolate import interp1d
import sys,os,numpy,scipy,subprocess
#from neuron import h
from math import ceil
#h.load_file("stdrun.hoc") # creates cvode object
#h.install_sampen() # installs sample entropy (sampen) routine for h.Vector() objects
#from vector import *
#from nqs import *
from scipy.stats.stats import pearsonr,kendalltau
from filter import lowpass,bandpass
#from filt import hilblist,hilb
from multiprocessing import Pool
from modindex import *
from scipy.signal import decimate, find_peaks
#from bsmart import *
import pickle
import h5py
from morlet import MorletSpec
import matplotlib.patches as mpatches
#h.install_infot() # for transfer entropy
#h.usetable_infot = 1.0
#h.MINLOG2_infot = 0.0001
from nhpdat import *
from hecogdat import rdecog, rerefavg
from csd import *
from lc import lagged_coherence # for quantifying rhythmicity
from collections import OrderedDict
import scipy.signal as sps
from evstats import *
import gc # garbage collector
from cyc import getcyclefeatures, getcyclekeys
#from phaselock import phslock, getphaselockv
#from infot import gethist
from bbox import bbox, p2d
import pandas as pd

tl = tight_layout

rcParams['lines.markersize'] = 15
rcParams['lines.linewidth'] = 4
rcParams['font.size'] = 25

chan = 12 # pick a granular layer
noiseampLFP = 200.0 # 125.0 # amplitude cutoff for LFP noise -- not really used
noiseampCSD = 200.0 / 10.0 # amplitude cutoff for CSD noise; was 200 before units fix
noiseampBIP = 200.0 / 100.0 # amplitude cutoff for BIP noise (BIP ~10X smaller than CSD)

# get frequencies of interest + bandwidths for filtering
def getlfreqwidths (minf=0.5,maxf=125.0,step=0.5):
  lfreq=arange(minf,maxf,step); lfwidth=[]
  off=0.0
  if minf < 1.0: off = 0.5 - log2(minf) # min freq get
  lfwidth = [log2(f) + off for f in lfreq] # logarithmic + shift
  return lfreq,lfwidth

#
def getlfreq (freqmin,freqmax,getinc=False):
  dinc = {'delta':0.25,'theta':0.5,'alpha':1.0,'beta':1.0,'gamma':1.0,'hgamma':1.0}
  freqcur = freqmin
  lfreq = [freqcur]
  inc = dinc[getband(freqcur)]
  linc = [inc]
  while freqcur < freqmax:
    b = getband(freqcur)
    if b in dinc: inc = dinc[b]
    linc.append(inc)
    freqcur += inc
    lfreq.append(freqcur)
  if getinc:
    return lfreq,linc
  else:
    return lfreq

# get logarithmically spaced frequencies  
def getloglfreq (freqmin,freqmax,minstep,getstep=False):
  off = 0.0
  freqcur = freqmin
  lfreq = [freqcur]
  l2fc = log2(freqcur)
  if l2fc < 0.:
    freqstep = minstep
  else:
    freqstep = l2fc
  lfreqstep = [freqstep]
  while freqcur < freqmax:
    l2fc = log2(freqcur)
    if l2fc < 0.:
      freqstep = minstep
    else:
      freqstep = l2fc
    freqcur += freqstep
    lfreq.append(freqcur)
    lfreqstep.append(freqstep)
  if getstep:
    return lfreq,lfreqstep
  else:
    return lfreq  

def index2ms (idx, sampr): return 1e3*idx/sampr
def ms2index (ms, sampr): return int(sampr*ms/1e3)

ion() # interactive mode for pylab

# get correlation matrix between all pairs of columns
def cormat (mat):
  rv = numpy.zeros( (len(mat[0]),len(mat[0])) )
  for i in range(len(mat[0])):
    rv[i][i]=1.0
    for j in range(i+1,len(mat[0]),1):
      rv[i][j] = rv[j][i] = pearsonr(mat[:,i],mat[:,j])[0]
  return rv

# get euclidean distance
def dist (x,y):
  return numpy.sqrt(numpy.sum((x-y)**2))

# get distance matrix between all pairs of columns
def distmat (mat):
  rv = numpy.zeros( (len(mat[0]),len(mat[0])) )
  for i in range(len(mat[0])):
    rv[i][i]=1.0
    for j in range(i+1,len(mat[0]),1):
      rv[i][j] = rv[j][i] = dist(mat[:,i],mat[:,j])
  return rv

#
def plotspec (T,F,S,vc=[],newFig=False,cbar=False,ax=None):
  if len(vc) == 0: vc = [amin(S), amax(S)]
  if newFig: figure();
  if ax is None: ax=gca()
  print(amin(T),amax(T),amin(F),amax(F))
  #ax.imshow(S,extent=[amin(T),amax(T),amin(F),amax(F)],origin='lower',interpolation='None',aspect='auto',vmin=vc[0],vmax=vc[1],cmap=plt.get_cmap('jet'));
  ax.imshow(S,extent=[amin(T),amax(T),amin(F),amax(F)],origin='lower',aspect='auto',vmin=vc[0],vmax=vc[1],cmap=plt.get_cmap('jet'));
  if cbar: ax.colorbar(); 
  ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)');

#
def slicenoise (arr,F,minF=58,maxF=62):
  sidx,eidx = -1,-1
  for i in range(len(arr)):
    if F[i] >= minF and sidx == -1:
      sidx = i
    if F[i] >= maxF and eidx == -1:
      eidx = i
  return numpy.append(arr[0:sidx+1], arr[eidx+1:len(arr)])

#
def plotsigwavecut (sig,sampr,freqmax=100,thresh=1.0):
  dt = 1.0 / sampr
  ms = MorletSpec(sig,sampr,freqmin=1.0,freqmax=freqmax,freqstep=1.0) 
  ll,nl = blobcut(ms.TFR,np.mean(ms.TFR)+thresh*np.std(ms.TFR));
  subplot(3,1,1)
  ttt = linspace(0,len(sig)*dt,len(sig))
  plot(ttt,sig-mean(sig))
  xlim((ttt[0],ttt[-1]))
  subplot(3,1,2)
  imshow(ms.TFR,extent=[ttt[0], ttt[-1], ms.f[0], ms.f[-1]], aspect='auto', origin='lower',cmap=plt.get_cmap('jet'))
  xlabel('Time (s)'); ylabel('Frequency (Hz)');
  subplot(3,1,3)
  plotspec(ttt,ms.f,ll) 
  return ms,ll,nl

#
def slicenoisebycol (arr2D,F,minF=58,maxF=62):
  aout = []
  for i in range(arr2D.shape[1]):
    aout.append(slicenoise(arr2D[:,i],F,minF,maxF))
  tmp = numpy.zeros( (len(aout[0]), len(aout) ) )
  for i in range(len(aout)):
    tmp[:,i] = aout[i]
  return tmp

#
def keepF (arr,F,minF=25,maxF=55):
  sidx,eidx = -1,-1
  for i in range(len(arr)):
    if F[i] >= minF and sidx == -1:
      sidx = i
    if F[i] >= maxF and eidx == -1:
      eidx = i
  return numpy.array(arr[sidx:eidx+1])

#
def keepFbycol (arr2D,F,minF=25,maxF=55):
  aout = []
  for i in range(arr2D.shape[1]):
    aout.append(keepF(arr2D[:,i],F,minF,maxF))
  tmp = numpy.zeros( (len(aout[0]), len(aout) ) )
  for i in range(len(aout)):
    tmp[:,i] = aout[i]
  return tmp

#
def checknoise (dat,winsz,sampr,noiseamp=noiseampCSD):
  lnoise = [];
  n,sz = len(dat),len(dat)  
  for sidx in range(0,sz,winsz):
    eidx = sidx + winsz
    if eidx >= sz: eidx = sz - 1
    print(sidx,eidx)
    sig = dat[sidx:eidx]
    lnoise.append(max(abs(sig)) > noiseamp)
  return lnoise

# get morlet specgrams on windows of dat time series (window size in samples = winsz)
def getmorletwin (dat,winsz,sampr,freqmin=1.0,freqmax=100.0,freqstep=1.0,\
                  noiseamp=noiseampCSD,getphase=False,useloglfreq=False,mspecwidth=7.0):
  lms = []
  n,sz = len(dat),len(dat)
  lnoise = []; lsidx = []; leidx = []
  if useloglfreq:
    minstep=0.1
    loglfreq = getloglfreq(freqmin,freqmax,minstep)  
  for sidx in range(0,sz,winsz):
    lsidx.append(sidx)
    eidx = sidx + winsz
    if eidx >= sz: eidx = sz - 1
    leidx.append(eidx)
    print(sidx,eidx)
    sig = dat[sidx:eidx]
    lnoise.append(max(abs(sig)) > noiseamp)
    if useloglfreq:
      ms = MorletSpec(sig,sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,lfreq=loglfreq,width=mspecwidth)
    else:
      ms = MorletSpec(sig,sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,width=mspecwidth)
    lms.append(ms)
  return lms,lnoise,lsidx,leidx

# median normalization
def mednorm (dat,byRow=True):
  nrow,ncol = dat.shape[0],dat.shape[1]
  out = zeros((nrow,ncol))
  if byRow:
    for row in range(nrow):
      med = median(dat[row,:])
      if med != 0.0:
        out[row,:] = dat[row,:] / med
      else:
        out[row,:] = dat[row,:]
  else:
    for col in range(ncol):
      med = median(dat[:,col])
      if med != 0.0:
        out[:,col] = dat[:,col] / med
      else:
        out[:,col] = dat[:,col]
  return out

# maximum filter on an image
# from https://stackoverflow.com/questions/27598103/what-is-the-difference-between-imregionalmax-of-matlab-and-scipy-ndimage-filte
def maxfilt (dat,sz=3):
  lm = scipy.ndimage.filters.maximum_filter(dat,size=sz)
  msk = (dat == lm) # convert local max values to binary mask
  return msk

# simple 2D peak finding
def simple2Dpeak (dat,sz=1):
  pkx,pky=[],[]
  nrow,ncol = dat.shape[0],dat.shape[1]
  for y in range(sz,nrow-sz,1):
    if y % 100 == 0: print('.')
    for x in range(sz,ncol-sz,1):
      ispk = True
      for y0 in range(y-sz,y+sz+1,1):
        for x0 in range(x-sz,x+sz+1,1):
          if dat[y][x] < dat[y0][x0]:
            ispk = False
            break
      if ispk:
        pkx.append(x)
        pky.append(y)
  return pkx,pky
  
# downsamp - moving average downsampling
def downsamp (vec,winsz):
  sz = int(vec.size())
  i = 0
  k = 0
  vtmp = Vector(sz / winsz + 1)
  while i < sz:
    j = i + winsz - 1
    if j >= sz: j = sz - 1
    if j > i:            
      vtmp.x[k] = vec.mean(i,j)
    else:
      vtmp.x[k] = vec.x[i]
    k += 1
    i += winsz
  return vtmp

# downsamples the list of python lists using a moving average (using winsz samples)
def downsamplpy (lpy, winsz):
  vec,lout=Vector(),[]
  for py in lpy:
    vec.from_python(py)
    lout.append(downsamp(vec,winsz))
    lout[-1] = numpy.array(lout[-1].to_python())
  return lout

# cut out the individual blobs via thresholding and component labeling
def blobcut (im,thresh):
  mask = im > thresh
  labelim, nlabels = ndimage.label(mask)
  return labelim, nlabels

# binarize image (im) using a different threshold (lthresh) for each row
def blobcutlines (im,lthresh):
  lmask = []
  for row,th in enumerate(lthresh):
    mask = im[row] >= thresh
    lmask.append(mask)
  mask = np.array(lmask)
  labelim, nlabels = ndimage.label(mask)
  return labelim, nlabels

# draw a line
def drline (x0,x1,y0,y1,clr,w,ax=None):
  if ax is None: ax=gca()
  ax.plot([x0,x1],[y0,y1],clr,linewidth=w)

#
def drbox (x0,x1,y0,y1,clr,w,ax=None):
  # draw a box
  drline(x0,x0,y0,y1,clr,w,ax)
  drline(x1,x1,y0,y1,clr,w,ax)
  drline(x0,x1,y0,y0,clr,w,ax)
  drline(x0,x1,y1,y1,clr,w,ax)

# get threshold that minimizes bimodal variance
def getminbimodalvarthresh (arr,minprct=0.1,maxprct=0.9,nlevel=10,draw=False):
  minval = amin(arr)
  maxval = amax(arr)
  rng = maxval - minval
  lthresh = linspace(minprct*rng+minval,maxprct*rng+minval,nlevel)
  lstd0,lstd1,lstdA=[],[],[]
  lthreshused = []
  for thresh in lthresh:
    x0 = [x for x in arr if x < thresh]
    x1 = [x for x in arr if x >= thresh]
    if len(x0) <= 0 or len(x1) <= 0: continue
    lstd0.append(np.std(x0))
    lstd1.append(np.std(x1))
    lstdA.append((lstd0[-1]+lstd1[-1])/2.0)
    lthreshused.append(thresh)
  if draw:
    plot(lthresh,lstd0,'r');plot(lthresh,lstd0,'ro')
    plot(lthresh,lstd1,'b');plot(lthresh,lstd1,'bo')
    plot(lthresh,lstdA,'k');plot(lthresh,lstdA,'ko')
  #print(lstdA)
  return lthreshused[np.argmin(np.array(lstdA))],lthreshused,lstd0,lstd1,lstdA

# container for convenience - wavelet info from a single sample
class WaveletInfo:
  def __init__ (self,phs=0.0,idx=0,T=0.0,val=0.0):
    self.phs=phs
    self.idx=idx
    self.T=T
    self.val=val
    
#
class evblob(bbox):
  """ event blob class, inherits from bbox
  """
  NoneVal = -1e9
  def __init__ (self):
    self.avgpowevent=0 # avg val during event but only including suprathreshold pixels
    self.avgpow=self.avgpoworig=0 # avg val during event bounds
    self.cmass=evblob.NoneVal
    self.maxval=self.maxpos=self.maxvalorig=evblob.NoneVal # max spectral amplitude value during event
    self.minval=evblob.NoneVal # min val during event
    self.minvalbefore=self.maxvalbefore=self.avgpowbefore=evblob.NoneVal
    self.minvalafter=self.maxvalafter=self.avgpowafter=evblob.NoneVal
    self.MUAbefore=self.MUA=self.MUAafter=evblob.NoneVal
    self.sampen=self.sampenbefore=self.sampenafter=evblob.NoneVal
    self.arrMUAbefore=self.arrMUA=self.arrMUAafter=evblob.NoneVal # arrays (across channels) of avg MUA values before,during,after event   
    self.slicex=self.slicey=evblob.NoneVal
    self.minF=self.maxF=self.peakF=0 # min,max,peak frequencies
    self.minT=self.maxT=self.peakT=0 # min,max,peak times
    self.dur = self.Fspan = self.ncycle = self.dom = self.dombefore = self.domafter = self.Foct = 0
    self.domevent = 0 # dom during event but only including suprathreshold pixels
    # Foct is logarithmic frequency span
    self.hasbefore = self.hasafter = False # whether has before,after period    
    self.ID = -1
    self.bbox = bbox()
    self.windowidx = 0 # window index (from which spectrogram obtained)
    self.offidx = 0 # offset into time-series (since taking windows)
    self.duringnoise = 0 # during a noise window?
    # indicates whether other events of given frequency co-occur (on same channel)
    self.codelta = self.cotheta = self.coalpha = self.cobeta = self.cogamma = self.cohgamma = self.coother = 0
    self.band = evblob.NoneVal
    # correlation between CSD and MUA before,during,after event
    self.CSDMUACorrbefore = self.CSDMUACorr = self.CSDMUACorrafter = 0.0
    # a few waveform features:
    # peak and trough values close to the spectral amplitude peak, used to align signals
    self.WavePeakVal = self.WavePeakIDX = self.WaveTroughVal = self.WaveTroughIDX = 0
    self.WaveH = self.WaveW = 0 # wave height and width
    self.WavePeakT = self.WaveTroughT = 0
    self.WaveletPeak = WaveletInfo() # for alignment by wavelet peak phase (0)
    self.WaveletLeftTrough = WaveletInfo() # for alignment by wavelet trough phase (-pi)
    self.WaveletRightTrough = WaveletInfo() # for alignment by wavelet trough phase (pi)
    self.WaveletLeftH = self.WaveletLeftW = self.WaveletLeftSlope = 0 # wavelet-based height and width
    self.WaveletRightH = self.WaveletRightW = self.WaveletRightSlope = 0 # wavelet-based height and width
    self.WaveletFullW = 0 # full width right minus left trough times
    self.WaveletFullH = 0 # height offset of the right minus left trough values
    self.WaveletFullSlope = 0 # slope at full length of troughs (baseline) WaveletFullH/WaveletFullW
    # lagged coherence - for looking at rhythmicity of signal before,during,after oscillatory event
    #self.laggedCOH = self.laggedCOHbefore = self.laggedCOHafter = 0
    self.filtsig = [] # filtered signal
    self.lfiltpeak = []
    self.lfilttrough = []
    self.filtsigcor = 0.0 # correlation between filtered and raw signal
    # self.oscqual = 0.0
  def __str__ (self):
    return str(self.left)+' '+str(self.right)+' '+str(self.top)+' '+str(self.bottom)+' '+str(self.avgpow)+' '+str(self.cmass)+' '+str(self.maxval)+' '+str(self.maxpos)
  def draw (self,scalex,scaley,offidx=0,offidy=0,bbclr='white',mclr='r',linewidth=3):
    x0,x1=scalex*(self.left+offidx),scalex*(self.right+offidx)
    y0,y1=scaley*(self.top+offidy),scaley*(self.bottom+offidy)
    drline(x0,x0,y0,y1,bbclr,linewidth)
    drline(x1,x1,y0,y1,bbclr,linewidth)
    drline(x0,x1,y0,y0,bbclr,linewidth)
    drline(x0,x1,y1,y1,bbclr,linewidth)
    plot([scalex*(self.maxpos[1]+offidx)],[scaley*(self.maxpos[0]+offidy)],mclr+'o',markersize=12)

#    
class evblobset (bbox):
  def __init__ ():
    self.bbox = bbox() # bounding box
    self.IDs = set() # which IDs
  def lookup (ID):
    return ID in self.IDs
  def mergeblob (blob):
    if blob.ID not in self.IDs:
      self.IDs.append(blob.ID)
      if len(self.IDs) == 1:
        self.bbox = bbox(blob.left,blob.right,blob.bottom,blob.top)
      else:
        self.bbox = self.bbox.getunion(blob)
  def overlap (blob,prct):
    box = self.bbox.getintersection(blob)
    return box.area() >= min(self.bbox.area(),blob.area()) * prct

#
def getmergesets (lblob,prct):
  """ get the merged blobs (bounding boxes)
  lblob is a list of blos (input)
  prct is the threshold for fraction of overlap required to merge two blobs (boxes)
  returns a list of sets of merged blobs and a bool list of whether each original blob was merged
  """                                         
  sz = len(lblob)
  bmerged = [False for i in range(sz)]
  for i,blob in enumerate(lblob): blob.ID = i # make sure ID assigned
  lmergeset = [] # set of merged blobs (boxes)
  for i in range(sz):
    blob0 = lblob[i]
    for j in range(sz):
      if i == j: continue
      blob1 = lblob[j]
      if blob0.getintersection(blob1).area() >= prct * min(blob0.area(),blob1.area()): # enough overlap between bboxes?
        # merge them
        bmerged[i]=bmerged[j]=True
        found = False
        for k,mergeset in enumerate(lmergeset): # determine if either of these bboxes are in existing mergesets
          if i in mergeset or j in mergeset: # one of the bboxes in an existing mergeset?
            found = True
            if i not in mergeset: mergeset.add(i) # i not already there? add it in
            if j not in mergeset: mergeset.add(j) # j not already there? add it in
        if not found: # did not find either bbox in an existing mergeset? then create a new mergeset
          mergeset = set()
          mergeset.add(i)
          mergeset.add(j)
          lmergeset.append(mergeset)
  return lmergeset, bmerged

#
def getmergedblobs (lblob,lmergeset,bmerged):
  """ create a new list of blobs (boxes) based on lmergeset, and update the new blobs' properties
  """ 
  lblobnew = [] # list of new blobs
  for i,blob in enumerate(lblob):
    if not bmerged[i]: lblobnew.append(blob) # non-merged blobs are copied as is
  for mergeset in lmergeset: # now go through the list of mergesets and create the new blobs
    lblobtmp = [lblob[ID] for ID in mergeset]
    for i,blob in enumerate(lblobtmp):
      if i == 0:
        box = bbox(blob.left,blob.right,blob.bottom,blob.top)
        peakF = blob.peakF
        minF = blob.minF
        maxF = blob.maxF
        minT = blob.minT
        maxT = blob.maxT
        peakT = blob.peakT
        maxpos = blob.maxpos
        maxval = blob.maxval
        minval = blob.minval
      else:
        box = box.getunion(blob)
        minF = min(minF, blob.minF)
        maxF = max(maxF, blob.maxF)
        minT = min(minT, blob.minT)
        maxT = max(maxT, blob.maxT)
        if blob.maxval > maxval:
          peakF = blob.peakF
          peakT = blob.peakT
          maxpos = blob.maxpos
          maxval = blob.maxval
        if blob.minval < minval:
          minval = blob.minval
    blob.left,blob.right,blob.bottom,blob.top = box.left,box.right,box.bottom,box.top
    blob.minF,blob.maxF,blob.peakF,blob.minT,blob.maxT,blob.peakT=minF,maxF,peakF,minT,maxT,peakT
    blob.maxpos,blob.maxval = maxpos,maxval
    blob.minval = minval
    lblobnew.append(blob)
  return lblobnew

# gets blob features in original image coordinates
def getblobfeatures (imnorm,lbl,imorig=None,T=None,F=None):
  # imnorm is normalized image, lbl is label image obtained from imnorm, imorig is original un-normalized image
  # getblobfeatures returns features of blobs in lbl using imnorm
  nlabel = amax(lbl)
  lblob = [] # blob output (need better name than blob! how about object?)
  lblobidx = linspace(1,nlabel,nlabel)
  lavg = ndimage.mean(imnorm,lbl,lblobidx)
  lcmass = ndimage.center_of_mass(imnorm,lbl,lblobidx)
  lmaxval = ndimage.maximum(imnorm,lbl,lblobidx)
  lmaxpos = ndimage.maximum_position(imnorm,lbl,lblobidx)
  lavgorig = lmaxvalorig = None
  if imorig is not None:
    lavgorig = ndimage.mean(imorig,lbl,lblobidx)
    lmaxvalorig = ndimage.maximum(imorig,lbl,lblobidx)
  for blobidx in range(1,nlabel+1,1):
    msk = lbl==blobidx
    b = evblob()
    slicey, slicex = ndimage.find_objects(msk)[0]
    b.slicey = slicey
    b.slicex = slicex
    b.left = slicex.start
    b.right = slicex.stop
    b.top = slicey.stop
    b.bottom = slicey.start
    b.avgpow = lavg[blobidx-1]
    if lavgorig is not None: b.avgpoworig = lavgorig[blobidx-1]
    if lmaxvalorig is not None: b.maxvalorig = lmaxvalorig[blobidx-1]
    b.cmass = lcmass[blobidx-1]
    b.maxval = lmaxval[blobidx-1]
    b.maxpos = lmaxpos[blobidx-1]
    if F is not None:
      b.minF = F[b.bottom] # get the frequencies
      b.maxF = F[min(b.top,len(F)-1)]
      b.peakF = F[b.maxpos[0]]
    if T is not None:
      b.minT = T[b.left]
      b.maxT = T[min(b.right,len(T)-1)]
      b.peakT = T[b.maxpos[1]]
    lblob.append(b)
  return lblob

#
def getampinrange (ms, F, minF, maxF):
  sidx = 0
  while F[sidx] < minF and sidx + 1 < len(F): sidx += 1
  eidx = len(F) - 1
  while F[eidx] > maxF and sidx - 1 > 0: eidx -= 1
  print(minF,maxF,sidx,eidx,F[sidx],F[eidx])
  return sum(ms[sidx:eidx+1,:],axis=0)/(eidx-sidx+1.0)

# get event blobs in (inclusive for lower bound, strictly less than for upper bound) range of minf,maxf
def getblobinrange (lblobf, minF,maxF): return [blob for blob in lblobf if blob.peakF >= minF and blob.peakF < maxF]

# get interevent interval distribution
def getblobIEI (lblob,scalex=1.0):
  liei = []
  newlist = sorted(lblob, key=lambda x: x.left)
  for i in range(1,len(newlist),1):
    liei.append((newlist[i].left-newlist[i-1].right)*scalex)
  return liei

# get peak-to-peak time interevent interval distribution
def getpeakTIEI (dframe, levidx):
  pt = dframe['absPeakT']
  lpt = [pt[idx] for idx in levidx]
  lpt.sort()
  return [lpt[i]-lpt[i-1] for i in range(1,len(lpt),1)]

# get inter-event interval distribution based on end to start time intervals
def getinterTIEI (dframe, levidx):
  liei = []
  ptmin = dframe['absminT']
  ptmax = dframe['absmaxT']
  lpt2d = [p2d(ptmin[idx],ptmax[idx]) for idx in levidx]
  newlist = sorted(lpt2d, key=lambda x: x.x)
  for i in range(1,len(newlist),1):
    dt = newlist[i].x-newlist[i-1].y
    if dt < 0: continue # sometimes previous event finishes after next event if they're at different peak frequencies
    liei.append(dt)
  return liei

# get CV2 using variable duration windows specified in lwinsz (in seconds, entries correspond to lband)
def getvarwindCV2 (dframe, chan, \
                   lband = ['delta','theta','alpha','beta','gamma','hgamma'],\
                   lwinsz=[72.0, 32.0, 25.6, 10.7, 2.8, 1.2],FoctTH=1.5,ERPscoreTH=0.8,ERPDurTH=(75,300)):
  maxt = max(dframe['absPeakT']) / 1e3 # convert to s, lwinsz is in s
  dcv = {}  
  for b,winsz in zip(lband,lwinsz):
    print(chan,b,winsz)    
    dcv[b] = {'startt':[],'peaktieiCV2':[],'intertieiCV2':[],'peaktiei':[],'intertiei':[],'N':[],'Rate':[],\
              'peaktieiLV':[],'intertieiLV':[]}
    for startt in arange(0,maxt-winsz,winsz):
      endt = startt + winsz
      if 'ERPscore' in dframe.columns:
        dfs = dframe[(dframe.band==b) & (dframe.Foct<FoctTH) & (dframe.absPeakT>=startt*1e3) & (dframe.absPeakT<=endt*1e3) & (dframe.chan==chan) & ((dframe.ERPscore<ERPscoreTH)|(dframe.dur<ERPDurTH[0])|(dframe.dur>ERPDurTH[1]))]
      else:
        dfs = dframe[(dframe.band==b) & (dframe.Foct<FoctTH) & (dframe.absPeakT>=startt*1e3) & (dframe.absPeakT<=endt*1e3) & (dframe.chan==chan)]      
      N = len(dfs)
      dcv[b]['startt'].append(startt)
      dcv[b]['N'].append(N)
      dcv[b]['Rate'].append(float(N)/winsz)
      if N > 1:
        dcv[b]['peaktiei'].append(getpeakTIEI(dframe,dfs.index))
        dcv[b]['intertiei'].append(getinterTIEI(dframe,dfs.index))
        if N > 2:
          dcv[b]['peaktieiCV2'].append(getCV2(dcv[b]['peaktiei'][-1]))
          dcv[b]['intertieiCV2'].append(getCV2(dcv[b]['intertiei'][-1]))
          if N > 3:
            dcv[b]['peaktieiLV'].append(getLV(dcv[b]['peaktiei'][-1]))
            dcv[b]['intertieiLV'].append(getLV(dcv[b]['intertiei'][-1]))
          else:
            dcv[b]['peaktieiLV'].append(nan)
            dcv[b]['intertieiLV'].append(nan)
        else:
          dcv[b]['peaktieiCV2'].append(nan)
          dcv[b]['intertieiCV2'].append(nan)
          dcv[b]['peaktieiLV'].append(nan)
          dcv[b]['intertieiLV'].append(nan)                  
      else:
        dcv[b]['peaktiei'].append([])
        dcv[b]['intertiei'].append([])
        dcv[b]['peaktieiCV2'].append(nan)
        dcv[b]['intertieiCV2'].append(nan)
        dcv[b]['peaktieiLV'].append(nan)
        dcv[b]['intertieiLV'].append(nan)        
    dcv[b]['FF'] = getFF(dcv[b]['N'])
  return dcv

#
def getdCV2 (dframe, chan, \
             lband = ['delta','theta','alpha','beta','gamma','hgamma'], \
             lwinsz=[1,2,5,10,15,20,25,50,100,200], \
             FoctTH=1.5,ERPscoreTH=0.8,ERPDurTH=(75,300)):
  maxt = max(dframe['absPeakT']) / 1e3 # convert to s, lwinsz is in s
  dcv = {}  
  for b in lband:
    dcv[b]={}
    for winsz in lwinsz:
      print(b,winsz)
      dcv[b][winsz] = {'startt':[],'peaktieiCV2':[],'intertieiCV2':[],'peaktiei':[],'intertiei':[],'N':[],'Rate':[]}
      for startt in arange(0,maxt-winsz,winsz):
        endt = startt + winsz
        if 'ERPscore' in dframe.columns:
          dfs = dframe[(dframe.band==b) & (dframe.Foct<FoctTH) & (dframe.absPeakT>=startt*1e3) & (dframe.absPeakT<=endt*1e3) & (dframe.chan==chan) & ((dframe.ERPscore<ERPscoreTH)|(dframe.dur<ERPDurTH[0])|(dframe.dur>ERPDurTH[1]))]
        else:
          dfs = dframe[(dframe.band==b) & (dframe.Foct<FoctTH) & (dframe.absPeakT>=startt*1e3) & (dframe.absPeakT<=endt*1e3) & (dframe.chan==chan)]
        N = len(dfs)
        if N > 2:
          dcv[b][winsz]['peaktiei'].append(getpeakTIEI(dframe,dfs.index))
          dcv[b][winsz]['peaktieiCV2'].append(getCV2(dcv[b][winsz]['peaktiei'][-1]))
          dcv[b][winsz]['intertiei'].append(getinterTIEI(dframe,dfs.index))
          dcv[b][winsz]['intertieiCV2'].append(getCV2(dcv[b][winsz]['intertiei'][-1]))
        dcv[b][winsz]['startt'].append(startt)          
        dcv[b][winsz]['N'].append(N)
        dcv[b][winsz]['Rate'].append(float(N)/winsz)
  return dcv

#
def drawdCV2 (ddcv2,lchan,lwinsz,k='intertieiCV2',xl=None,yl=None,ylab=r'Average $CV^2$',lclr = ['r','g','b','c','m'], lsty='solid',marker='o'):
  for clr,chan in zip(lclr,lchan):
    dcv2 = ddcv2[chan]
    for bdx,b in enumerate(lband):
      subplot(3,2,bdx+1); title(b)
      lWinter = []; lMinter = []; 
      for winsz in lwinsz:
        m = mean(dcv2[b][winsz][k])
        if not isnan(m):
          lWinter.append(winsz)
          lMinter.append(m)
        plot(lWinter,lMinter,clr,linestyle=lsty);
        plot(lWinter,lMinter,clr+marker,markersize=25)
      title(b)
      if ylab is not None: ylabel(ylab)
      xlabel('Window size (s)')
      if yl is not None: ylim(yl)
      if xl is not None: xlim(xl)

# finds boundaries where the image dips below the threshold, starting from x,y and moving left,right,up,down
def findbounds (img,x,y,thresh):
  ysz, xsz = img.shape
  y0 = y
  x0 = x - 1
  # look left
  while True:
    if x0 < 0:
      x0 = 0
      break
    if img[y0][x0] < thresh: break
    x0 -= 1
  left = x0
  # look right
  x0 = x + 1
  while True:
    if x0 >= xsz:
      x0 = xsz - 1
      break
    if img[y0][x0] < thresh: break
    x0 += 1
  right = x0
  # look down
  x0 = x
  y0 = y - 1
  while True:
    if y0 < 0:
      y0 = 0
      break
    if img[y0][x0] < thresh: break
    y0 -= 1
  bottom = y0
  # look up
  x0 = x
  y0 = y + 1
  while True:
    if y0 >= ysz:
      y0 = ysz - 1
      break
    if img[y0][x0] < thresh: break      
    y0 += 1
  top = y0
  #print('left,right,top,bottom:',left,right,top,bottom)  
  return left,right,top,bottom

# extract the event blobs from local maxima image (impk)
def getblobsfrompeaks (imnorm,impk,imorig,medthresh,fctr=0.5,T=None,F=None):
  # imnorm is normalized image, lbl is label image obtained from imnorm, imorig is original un-normalized image
  # medthresh is median threshold for significant peaks
  # getblobfeatures returns features of blobs in lbl using imnorm
  lpky,lpkx = np.where(impk) # get the peak coordinates
  lblob = []
  for y,x in zip(lpky,lpkx):
    pkval = imnorm[y][x]
    thresh = min(medthresh, fctr * pkval) # lower value threshold used to find end of event 
    left,right,top,bottom = findbounds(imnorm,x,y,thresh)
    #subimg = imnorm[bottom:top+1,left:right+1]
    #thsubimg = subimg > thresh
    #print('L,R,T,B:',left,right,top,bottom,subimg.shape,thsubimg.shape,sum(thsubimg))
    #print('sum(thsubimg)',sum(thsubimg),'amax(subimg)',amax(subimg))    
    b = evblob()
    #b.avgpoworig = ndimage.mean(imorig[bottom:top+1,left:right+1],thsubimg,[1])
    b.maxvalorig = imorig[y][x]
    #b.avgpow = ndimage.mean(subimg,thsubimg,[1])
    b.maxval = pkval
    b.minval = amin(imnorm[bottom:top+1,left:right+1])
    b.left = left
    b.right = right
    b.top = top
    b.bottom = bottom
    b.maxpos = (y,x)
    if F is not None:
      b.minF = F[b.bottom] # get the frequencies
      b.maxF = F[min(b.top,len(F)-1)]
      b.peakF = F[b.maxpos[0]]
      b.band = getband(b.peakF)
    if T is not None:
      b.minT = T[b.left]
      b.maxT = T[min(b.right,len(T)-1)]
      b.peakT = T[b.maxpos[1]]
    lblob.append(b)
  return lblob

#
def getbandrange (lblob):
  drange = {'delta':[],'theta':[],'alpha':[],'beta':[],'gamma':[],'hgamma':[],'unknown':[]}
  for blob in lblob:
    drange[blob.band].append((blob.minT,blob.maxT))
  return drange

#
def checkcorange (drange,band,ev):
  for rng in drange[band]:
    if ev.maxT < rng[0] or ev.minT > rng[1]:
      pass
    else:
      return True
  return False

#
def getcoband (levblob):
  drange = getbandrange(levblob)
  for ev in levblob:
    if ev.band == 'delta': ev.codelta = 1
    else: ev.codelta = int(checkcorange(drange,'delta',ev))
    if ev.band == 'theta': ev.cotheta = 1
    else: ev.cotheta = int(checkcorange(drange,'theta',ev))
    if ev.band == 'alpha': ev.coalpha = 1
    else: ev.coalpha = int(checkcorange(drange,'alpha',ev))    
    if ev.band == 'beta': ev.cobeta = 1
    else: ev.cobeta = int(checkcorange(drange,'beta',ev))
    if ev.band == 'gamma': ev.cogamma = 1
    else: ev.cogamma = int(checkcorange(drange,'gamma',ev))    
    if ev.band == 'hgamma': ev.cohgamma = 1
    else: ev.cohgamma = int(checkcorange(drange,'hgamma',ev))
    if ev.band == 'delta':
      ev.coother = int(ev.cotheta or ev.coalpha or ev.cobeta or ev.cogamma or ev.cohgamma)
    if ev.band == 'theta':
      ev.coother = int(ev.codelta or ev.coalpha or ev.cobeta or ev.cogamma or ev.cohgamma)
    if ev.band == 'alpha':
      ev.coother = int(ev.codelta or ev.cotheta or ev.cobeta or ev.cogamma or ev.cohgamma)
    if ev.band == 'beta':
      ev.coother = int(ev.codelta or ev.cotheta or ev.coalpha or ev.cogamma or ev.cohgamma)
    if ev.band == 'gamma':
      ev.coother = int(ev.codelta or ev.cotheta or ev.coalpha or ev.cobeta or ev.cohgamma)
    if ev.band == 'hgamma':
      ev.coother = int(ev.codelta or ev.cotheta or ev.coalpha or ev.cobeta or ev.cogamma)                  

#      
def getFoct (minF, maxF):
  if maxF - minF > 0.0 and minF > 0.0: return log(maxF/minF)
  return 0.0

#
def getextrafeatures (lblob, ms, img, medthresh, csd, MUA, chan, offidx, sampr, fctr = 0.5, getphase = True, getfilt = True):
  # get extra features for the event blobs, including:
  # MUA before/after, min/max/avg power before/after, sampen before/during/after
  # ms is the MorletSpec object (contains non-normalized TFR and PHS when getphase==True
  # img is the median normalized spectrogram image; MUA is the multiunit activity (should have same sampling rate)
  # chan is CSD channel (where events detected), note that csd is 1D while MUA is 2D (for now)
  vec=h.Vector() # for getting the sample entropy
  mua = None
  if MUA is not None: mua = MUA[chan+1,:] # mua on same channel
  for bdx,blob in enumerate(lblob):
    # duration/frequency features
    blob.dur = blob.maxT - blob.minT # duration
    blob.Fspan = blob.maxF - blob.minF # linear frequency span
    blob.ncycle = blob.dur*blob.peakF/1e3 # number of cycles
    blob.Foct = getFoct(blob.minF,blob.maxF)
    ###
    w2 = int(blob.width() / 2.)
    left,right,bottom,top = blob.left,blob.right+1,blob.bottom,blob.top # these are indices into TFR (wavelet spectrogram)
    #print(bdx,left,right,bottom,top,offidx)
    subimg = img[bottom:top+1,left:right+1] # is right+1 correct if already inc'ed above?
    blob.avgpow = mean(subimg) # avg power of all pixels in event bounds
    thresh = min(medthresh, fctr * blob.maxval) # lower value threshold used to find end of event 
    thsubimg = subimg >= thresh  # 
    #print(bdx,fctr,blob.maxval,thresh,subimg.shape,thsubimg.shape,left,right,bottom,top)
    #print(amax(subimg),amin(thsubimg),amax(thsubimg))
    blob.avgpowevent = ndimage.mean(subimg,thsubimg,[1])[0] # avg power of suprathreshold pixels    
    if blob.avgpow>0.0: blob.dom = float(blob.maxval/blob.avgpow) # depth of modulation (all pixels)
    if blob.avgpowevent>0.0: blob.domevent = float(blob.maxval/blob.avgpowevent) # depth of modulation (suprathreshold pixels)
    if mua is not None:
      blob.MUA = mean(mua[left+offidx:right+offidx]) # offset from spectrogram index into original MUA,CSD time-series
      blob.arrMUA = mean(MUA[:,left+offidx:right+offidx],axis=1) # avg MUA from each channel during the event
    vec.from_python(csd[left+offidx:right+offidx])
    blob.sampen = vec.vsampen() # may want to test diff params/timescales of sampen
    if mua is not None: blob.CSDMUACorr = pearsonr(csd[left+offidx:right+offidx],mua[left+offidx:right+offidx])[0]
    # a few waveform features    
    wvlen2 = (1e3/blob.peakF)/2 # 1/2 wavelength in milliseconds
    wvlensz2 = int(wvlen2*sampr/1e3) # 1/2 wavelength in samples
    blob.WavePeakVal,blob.WavePeakIDX = findpeak(csd, int(blob.maxpos[1])+offidx, left+offidx, right+offidx, wvlensz2)
    blob.WaveTroughVal,blob.WaveTroughIDX = findtrough(csd, int(blob.maxpos[1])+offidx, left+offidx, right+offidx, wvlensz2)
    blob.WavePeakIDX -= offidx; blob.WaveTroughIDX -= offidx; # keep indices within spectrogram image
    blob.WaveH = blob.WavePeakVal - blob.WaveTroughVal
    blob.WavePeakT = 1e3*blob.WavePeakIDX/sampr
    blob.WaveTroughT = 1e3*blob.WaveTroughIDX/sampr
    blob.WaveW = 2 * abs(blob.WavePeakT - blob.WaveTroughT) # should update to use wavelet peak (phase= 0)/trough(phase= -PI) 
    if getphase: # wavelet-based features
      freqIDX = list(ms.f).index(blob.peakF) # index into frequency array
      PHS = ms.PHS[freqIDX,:]
      blob.WaveletPeak.phs,blob.WaveletPeak.idx=findclosest(PHS,int(blob.maxpos[1]),left,right,wvlensz2,0.0)
      blob.WaveletPeak.val = csd[blob.WaveletPeak.idx+offidx] # +offidx for correct index into csd (blob.WaveletPeak.idx is into PHS)
      blob.WaveletPeak.T = 1e3*blob.WaveletPeak.idx/sampr #
      blob.WaveletLeftTrough.phs,blob.WaveletLeftTrough.idx=findclosest(PHS,int(blob.WaveletPeak.idx),left,right,wvlensz2+int(wvlensz2/2),-pi,lookleft=True,lookright=False)
      blob.WaveletLeftTrough.val = csd[blob.WaveletLeftTrough.idx+offidx]# +offidx for correct index into csd (WaveletLeftTrough.idx is into PHS)
      blob.WaveletLeftTrough.T = 1e3*blob.WaveletLeftTrough.idx/sampr #
      blob.WaveletRightTrough.phs,blob.WaveletRightTrough.idx=findclosest(PHS,int(blob.WaveletPeak.idx),left,right,wvlensz2+int(wvlensz2/2),pi,lookleft=False,lookright=True)
      blob.WaveletRightTrough.val = csd[blob.WaveletRightTrough.idx+offidx]# +offidx for correct index into csd (WaveletRightTrough.idx is into PHS)
      blob.WaveletRightTrough.T = 1e3*blob.WaveletRightTrough.idx/sampr #      
      blob.WaveletLeftH = blob.WaveletPeak.val - blob.WaveletLeftTrough.val
      blob.WaveletLeftW = blob.WaveletPeak.T - blob.WaveletLeftTrough.T
      if blob.WaveletLeftW != 0.0: blob.WaveletLeftSlope = blob.WaveletLeftH / blob.WaveletLeftW      
      blob.WaveletRightH = blob.WaveletPeak.val - blob.WaveletRightTrough.val
      blob.WaveletRightW = blob.WaveletRightTrough.T - blob.WaveletPeak.T
      if blob.WaveletRightW != 0.0: blob.WaveletRightSlope = blob.WaveletRightH / blob.WaveletRightW
      blob.WaveletFullH = blob.WaveletRightTrough.val - blob.WaveletLeftTrough.val
      blob.WaveletFullW = blob.WaveletRightTrough.T - blob.WaveletLeftTrough.T
      if blob.WaveletFullW != 0.0: blob.WaveletFullSlope = blob.WaveletFullH / blob.WaveletFullW
    if getfilt:
      padsz = int(sampr*0.2)
      x0 = left+offidx
      x1 = right+offidx-1
      x0p = max(0,x0-padsz)
      x1p = min(len(csd),x1+padsz)
      fsig = bandpass(csd[x0p:x1p], blob.minF, blob.maxF, sampr, zerophase=True)
      #print(x0,x1,x0p,x1p,len(fsig))
      blob.filtsig = fsig[x0-x0p:x0-x0p+x1-x0]
      blob.filtsigcor = pearsonr(blob.filtsig,csd[x0:x1])[0]
    # look at values in period before event
    idx0 = max(0,blob.left - wvlensz2) #max(0,blob.left - w2)
    idx1 = blob.left
    if idx1 > idx0 + 1: # any period before?
      subimg = img[blob.bottom:blob.top+1,idx0:idx1]
      blob.minvalbefore = amin(subimg)
      blob.maxvalbefore = amax(subimg)
      blob.avgpowbefore = mean(subimg)
      if blob.avgpowbefore>0.0: blob.dombefore = float(blob.maxvalbefore/blob.avgpowbefore)
      idx0 += offidx; idx1 += offidx # offset from spectrogram index into original MUA,CSD time-series
      if mua is not None:
        blob.MUAbefore = mean(mua[idx0:idx1]) # offset from spectrogram index into original MUA,CSD time-series
        blob.arrMUAbefore = mean(MUA[:,idx0:idx1],axis=1) # avg MUA from each channel before the event
      vec.from_python(csd[idx0:idx1])
      blob.sampenbefore = vec.vsampen()
      if mua is not None:
        blob.CSDMUACorrbefore = pearsonr(csd[idx0:idx1],mua[idx0:idx1])[0]
      blob.hasbefore = True      
      #idx0 = max(0,blob.left - wvlensz2*2) 
      #idx1 = blob.left    
      #blob.laggedCOHbefore = lagged_coherence(csd[idx0:idx1], (blob.minF,blob.maxF), sampr)#, n_cycles=blob.ncycle) # quantifies rhythmicity
      #if isnan(blob.laggedCOHbefore): blob.laggedCOHbefore = -1.0
    else:
      blob.hasbefore = False
    # look at values in period after event
    idx0 = blob.right+1
    idx1 = min(idx0 + wvlensz2,img.shape[1]) # min(idx0 + w2,img.shape[1])
    if idx1 > idx0 + 1: # any period after?
      subimg = img[blob.bottom:blob.top+1,idx0:idx1]
      blob.minvalafter = amin(subimg)
      blob.maxvalafter = amax(subimg)
      blob.avgpowafter = mean(subimg)
      if blob.avgpowafter>0.0: blob.domafter = float(blob.maxvalafter/blob.avgpowafter)
      idx0 += offidx; idx1 += offidx # offset from spectrogram index into original MUA,CSD time-series
      if mua is not None:
        blob.MUAafter = mean(mua[idx0:idx1])
        blob.arrMUAafter = mean(MUA[:,idx0:idx1],axis=1) # avg MUA from each channel after the event
      vec.from_python(csd[idx0:idx1])
      blob.sampenafter = vec.vsampen()
      if mua is not None:
        blob.CSDMUACorrafter = pearsonr(csd[idx0:idx1],mua[idx0:idx1])[0]
      blob.hasafter = True      
      #idx1 = min(idx0 + wvlensz2*2,img.shape[1]) # min(idx0 + w2,img.shape[1])
      #blob.laggedCOHafter = lagged_coherence(csd[idx0:idx1], (blob.minF,blob.maxF), sampr)#, n_cycles=blob.ncycle) # quantifies rhythmicity
      #if isnan(blob.laggedCOHafter): blob.laggedCOHafter = -1.0
    else:
      blob.hasafter = False
  getcoband(lblob) # get band of events co-occuring on same channel

#  
def getcycbyband (dframe, lband, sampr):
  dbprop = {b:{} for b in lband}
  for b in lband:
    dfb = dframe[dframe.band == b]
    for idx in dfb.index:
      sig = dframe.at[idx,'filtsig']
      dprop = getcyclefeatures(sig, sampr, 1.5 * dframe.at[idx,'maxF'])
      if 'rdsym' not in dbprop[b]:
        for k in dprop.keys(): dbprop[b][k] = []
      for k in dprop.keys():
        if not iterable(dprop[k]):
          if not isnan(dprop[k]):
            dbprop[b][k].append(dprop[k])
        else:
          for x in dprop[k]:
            if not isnan(x):
              dbprop[b][k].append(x)
  return dbprop

# get cycle information in a dictionary, using same index as into dframe
# operates on either select or full set of osc. events
def getcycbyevidx (dframe, levidx, sampr):
  ddprop = OrderedDict() # to make sure the event time order is preserved
  for evidx in levidx:
    sig = dframe.at[evidx,'filtsig']
    dprop = getcyclefeatures(sig, sampr, 1.5 * dframe.at[evidx,'maxF'])
    ddprop[evidx] = dprop
  return ddprop

#
def getintrapeakdistrib (ddprop):
  lout = []
  for idx in ddprop.keys():
    for x in ddprop[idx]['interpeakt']: lout.append(x)
  return lout

# this function assumes that ddprop and dframe have events sorted by increasing time
def getinterpeakdistrib (dframe, ddprop):
  #newlist = sorted(lblob, key=lambda x: x.left)
  lout = []
  lkey = list(ddprop.keys())
  for i in range(0,len(lkey)-1,1):
    idx = lkey[i]
    lastpkT = dframe.at[idx,'absminT'] + ddprop[idx]['peakt'][-1]
    jdx = lkey[i+1]
    nextpkT = dframe.at[jdx,'absminT'] + ddprop[jdx]['peakt'][0]
    interT = nextpkT - lastpkT
    if interT >= 0: lout.append(interT)
  return lout

# adjust blobs after merge - not used now
def adjustblobs (lblob,image,fctr=0.5,F=None,T=None):
  rows,cols = image.shape
  for blob in lblob:
    pkx,pky = blob.maxpos[1],blob.maxpos[0]
    mxval = blob.maxvalorig
    blob.left,blob.right,blob.top,blob.bottom = findbounds(image,pkx,pky,mxval*fctr)
    if F is not None:
      blob.minF = F[blob.bottom] # get the frequencies
      blob.maxF = F[min(blob.top,len(F)-1)]
      blob.peakF = F[blob.maxpos[0]]
    if T is not None:
      blob.minT = T[blob.left]
      blob.maxT = T[min(blob.right,len(T)-1)]
      blob.peakT = T[blob.maxpos[1]]

# from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detectpeaks (image):
  """
  Takes an image and detect the peaks usingthe local maximum filter.
  Returns a boolean mask of the peaks (i.e. 1 when
  the pixel's value is the neighborhood maximum, 0 otherwise)
  """
  # define an 8-connected neighborhood
  neighborhood = generate_binary_structure(2,2)
  #apply the local maximum filter; all pixel of maximal value 
  #in their neighborhood are set to 1
  local_max = maximum_filter(image, footprint=neighborhood)==image
  #local_max is a mask that contains the peaks we are 
  #looking for, but also the background.
  #In order to isolate the peaks we must remove the background from the mask.
  #we create the mask of the background
  background = (image==0)
  #a little technicality: we must erode the background in order to 
  #successfully subtract it form local_max, otherwise a line will 
  #appear along the background border (artifact of the local maximum filter)
  eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
  #we obtain the final mask, containing only peaks, 
  #by removing the background from the local_max mask (xor operation)
  detected_peaks = local_max ^ eroded_background
  return detected_peaks

#
def getallchanblobs (dat,medthresh):
  outd = {}
  nchan = dat.shape[1]
  for chan in range(dat.shape[1]):
    outd[chan] = {}
    print('up to channel',chan)
    lms,lnoise,lsidx,leidx = getmorletwin(dat[:,chan],int(10*sampr),sampr)
    lmsnorm = [mednorm(ms.TFR) for ms in lms]
    specsamp = lms[0].TFR.shape[1] # 
    specdur = specsamp / sampr # 
    llblob, llnblob = [],[]
    for lmsn in lmsnorm:
      lblob, lnblob = blobcut(lmsn,medthresh);
      llblob.append(lblob)
      llnblob.append(lnblob)
    levblob = []; llblobf = []
    for i in range(len(llblob)):
      if lnoise[i]: continue
      print('getting blobs from window',i)
      lblob, lnblob = llblob[i], llnblob[i]  
      lblobf = getblobfeatures(lmsnorm[i],lblob,lms[i].TFR,lms[i].t,lms[i].f) # now fast
      llblobf.append(lblobf)
      for blobf in lblobf: levblob.append(blobf)
      ldur = [blobf.maxT - blobf.minT for blobf in levblob]
      lpeakF = [blobf.peakF for blobf in levblob]
      lmaxval = [blobf.maxvalorig for blobf in levblob]
      lavgval = [blobf.avgpoworig for blobf in levblob]
      lncycle = [dur*peak/1e3 for dur,peak in zip(ldur,lpeakF)] # had mistake in ncycle        
    #outd[chan]['lms']=lms
    outd[chan]['lnoise']=lnoise
    #outd[chan]['lmsnorm']=lmsnorm
    outd[chan]['specsamp']=specsamp
    outd[chan]['specdur']=specdur
    #outd[chan]['llblob'] = llblob
    #outd[chan]['llnblob'] = llnblob
    outd[chan]['levblob'] = levblob
    outd[chan]['llblobf'] = llblobf
    outd[chan]['ldur'] = ldur
    outd[chan]['lpeakF'] = lpeakF
    outd[chan]['lmaxval'] = lmaxval
    outd[chan]['lavgval'] = lavgval
    outd[chan]['lncycle'] = lncycle
  return outd
  
#
def normarr (a):
  sd = numpy.std(a)
  return (a - mean(a)) / sd

# downsample the signal using scipy decimate
def downsamp (vec,fctr):
  pl = vec.to_python()
  pld = decimate(pl,fctr)
  vd = h.Vector()
  vd.from_python(pld)
  return vd

# get index of first element in lfreq >= f1
def firstIDX (f1,lfreq):
  for i in range(len(lfreq)):
    if lfreq[i] >= f1:
      return i
  return -1

# get band's indices into lfreq
def minmaxIDX (band,lfreq):
  minf,maxf=dbands[band]
  i = 0; sidx,eidx=-1,-1
  for i in range(len(lfreq)):
    if lfreq[i] >= minf and sidx == -1:
      sidx = i
    if lfreq[i] >= maxf and eidx == -1:
      eidx = i
  return sidx,eidx

#
def normarr (x):
  a = np.array(x)
  m = mean(a)
  s = std(a)
  return (a - m) / s

# get oscillatory events
# lms is list of windowed morlet spectrograms, lmsnorm is spectrograms normalized by median in each power
# lnoise is whether the window had noise, medthresh is median threshold for significant events,
# lsidx,leidx are starting/ending indices into original time-series, csd is current source density
# on the single chan, MUA is multi-channel multiunit activity, overlapth is threshold for merging
# events when bounding boxes overlap, fctr is fraction of event amplitude to search left/right/up/down
# when terminating events
def getspecevents (lms,lmsnorm,lnoise,medthresh,lsidx,leidx,csd,MUA,chan,sampr,overlapth=0.5,fctr=0.5,getphase=False):
  llevent = []
  for windowidx,offidx,ms,msn,noise in zip(arange(len(lms)),lsidx,lms,lmsnorm,lnoise): 
    imgpk = detectpeaks(msn) # detect the 2D local maxima
    lblob = getblobsfrompeaks(msn,imgpk,ms.TFR,medthresh,fctr=fctr,T=ms.t,F=ms.f) # cut out the blobs/events
    lblobsig = [blob for blob in lblob if blob.maxval >= medthresh] # take only significant events
    lmergeset,bmerged = getmergesets(lblobsig,overlapth) # determine overlapping events
    lmergedblobs = getmergedblobs(lblobsig,lmergeset,bmerged)
    # get the extra features (before/during/after with MUA,avg,etc.)
    getextrafeatures(lmergedblobs,ms,msn,medthresh,csd,MUA,chan,offidx,sampr,fctr=fctr,getphase=getphase)
    for blob in lmergedblobs: # store offsets for getting to time-series / wavelet spectrograms
      blob.windowidx = windowidx
      blob.offidx = offidx
      blob.duringnoise = noise
    llevent.append(lmergedblobs) # save merged events
  return llevent

#
def getDynamicThresh (lmsn, lnoise, thfctr, defthresh):
  lthresh = [mean(x)+thfctr*std(x) for x,n in zip(lmsn,lnoise) if not n]
  if len(lthresh) > 0:
    print('Mean/min/max:',mean(lthresh),min(lthresh),max(lthresh))
    return min(lthresh)
  return defthresh # default is 4.0

#
def getIEIstatsbyBand (dat,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,MUA,overlapth=0.5,getphase=True,savespec=False,useDynThresh=False,threshfctr=2.0,useloglfreq=False,mspecwidth=7.0,noiseamp=noiseampCSD):
  # get the interevent statistics split up by frequency band
  dout = {'sampr':sampr,'medthresh':medthresh,'winsz':winsz,'freqmin':freqmin,'freqmax':freqmax,'freqstep':freqstep,'overlapth':overlapth}
  dout['threshfctr'] = threshfctr; dout['useDynThresh']=useDynThresh; dout['mspecwidth'] = mspecwidth; dout['noiseamp']=noiseamp
  for chan in lchan:
    dout[chan] = doutC = {'delta':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'theta':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'alpha':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'beta':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'gamma':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'hgamma':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'lnoise':[]}
    print('up to channel', chan,'getphase:',getphase)
    if dat.shape[0] > dat.shape[1]:
      sig = dat[:,chan] # signal (either CSD or LFP)
      lms,lnoise,lsidx,leidx = getmorletwin(dat[:,chan],int(winsz*sampr),sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,useloglfreq=useloglfreq,mspecwidth=mspecwidth,noiseamp=noiseamp)
    else:
      sig = dat[chan,:] # signal (either CSD or LFP)
      lms,lnoise,lsidx,leidx = getmorletwin(dat[chan,:],int(winsz*sampr),sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,useloglfreq=useloglfreq,mspecwidth=mspecwidth,noiseamp=noiseamp)
    if 'lsidx' not in dout: dout['lsidx'] = lsidx # save starting indices into original data array
    if 'leidx' not in dout: dout['leidx'] = leidx # save ending indices into original data array
    lmsnorm = [mednorm(ms.TFR) for ms in lms] # normalize wavelet specgram by median
    if useDynThresh: # using dynamic threshold?
      evthresh = getDynamicThresh(lmsnorm, lnoise, threshfctr, medthresh)
    else: #  otherwise use the default medthresh
      evthresh = medthresh
    doutC['evthresh'] = evthresh # save the threshold used    
    specsamp = lms[0].TFR.shape[1] # number of samples in spectrogram time axis
    specdur = specsamp / sampr # spectrogram duration in seconds
    if 'specsamp' not in dout: dout['specsamp'] = specsamp
    if 'specdur' not in dout: dout['specdur'] = specdur    
    llevent = getspecevents(lms,lmsnorm,lnoise,evthresh,lsidx,leidx,sig,MUA,chan,sampr,overlapth=overlapth,getphase=getphase) # get the spectral events
    scalex = 1e3*specdur/specsamp # to scale indices to times
    if 'scalex' not in dout: dout['scalex'] = scalex
    doutC['lnoise'] = lnoise # this is per channel - diff noise on each channel
    myt = 0
    for levent,msn,ms in zip(llevent,lmsnorm,lms):
      print(myt)
      """ do not skip noise so can look at noise event waveforms in eventviewer; can always filter out noise from dframe
      if lnoise[myt]: # skip noise
        myt+=1
        continue      
      """
      for band in ['delta','theta','alpha','beta','gamma','hgamma']: # check events by band
        lband = getblobinrange(levent,dbands[band][0],dbands[band][1])
        count = len(lband)
        doutC[band]['Count'].append(count)
        doutC[band]['levent'].append(lband)
        if count > 2:
          lbandIEI = getblobIEI(lband,scalex)
          cv = getCV2(lbandIEI)
          doutC[band]['CV'].append(cv)
          doutC[band]['IEI'].append(lbandIEI)
        else:
          doutC[band]['IEI'].append([])
        if count > 3:
          lv = getLV(lbandIEI)
          doutC[band]['LV'].append(lv)
          print(band,len(lband),lv,cv)
      myt+=1
      for band in ['delta','theta','alpha','beta','gamma','hgamma']: doutC[band]['FF'] = getFF(doutC[band]['Count'])
    if savespec:
      for MS,MSN in zip(lms,lmsnorm): MS.TFR = MSN # do not save lmsnorm separately, just copy it over to lms
      doutC['lms'] = lms
    else:
      del lms,lmsnorm # cleanup memory
      gc.collect()
  dout['lchan'] = lchan
  return dout

#
def GetDFrame (dout,sampr,CSD, MUA, alignby = 'bywaveletpeak',haveMUA=True):
  totsize = 0 # total number of events
  for chan in dout['lchan']:
    for band in lband:
      for levent in dout[chan][band]['levent']:
        totsize += len(levent)
  row_list = []
  columns=['chan','dur','maxvalbefore','maxval','maxvalafter','ncycle',\
           'dom','dombefore','domafter','domevent',\
           'MUA','MUAbefore','MUAafter','avgpowbefore','avgpow','avgpowafter','avgpowevent',\
           'sampen','sampenbefore','sampenafter',\
           'minvalbefore','minval','minvalafter','hasbefore','hasafter',\
           'band','windowidx','offidx','duringnoise',\
           'minF','maxF','peakF','Fspan','Foct',\
           'minT','maxT','peakT','left','right','bottom','top','maxpos',\
           'codelta','cotheta','coalpha','cobeta','cogamma','cohgamma','coother',\
           'CSDMUACorr','CSDMUACorrbefore','CSDMUACorrafter',\
           'WavePeakVal','WavePeakIDX','WaveTroughVal','WaveTroughIDX','WaveH','WaveW','WavePeakT','WaveTroughT',\
           'WaveletPeakPhase','WaveletPeakVal','WaveletPeakIDX','WaveletPeakT',\
           'WaveletLeftTroughPhase','WaveletLeftTroughVal','WaveletLeftTroughIDX','WaveletLeftTroughT',\
           'WaveletRightTroughPhase','WaveletRightTroughVal','WaveletRightTroughIDX','WaveletRightTroughT',\
           'WaveletLeftH','WaveletLeftW','WaveletLeftSlope',\
           'WaveletRightH','WaveletRightW','WaveletRightSlope',\
           'WaveletFullH','WaveletFullW','WaveletFullSlope',\
           'absPeakT',\
           'absminT',\
           'absmaxT',\
           'absWaveletLeftTroughT',\
           'absWaveletRightTroughT',\
           'absWaveletPeakT',\
           'filtsig','filtsigcor',\
           'MUARatDOB','MUARatDOA','SampenRatDOB','SampenRatDOA','arrMUAbefore','arrMUA','arrMUAafter',\
           'RLWidthRat','RLHeightRat','RLSlopeRat',
           'CSDwvf','MUAwvf','alignoffset','siglen']
  lcyckeys = getcyclekeys()
  for k in lcyckeys: columns.append('cyc_'+k)
  allevents = []
  dchanevents = {}
  MUAwvf = None # MUA waveform (if MUA provided)
  for chan in dout['lchan']:
    for band in lband:
      for levent in dout[chan][band]['levent']:
        for ev in levent:
          # more featurs - ratio of mua, ratio of sampen, during event over mua, sampen before,after
          MUARatDOB = MUARatDOA = SampenRatDOB = SampenRatDOA = arrMUAbefore = arrMUA = arrMUAafter = 0
          # right div by left width, height, slope ratios
          RLWidthRat = RLHeightRat = RLSlopeRat = 0
          if ev.hasbefore and ev.hasafter:
            if ev.sampenbefore > 0: SampenRatDOB = ev.sampen / ev.sampenbefore
            if ev.sampenafter > 0: SampenRatDOA = ev.sampen / ev.sampenafter
          if haveMUA:
            if ev.hasbefore and ev.hasafter:
              if ev.MUAbefore > 0: MUARatDOB = ev.MUA / ev.MUAbefore
              if ev.MUAafter > 0: MUARatDOA = ev.MUA / ev.MUAafter
            arrMUAbefore = ev.arrMUAbefore
            arrMUA = ev.arrMUA
            arrMUAafter = ev.arrMUAafter
          # ratio of wavelet left,right widths, heights, slopes
          if ev.WaveletLeftW > 0.: RLWidthRat = ev.WaveletRightW / ev.WaveletLeftW
          if ev.WaveletLeftH > 0.: RLHeightRat = ev.WaveletRightH / ev.WaveletLeftH
          if ev.WaveletLeftSlope != 0.: RLSlopeRat = ev.WaveletRightSlope / ev.WaveletLeftSlope
          ######################################################################################### 
          # get the waveforms for storage in the dataframe
          left,right = int(ev.left+ev.offidx), int(ev.right+ev.offidx) # offidx to get back into the original time-series
          alignoffset = getalignoffset(ev, alignby)          
          CSDwvf = CSD[chan,left:right] # CSD waveform
          if haveMUA: MUAwvf = MUA[chan+1,left:right] # MUA waveform
          siglen = len(CSDwvf) # signal length
          #########################################################################################
          # vals is a list of values for each event
          vals = [chan,ev.dur,ev.maxvalbefore,ev.maxval,ev.maxvalafter,ev.ncycle,\
                  ev.dom,ev.dombefore,ev.domafter,ev.domevent,\
                  ev.MUA,ev.MUAbefore,ev.MUAafter,ev.avgpowbefore,ev.avgpow,ev.avgpowafter,ev.avgpowevent,\
                  ev.sampen,ev.sampenbefore,ev.sampenafter,\
                  ev.minvalbefore,ev.minval,ev.minvalafter,int(ev.hasbefore),int(ev.hasafter),\
                  band,ev.windowidx,ev.offidx,ev.duringnoise,\
                  ev.minF,ev.maxF,ev.peakF,ev.Fspan,ev.Foct,\
                  ev.minT,ev.maxT,ev.peakT,ev.left,ev.right,ev.bottom,ev.top,ev.maxpos[1],\
                  ev.codelta,ev.cotheta,ev.coalpha,ev.cobeta,ev.cogamma,ev.cohgamma,ev.coother,\
                  ev.CSDMUACorr,ev.CSDMUACorrbefore,ev.CSDMUACorrafter,\
                  ev.WavePeakVal,ev.WavePeakIDX,ev.WaveTroughVal,ev.WaveTroughIDX,ev.WaveH,ev.WaveW,ev.WavePeakT,ev.WaveTroughT,\
                  ev.WaveletPeak.phs,ev.WaveletPeak.val,ev.WaveletPeak.idx,ev.WaveletPeak.T,\
                  ev.WaveletLeftTrough.phs,ev.WaveletLeftTrough.val,ev.WaveletLeftTrough.idx,ev.WaveletLeftTrough.T,\
                  ev.WaveletRightTrough.phs,ev.WaveletRightTrough.val,ev.WaveletRightTrough.idx,ev.WaveletRightTrough.T,\
                  ev.WaveletLeftH,ev.WaveletLeftW,ev.WaveletLeftSlope,\
                  ev.WaveletRightH,ev.WaveletRightW,ev.WaveletRightSlope,\
                  ev.WaveletFullH,ev.WaveletFullW,ev.WaveletFullSlope,\
                  ev.peakT+index2ms(ev.offidx,sampr),\
                  ev.minT+index2ms(ev.offidx,sampr),\
                  ev.maxT+index2ms(ev.offidx,sampr),\
                  ev.WaveletLeftTrough.T+index2ms(ev.offidx,sampr),\
                  ev.WaveletRightTrough.T+index2ms(ev.offidx,sampr),\
                  ev.WaveletPeak.T+index2ms(ev.offidx,sampr),\
                  ev.filtsig,ev.filtsigcor,\
                  MUARatDOB,MUARatDOA,SampenRatDOB,SampenRatDOA,arrMUAbefore,arrMUA,arrMUAafter,\
                  RLWidthRat,RLHeightRat,RLSlopeRat,
                  CSDwvf, MUAwvf, alignoffset, siglen]
          ######################################################################################### 
          # get the cycle features for storage in the dataframe
          dprop = getcyclefeatures(ev.filtsig, sampr, 1.5 * ev.maxF)
          for k in lcyckeys: vals.append(dprop[k])
          ######################################################################################### 
          # based on https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
          row_list.append(dict((c,v) for c,v in zip(columns,vals)))
          allevents.append(ev)
  # now create the final dataframe
  pdf = pd.DataFrame(row_list, index=np.arange(0,totsize), columns=columns)
  pdf = pdf.sort_values('absPeakT') # sort by absPeakT; index will be out of order, but will correspond to dout order
  return pdf        

  
# find closest to val, associated index around sig[sidx-winsz:sidx+winsz] as long as within bounds of left,right
def findclosest (sig, sidx, left, right, winsz, val, lookleft=True, lookright=True):
  sz = len(sig)
  closeerr = abs(sig[sidx]-val)  
  closeidx = sidx
  closeval = sig[sidx]
  left=int(left); right=int(right); sidx=int(sidx); winsz=int(winsz)
  if lookright:
    SIDX = min(right,sidx+1); EIDX = min(right,sidx+winsz+1)
    for idx in range(SIDX,EIDX,1):
      err = abs(sig[idx] - val)
      if err < closeerr:
        closeerr = err
        closeval = sig[idx]
        closeidx = idx
        #print(err,closeval,closeidx)
  if lookleft:
    SIDX = max(left,sidx-1); EIDX = max(left,sidx-winsz)
    for idx in range(SIDX,EIDX,-1):
      err = abs(sig[idx] - val)
      if err < closeerr:
        closeerr = err      
        closeval = sig[idx]
        closeidx = idx
        #print(err,closeval,closeidx)      
  return closeval,closeidx

# find minimum value, associated index around sig[sidx-winsz:sidx+winsz] as long as within bounds of left,right
def findtrough (sig, sidx, left, right, winsz):
  sz = len(sig)
  minval = sig[sidx]
  minidx = sidx
  left=int(left); right=int(right); sidx=int(sidx); winsz=int(winsz)
  SIDX = min(right,sidx+1); EIDX = min(right,sidx+winsz+1)
  for idx in range(SIDX,EIDX,1):
    if sig[idx] < minval:
      minval = sig[idx]
      minidx = idx
  SIDX = max(left,sidx-1); EIDX = max(left,sidx-winsz)
  for idx in range(SIDX,EIDX,-1):
    if sig[idx] < minval:
      minval = sig[idx]
      minidx = idx
  return minval,minidx

# find maximum value, associated index around sig[sidx-winsz:sidx+winsz] as long as within bounds of left,right
def findpeak (sig, sidx, left, right, winsz):
  sz = len(sig)
  maxval = sig[sidx]
  maxidx = sidx
  left=int(left); right=int(right); sidx=int(sidx); winsz=int(winsz)
  SIDX = min(right,sidx+1); EIDX = min(right,sidx+winsz+1)
  for idx in range(SIDX,EIDX,1):
    if sig[idx] > maxval:
      maxval = sig[idx]
      maxidx = idx
  SIDX = max(left,sidx-1); EIDX = max(left,sidx-winsz)
  for idx in range(SIDX,EIDX,-1):
    if sig[idx] > maxval:
      maxval = sig[idx]
      maxidx = idx
  return maxval,maxidx

# get position alignment vector
def getalignvec (dframe,align):
  if align == 'bywavepeak':
    return dframe['WavePeakIDX']
  elif align == 'bywavetrough':
    return dframe['WaveTroughIDX']
  elif align == 'bywaveletpeak':
    return dframe['WaveletPeakIDX']
  elif align == 'bywaveletlefttrough':
    return dframe['WaveletLeftTroughIDX']
  elif align == 'bywaveletrighttrough':
    return dframe['WaveletRightTroughIDX']
  return dframe['maxpos']

#
def getminwavewidth (dframe,lidx):
  vL,vR = dframe['left'],dframe['right']
  minw=1e9
  for idx in lidx: minw = min(minw,vR[idx]-vL[idx]+1) # right side included
  if minw%2==0: minw+=1
  minw=int(minw)
  minw2=int(minw/2)
  return minw,minw2

#
def getmaxwavewidth (dframe,lidx):
  vL,vR = dframe['left'],dframe['right']
  maxw=-1e9
  for idx in lidx: maxw = max(maxw,vR[idx]-vL[idx]+1) # right side included
  if maxw%2==0: maxw+=1
  maxw=int(maxw)
  maxw2=int(maxw/2)
  return maxw,maxw2

def vtoint (vec): return [int(x) for x in vec]

# get average event with specified alignment
def getavgevent (dframe,sampr,levidx,CSD,MUA,bpass=None,align='bywaveletpeak',\
                 useMUA=False,allchan=False,usedepsyn=False,usehypsyn=False,muachan=None,
                 usenqwvf=False,usemaxw=False):
  vchan = dframe['chan']
  voffidx = dframe['offidx']
  vwindowidx = dframe['windowidx']
  vL,vR = dframe['left'],dframe['right']
  if usenqwvf: vlen=dframe['siglen']
  valignpos = getalignvec(dframe,align)
  minw,minw2 = getminwavewidth(dframe,levidx)
  maxw,maxw2 = getmaxwavewidth(dframe,levidx)
  nrow = 1
  vec = h.Vector()
  if allchan:
    if useMUA: nrow = MUA.shape[0]
    else: nrow = CSD.shape[0]
  sz = 0
  if CSD is not None: sz = max(CSD.shape[0],CSD.shape[1])
  if usemaxw:
    avge = zeros((nrow,maxw))
  else:
    avge = zeros((nrow,minw))
  lrow = [0]  
  if allchan: lrow = [r for r in range(nrow)]  
  for idx in levidx:
    print(idx)
    chan = int(vchan[idx]); offidx = int(voffidx[idx])
    if CSD is not None: # +/-minw2, centered around alignment position; l,r indices into original time-series so need to use offidx too
      if usemaxw:
        l,r = int(valignpos[idx]-vL[idx]),min(vlen[idx],int(valignpos[idx]-vL[idx]+vR[idx]-vL[idx]))
      else:
        l,r = int(valignpos[idx]-minw2+offidx),min(sz,int(valignpos[idx]+minw2+offidx))
    else: # using dframe (usenqwvf==True)
      if usemaxw:
        l,r = int(valignpos[idx]-vL[idx]),min(vlen[idx],int(valignpos[idx]-vL[idx]+vR[idx]-vL[idx]))
      else:
        l,r = int(valignpos[idx]-vL[idx]-minw2),min(vlen[idx],int(valignpos[idx]-vL[idx]+minw2))
    shift = 0 
    if l < 0: # need to shift the index into average if alignment position has less than l values to the left of it
      shift = -l
      l = 0
    w = int(r-l+1)
    w2 = int(w/2)
    if usemaxw:
      avgidx = shift+maxw2-w2
      lenA = len(avge[0,shift+maxw2-w2:shift+maxw2+w2]) # due to numpy indexing can't just take w2*2 as length        
    else:
      avgidx = shift+minw2-w2
      lenA = len(avge[0,shift+minw2-w2:shift+minw2+w2]) # due to numpy indexing can't just take w2*2 as length          
    if allchan:
      for row in range(nrow):
        if usenqwvf:
          if chan == row:
            if useMUA: vec = dframe.at[idx,'MUAwvf']
            else: vec = dframe.at[idx,'CSDwvf']
            if usemaxw: sig = vec[l-w2:r-w2]
            else: sig = vec[l:r]        
        elif usedepsyn:
          sig = depsig[row,l:r]
        elif usehypsyn:
          sig = hypsig[row,l:r]
        elif useMUA:
          sig = MUA[row,l:r]
        else:
          sig = CSD[row,l:r]
        if bpass: sig = bandpass(sig,bpass[0],bpass[1],sampr,True)
        Navg = 0
        for sigidx in range(len(sig)): # use an inefficient (non-vectorized) loop to avoid dealing with all the np indexing issues
          if Navg >= lenA: break
          avge[row,avgidx] += sig[sigidx]
          avgidx += 1
          Navg += 1                
    else:
      if usenqwvf:
        if useMUA: vec = dframe.at[idx,'MUAwvf']
        else: vec = dframe.at[idx,'CSDwvf']
        if usemaxw: sig = vec[l-w2:r-w2]
        else: sig = vec[l:r]
        #print('l:',l,'r:',r,'w:',w,'len(sig):',len(sig),'minw2:',minw2,'w2:',w2,'valignpos[idx]:',valignpos[idx],'vL[idx]:',vL[idx],'vR[idx]:',vR[idx],'shift:',shift,'maxw2:',maxw2)
      elif useMUA:
        sig = MUA[chan+1,l:r]
      else:
        sig = CSD[chan,l:r]
      if bpass: sig = bandpass(sig,bpass[0],bpass[1],sampr,True)
      Navg = 0
      for sigidx in range(len(sig)): # use an inefficient (non-vectorized) loop to avoid dealing with all the np indexing issues
        if Navg >= lenA: break
        avge[0,avgidx] += sig[sigidx]
        avgidx += 1
        Navg += 1
      #print(shift+maxw2-w2,avgidx,lenA,Navg,sigidx,len(sig))
  avge /= float(len(levidx))
  if usemaxw:
    tt = linspace(-1e3*maxw2/sampr,1e3*maxw2/sampr,avge.shape[1])
  else:
    tt = linspace(-1e3*minw2/sampr,1e3*minw2/sampr,avge.shape[1])
  return tt,avge

# get average spectrogram of events specified by levidx
def getavgSPEC (dframe,sampr,levidx,dlms,align='bywavepeak'):
  vchan = dframe['chan']
  voffidx = dframe['offidx']
  vwindowidx = dframe['windowidx']
  valignpos = getalignvec(dframe,align)
  minw,minw2 = getminwavewidth(dframe,lidx)
  nrow,sz = dlms[vchan[0]][0].TFR.shape
  avgspec = zeros((nrow,minw))
  for idx in levidx:
    chan = vchan[idx]
    offidx = voffidx[idx]
    windowidx = vwindowidx[idx]
    l,r = max(0,int(valignpos[idx]-minw2+offidx)),min(sz,int(valignpos[idx]+minw2+offidx))
    w = int(r-l+1)
    w2 = int(w/2)
    if (l-r)%2==0:
      avgspec[:,minw2-w2:minw2+w2] += dlms[chan][windowidx].TFR[:,l:r]
    else:
      avgspec[:,minw2-w2:minw2+w2] += dlms[chan][windowidx].TFR[:,l:r+1]
  avgspec /= float(len(lidx))
  tt = linspace(-1e3*minw2/sampr,1e3*minw2/sampr,minw)
  return tt,avgspec

# get all major event properties, used for drawing the event or other...
def geteventprop (dframe,evidx,align):
  evidx=int(evidx)
  dur,chan,hasbefore,hasafter,windowidx,offidx,left,right,minT,maxT,peakT,minF,maxF,peakF,avgpowevent,ncycle,WavePeakT,WaveTroughT,WaveletPeakT,WaveletLeftTroughT,WaveletRightTroughT,filtsigcor,Foct = [dframe.at[evidx,c] for c in ['dur','chan','hasbefore','hasafter','windowidx','offidx','left','right','minT','maxT','peakT','minF','maxF','peakF','avgpowevent','ncycle','WavePeakT','WaveTroughT','WaveletPeakT','WaveletLeftTroughT','WaveletRightTroughT','filtsigcor','Foct']]
  if 'cyc_npeak' in dframe.columns:
    cycnpeak = dframe.at[evidx,'cyc_npeak']
  else:
    cycnpeak = -1
  if 'ERPscore' in dframe.columns:
    ERPscore = dframe.at[evidx,'ERPscore']
  else:
    ERPscore = -2
  if False and 'OSCscore' in dframe.columns:
    OSCscore = dframe.at[evidx,'OSCscore']
  else:
    OSCscore = -2
  band=dframe.at[evidx,'band']
  w2 = int((right-left+1)/2.)
  left=int(left+offidx); right=int(right+offidx);
  alignoffset = 0 # offset to align waveforms to 0, only used when specified as below
  if align == 'byspecpeak':
    alignoffset = -peakT
  elif align == 'bywavepeak':
    alignoffset = -WavePeakT
  elif align == 'bywavetrough':
    alignoffset = -WaveTroughT
  elif align == 'bywaveletpeak':
    alignoffset = -WaveletPeakT
  elif align == 'bywaveletlefttrough':
    alignoffset = -WaveletLeftTroughT
  elif align == 'bywaveletrighttrough':
    alignoffset = -WaveletRightTroughT
  #print('align:',peakT,align,alignoffset)
  return dur,int(chan),hasbefore,hasafter,int(windowidx),offidx,left,right,minT,maxT,peakT,minF,maxF,peakF,avgpowevent,ncycle,WavePeakT,WaveTroughT,WaveletPeakT,WaveletLeftTroughT,WaveletRightTroughT ,w2,left,right,band,alignoffset,filtsigcor,Foct,cycnpeak,ERPscore,OSCscore

#
def getalignoffset (ev, alignby):
  # offset to align waveforms to 0, only used when specified as below
  # 'WavePeakT','WaveTroughT','WaveletPeakT','WaveletLeftTroughT','WaveletRightTroughT'  
  if alignby == 'byspecpeak':
    return -ev.peakT
  elif alignby == 'bywavepeak':
    return -ev.WavePeakT
  elif alignby == 'bywavetrough':
    return -ev.WaveTroughT
  elif alignby == 'bywaveletpeak':
    return -ev.WaveletPeak.T
  elif alignby == 'bywaveletlefttrough':
    return -ev.WaveletLeftTrough.T
  elif alignby == 'bywaveletrighttrough':
    return -ev.WaveletRightTrough.T
  return 0
    
#
def drawwaveforms (dframe,levidx,clr='r',lw=1,getsig=False):
  align = 'bywaveletpeak'
  lsigcsd=[]; lsigmua=[]; ltt=[]
  for evidx in levidx:
    dur,chan,hasbefore,hasafter,windowidx,offidx,left,right,minT,maxT,peakT,minF,maxF,peakF,avgpowevent,ncycle,WavePeakT,WaveTroughT,WaveletPeakT,WaveletLeftTroughT,WaveletRightTroughT,w2,left,right,band,alignoffset,filtsigcor,Foct,cycnpeak,ERPscore,OSCscore = geteventprop(dframe,evidx,align)
    subplot(2,1,1)
    sig = dframe.at[evidx,'CSDwvf']
    tt = linspace(minT,maxT,len(sig)) + alignoffset
    plot(tt,sig,color=clr,linewidth=lw)
    if getsig:
      ltt.append(tt)
      lsigcsd.append(sig)
    subplot(2,1,2)
    sig = dframe.at[evidx,'MUAwvf']
    plot(tt,sig,color=clr,linewidth=lw)
    if getsig: lsigmua.append(sig)
  if getsig:
    return ltt,lsigcsd,lsigmua

#
def drawavgwaveforms (dframe,levidx,sampr,clr='r',lw=1,getsig=False,CSD=None,align='bywaveletpeak',xl=None,usemaxw=False,drawMUA=True,bpass=None):
  lsigcsd=[]; lsigmua=[]; ltt=[]
  tt,avgCSD=getavgevent(dframe,sampr,levidx,None,None,usenqwvf=True,align=align,usemaxw=usemaxw,bpass=bpass)
  avgCSD=avgCSD[0,:]
  tt,avgMUA=getavgevent(dframe,sampr,levidx,None,None,useMUA=True,usenqwvf=True,align=align,usemaxw=usemaxw,bpass=bpass)
  avgMUA=avgMUA[0,:]
  if drawMUA: subplot(2,1,1)
  plot(tt[:-1],avgCSD[:-1],color=clr,linewidth=lw); ylabel('CSD');
  if xl is not None: xlim(xl)
  if drawMUA:
    subplot(2,1,2)
    plot(tt[:-1],avgMUA[:-1],color=clr,linewidth=lw); ylabel('MUA')
    if xl is not None: xlim(xl)  
  xlabel('Time (ms)')
  if getsig: return tt,avgCSD,avgMUA

# add OSCillation score to pdf
def addOSCscore (pdf):
  pdf['OSCscore'] = pd.Series([-2 for i in range(len(pdf))], index=pdf.index) 
  for idx in pdf.index:
    ncycle = pdf.at[idx, 'ncycle']
    Foct = pdf.at[idx, 'Foct']
    if Foct > 0.:
      pdf.loc[idx,'OSCscore'] = ncycle / Foct  

#
@plt.FuncFormatter
def fake_log(x, pos):  
  'The two args are the value and tick position'
  global lfreq
  x = int(x)
  #print('x is:',x)
  if x < 0:
    return r'${%.2f}$' % (lfreq[0])
  if x >= len(lfreq):
    return r'${%.2f}$' % (lfreq[-1])
  #print('x is :',x, lfreq[int(x)])        
  return r'${%.2f}$' % (lfreq[int(x)])
  
#
class eventviewer():
  # viewer for oscillatory events
  def __init__ (self,dframe,CSD,MUA,tt,sampr,winsz,dlms,useloglfreq=False):
    self.fig = figure()
    self.dframe = dframe
    self.CSD=CSD
    self.MUA=MUA
    self.tt = tt
    self.sampr=sampr
    self.winsz=winsz
    self.dlms=dlms
    self.avgCSD = self.avgMUA = self.avgSPEC = None
    self.ntrial = 0
    if self.MUA is not None:
      self.nrow = 3
    else:
      self.nrow = 2
    self.useloglfreq = useloglfreq    
    if useloglfreq:
      global lfreq,lfreqstep
      freqmin=0.5; freqmax=250.0; minstep=0.1; 
      lfreq,lfreqstep = getloglfreq(freqmin,freqmax,minstep,getstep=True)
    self.setupax()
    self.specrange = None
  def setupax (self):
    # setup axes
    self.lax = [self.fig.add_subplot(self.nrow,1,i+1) for i in range(self.nrow)]
    self.lax[-1].set_xlabel('Time (ms)')
    if self.MUA is not None:
      self.lax[-1].set_ylabel('MUA') # MUA is filtered, rectified version of the LFP, so its units should be mV?
      self.lax[-2].set_ylabel(r'CSD ($mV/mm^2$)')
    else:
      self.lax[-1].set_ylabel(r'CSD ($mV/mm^2$)')      
    self.lax[0].set_ylabel('Frequency (Hz)');
    #if self.useloglfreq: self.lax[0].yaxis.set_major_formatter(fake_log)
  def clf (self):
    # clear figure
    self.fig.clf()
    self.setupax()
  def clear (self): self.clf()
  def drawavgwaveformspec (self,vidx,freqmin=1.0,freqmax=250.0,freqstep=1.0,align='bywaveletpeak',ylspec=None,bpass=None):
    # draw average CSD
    ttavg,csdavg = getavgevent(self.dframe,self.sampr,vidx,self.CSD,self.MUA,bpass=bpass,align=align)
    csdavg = csdavg[0,:] # one row
    ttmp = (ttavg - ttavg[0])
    ms = MorletSpec(csdavg,self.sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep)
    if self.specrange:
      vmin,vmax=self.specrange
    else:
      vmin,vmax=amin(ms.TFR),amax(ms.TFR); 
    ax=self.lax[0]
    if self.useloglfreq:
      global lfreq
      ax.imshow(ms.TFR,extent=(ttavg[0],ttavg[-1],0,len(lfreq)-1),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax)
    else:
      ax.imshow(ms.TFR,extent=(ttavg[0],ttavg[-1],ms.f[0],ms.f[-1]),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax)      
    if ylspec is not None: ax.set_ylim(ylspec)
  def drawavgsyn (self,vidx,muachan,bpass=None,lw=4,clrdep='r',clrhyp='b',align='bywaveletpeak'):
    # draw measure of depolarization,hyperpolarization
    #  if CSD<0 and diff(MUA)>0=depolarization
    #  if CSD>0 and diff(MUA)<0=hyperpolarization    
    if not hasattr(self,'allsynfig'): self.allsynfig = figure()
    fig = self.allsynfig
    lax = fig.axes
    dframe,CSD,MUA,sampr,winsz = self.dframe,self.CSD,self.MUA,self.sampr,self.winsz
    ttavg,depsynavg = getavgevent(dframe,sampr,vidx,CSD,MUA,bpass=bpass,align=align,allchan=True,usedepsyn=True,muachan=muachan)
    ttavg,hypsynavg = getavgevent(dframe,sampr,vidx,CSD,MUA,bpass=bpass,align=align,allchan=True,usehypsyn=True,muachan=muachan)
    print(ttavg.shape,depsynavg.shape,hypsynavg.shape)
    mua = MUA[muachan,:]
    for cdx in range(CSD.shape[0]):
      if len(lax)==0:
        ax = fig.add_subplot(CSD.shape[0],1,cdx+1)
      else:
        ax = lax[cdx]
      ax.plot(ttavg,depsynavg[cdx,:],clrdep,linewidth=lw)
      ax.plot(ttavg,hypsynavg[cdx,:],clrhyp,linewidth=lw)    
  def drawavgcsd (self,vidx,bpass=None,lw=4,clr='b',align='bywaveletpeak',allchan=False):
    # draw average CSD
    ttavg,csdavg = getavgevent(self.dframe,self.sampr,vidx,self.CSD,self.MUA,bpass=bpass,align=align,allchan=allchan)
    if allchan:
      if not hasattr(self,'allcsdfig'): self.allcsdfig = figure()
      fig = self.allcsdfig
      lax = fig.axes
      CSD=self.CSD
      for cdx in range(CSD.shape[0]):
        if len(lax)==0:
          ax = fig.add_subplot(CSD.shape[0],1,cdx+1)
        else:
          ax = lax[cdx]
        ax.plot(ttavg,csdavg[cdx,:],clr,linewidth=lw)      
    else:
      csdavg = csdavg[0,:] # one row
      gdx = 0
      if self.nrow == 3 or self.dlms is not None: gdx = 1
      ax=self.lax[gdx]
      ax.plot(ttavg,csdavg,clr,linewidth=lw)
  def drawavgmua (self,vidx,bpass=None,lw=4,clr='b',align='bywaveletpeak',allchan=False):
    # draw average MUA
    ttavg,muaavg = getavgevent(self.dframe,self.sampr,vidx,self.CSD,self.MUA,bpass=bpass,align=align,useMUA=True,allchan=allchan)
    if allchan:
      if not hasattr(self,'allmuafig'):
        self.allmuafig = figure()
      fig = self.allmuafig
      lax = fig.axes
      MUA=self.MUA
      for cdx in range(MUA.shape[0]):
        if len(lax)==0:
          ax = fig.add_subplot(MUA.shape[0],1,cdx+1)
        else:
          ax = lax[cdx]
        ax.plot(ttavg,muaavg[cdx,:],clr,linewidth=lw)            
    else:
      muaavg = muaavg[0,:] # one row
      gdx = 1
      if self.nrow == 3: gdx = 2
      ax=self.lax[gdx]
      ax.plot(ttavg,muaavg,clr,linewidth=lw)
  def drawavgspec (self,vidx,align='bywaveletpeak',ylspec=None):
    # draw average spectrogram
    chan=int(self.dframe.at[0,'chan'])
    MS = dlms[chan][0].TFR
    tt,avgspec = getavgSPEC(self.dframe,self.sampr,vidx,self.dlms,align)
    if self.specrange:
      vmin,vmax=self.specrange
    else:
      vmin,vmax=amin(avgspec),amax(avgspec)
    ax=self.lax[0]
    if self.useloglfreq:
      global lfreq
      ax.imshow(avgspec,extent=(tt[0],tt[-1],0,len(lfreq)-1),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax)
    else:
      ax.imshow(avgspec,extent=(tt[0],tt[-1],MS.f[0],MS.f[-1]),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax)
    if ylspec is not None: ax.set_ylim(ylspec)
  def set_xlim (self,xl,fig=None):
    # set x axis limits
    if fig is not None:
      lax = fig.axes
    else:
      lax = self.lax
    for ax in lax: ax.set_xlim(xl)
  def set_ylim (self,yl,row,fig=None):
    # set y axis limits
    if fig is not None:
      lax = fig.axes
      for ax in lax: ax.set_ylim(yl)
    else:
      self.lax[row-1].set_ylim(yl)
  # highlights all the events in evidx that are on the channel in the specified window
  def highlightevents (self,windowidx,chan,levidx,ylspec=None,backclr='k',clr='r',lw=1,lwbox=2,xl=None):
    dframe,CSD,MUA,sampr,winsz = self.dframe,self.CSD,self.MUA,self.sampr,self.winsz
    ms = self.dlms[chan][windowidx]
    specsamp = ms.TFR.shape[1] # number of samples in spectrogram time axis
    specdur = specsamp / sampr # in seconds
    sidx = int(sampr*winsz*windowidx)
    eidx = int(sampr*winsz*(windowidx+1))
    scalex = 1e3*specdur/specsamp    
    scaley = ms.f[1]-ms.f[0]
    offidy = ms.f[0]
    if self.specrange:
      vmin,vmax=self.specrange
    else:
      vmin,vmax=amin(ms.TFR),amax(ms.TFR)
    if self.useloglfreq:
      global lfreq
      self.lax[0].imshow(ms.TFR,extent=(1e3*self.tt[sidx],1e3*self.tt[min(eidx,len(self.tt)-1)],0,len(lfreq)-1),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax);
      self.lax[0].set_yticks(range(len(lfreq)))
      self.lax[0].set_yticklabels([str(round(x,2)) for x in lfreq])
    else:
      self.lax[0].imshow(ms.TFR,extent=(1e3*self.tt[sidx],1e3*self.tt[min(eidx,len(self.tt)-1)],ms.f[0],ms.f[-1]),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax);      
    if ylspec is not None: self.lax[0].set_ylim(ylspec)
    self.lax[1].plot(1e3*self.tt[sidx:eidx],CSD[chan,sidx:eidx],backclr,linewidth=lw)
    if MUA is not None: self.lax[2].plot(1e3*self.tt[sidx:eidx],MUA[chan+1,sidx:eidx],backclr,linewidth=lw)
    vchan = dframe['chan']
    voffidx = dframe['offidx']
    vwindowidx = dframe['windowidx']
    vleft = dframe['left']
    vright = dframe['right']
    vtop = dframe['top']
    vbottom = dframe['bottom']
    vmaxpos = dframe['maxpos']
    vpeakF = dframe['peakF']
    vpeakT = dframe['peakT']
    for idx in levidx:
      if vwindowidx[idx] != windowidx or vchan[idx] != chan: continue
      offidx = voffidx[idx]
      leftidx,rightidx = vleft[idx]+offidx, vright[idx]+offidx
      leftT,rightT,bottom,top=1e3*self.tt[leftidx],1e3*self.tt[rightidx],scaley*vbottom[idx]+offidy,scaley*vtop[idx]+offidy
      #print('clr, lw:',clr,lw,leftT,rightT,bottom,top)
      drbox(leftT,rightT,bottom,top,clr,lwbox,self.lax[0])
      self.lax[0].plot([vpeakT[idx]+offidx*1e3/sampr],[vpeakF[idx]],clr+'o')
      self.lax[1].plot(1e3*self.tt[leftidx:rightidx],CSD[chan,leftidx:rightidx],clr,linewidth=lw)
      if MUA is not None: self.lax[2].plot(1e3*self.tt[leftidx:rightidx],MUA[chan+1,leftidx:rightidx],clr,linewidth=lw)
    if xl is None:
      self.set_xlim((1e3*self.tt[sidx],1e3*self.tt[min(eidx,len(self.tt)-1)]))
    else:
      self.set_xlim(xl)
  def getallprop(self, evidx, align): return geteventprop(self.dframe,evidx,align)
  def drawallCSD (self, evidx, align='bywaveletpeak', clr=None, lw=1):
    if not hasattr(self,'allcsdfig'): self.allcsdfig = figure()
    fig = self.allcsdfig    
    dframe,CSD,MUA,sampr,winsz = self.dframe,self.CSD,self.MUA,self.sampr,self.winsz
    lclr = ['r','g','b','c','m','y','k']
    evidx=int(evidx)
    if clr is None: clr = lclr[evidx%len(lclr)]
    dur,chan,hasbefore,hasafter,windowidx,offidx,left,right,minT,maxT,peakT,minF,maxF,peakF,avgpowevent,ncycle,WavePeakT,WaveTroughT,WaveletPeakT,WaveletLeftTroughT,WaveletRightTroughT,w2,left,right,band,alignoffset,filtsigcor,Foct,cycnpeak,ERPscore,OSCscore = self.getallprop(evidx,align)    
    for cdx in range(CSD.shape[0]):
      sig = CSD[cdx,left:right]
      tt = linspace(minT,maxT,len(sig)) + alignoffset
      ax = fig.add_subplot(CSD.shape[0],1,cdx+1)
      #ax.set_title('chan:'+str(chan)+', event:'+str(evidx)+', '+band+' peakF:'+str(round(peakF,1))+' Hz, power:'+str(round(avgpowevent,1))+', '+str(round(ncycle,1))+' cycles, duration:'+str(round(dur,1))+' ms.')
      if cdx == chan:
        CLR = clr
      else:
        CLR = 'k'
      ax.plot(tt,CSD[cdx,left:right],CLR,linewidth=lw)
  def drawallsyn (self, evidx, muachan, align='bywaveletpeak', lw=1):
    # draw measure of depolarization,hyperpolarization
    #  if CSD<0 and diff(MUA)>0=depolarization
    #  if CSD>0 and diff(MUA)<0=hyperpolarization    
    if not hasattr(self,'allsynfig'): self.allsynfig = figure()
    fig = self.allsynfig    
    dframe,CSD,MUA,sampr,winsz = self.dframe,self.CSD,self.MUA,self.sampr,self.winsz
    dur,chan,hasbefore,hasafter,windowidx,offidx,left,right,minT,maxT,peakT,minF,maxF,peakF,avgpowevent,ncycle,WavePeakT,WaveTroughT,WaveletPeakT,WaveletLeftTroughT,WaveletRightTroughT,w2,left,right,band,alignoffset,filtsigcor,Foct,cycnpeak,ERPscore,OSCscore = self.getallprop(evidx,align)      
    mua = MUA[muachan,:]
    for cdx in range(CSD.shape[0]):
      sig = CSD[cdx,left:right]
      msig = mua[left:right]
      depsig = getdepcsdmua(sig,msig)
      hypsig = gethypcsdmua(sig,msig)
      tt = linspace(minT,maxT,len(sig)) + alignoffset
      ax = fig.add_subplot(CSD.shape[0],1,cdx+1)
      ax.plot(tt,depsig,'r',linewidth=lw)
      ax.plot(tt,hypsig,'b',linewidth=lw)
  def draw (self, evidx, align='bywaveletpeak', ylspec=None, clr=None, lw=1, drawfilt=True, filtclr='b', lwfilt=3, lwbox=3, verbose=True):
    # draw an event
    dframe,CSD,MUA,sampr,winsz,dlms,fig = self.dframe,self.CSD,self.MUA,self.sampr,self.winsz,self.dlms,self.fig
    gdx = 0
    lclr = ['r','g','b','c','m','y','k']
    evidx=int(evidx)
    if clr is None: clr = lclr[evidx%len(lclr)]
    dur,chan,hasbefore,hasafter,windowidx,offidx,left,right,minT,maxT,peakT,minF,maxF,peakF,avgpowevent,ncycle,WavePeakT,WaveTroughT,WaveletPeakT,WaveletLeftTroughT,WaveletRightTroughT,w2,left,right,band,alignoffset,filtsigcor,Foct,cycnpeak,ERPscore,OSCscore = self.getallprop(evidx,align)  
    ax = self.lax[gdx]
    MS = dlms[chan][windowidx]
    if self.specrange is not None:
      vmin,vmax=self.specrange
    else:
      vmin,vmax=amin(MS.TFR),amax(MS.TFR)
    if self.useloglfreq:
      global lfreq
      ax.imshow(MS.TFR,extent=(MS.t[0]+alignoffset,MS.t[-1]+alignoffset,0,len(lfreq)-1),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax);
    else:
      ax.imshow(MS.TFR,extent=(MS.t[0]+alignoffset,MS.t[-1]+alignoffset,MS.f[0],MS.f[-1]),origin='lower',interpolation='None',aspect='auto',cmap=plt.get_cmap('jet'),vmin=vmin,vmax=vmax);
    drbox(minT+alignoffset,maxT+alignoffset,minF,maxF,'r',lwbox,ax)
    if ylspec is not None: ax.set_ylim(ylspec)
    axtstr = 'channel:'+str(chan)+', event:'+str(evidx)+', power:'+str(round(avgpowevent,1))+'\n'
    axtstr += band + ': minF:' + str(round(minF,2)) + ' Hz, maxF:' + str(round(maxF,2)) + ' Hz, '
    axtstr += 'peakF:' + str(round(peakF,2)) + ' Hz'
    axtstr += ', Foct:' + str(round(Foct,2))
    print(axtstr) # print the info
    if verbose: ax.set_title(axtstr)
    gdx += 1
    #####################################################      
    #                     PLOT BEFORE
    if hasbefore:
      idx0 = max(0,left - w2)
      idx1 = left    
      sig = CSD[chan,idx0:idx1]
      beforeT = (maxT-minT) * (idx1 - idx0) / (right - left + 1)
      tt = linspace(minT-beforeT,minT,len(sig)) + alignoffset
      ax = self.lax[gdx+0]
      ax.plot(tt,sig,'k',linewidth=lw)
      if MUA is not None:
        ax = self.lax[gdx+1]
        ax.plot(tt,MUA[chan+1,idx0:idx1],'k',linewidth=lw)
      print(tt[0],tt[-1])
    #####################################################      
    #                     PLOT DURING
    sig = CSD[chan,left:right]
    tt = linspace(minT,maxT,len(sig)) + alignoffset
    ax = self.lax[gdx+0]
    axtstr = 'duration:'+str(round(dur,1))+' ms, '
    if cycnpeak > -1: axtstr += str(int(cycnpeak)) + ' peaks, '
    axtstr += str(round(ncycle,1))+' cycles, '
    axtstr += 'filtsigcor:'+str(round(filtsigcor,2))
    if ERPscore > -2: axtstr += ', ERPscore:'+str(round(ERPscore,2))
    if OSCscore > -2: axtstr += ', OSCscore:'+str(round(OSCscore,2))
    print(axtstr) # print the info
    if verbose: ax.set_title(axtstr)
    ax.plot(tt,CSD[chan,left:right],clr,linewidth=lw)
    if drawfilt:
      fsig = np.array(dframe.at[evidx,'filtsig'])
      offY = mean(CSD[chan,left:right]) - mean(fsig)
      ax.plot(tt,fsig+offY,filtclr,linewidth=lwfilt)
    if MUA is not None:
      ax = self.lax[gdx+1]
      ax.plot(tt,MUA[chan+1,left:right],clr,linewidth=lw)
    #####################################################
    #                     PLOT AFTER
    if hasafter:
      idx0 = int(right)
      idx1 = min(idx0 + w2,max(CSD.shape[0],CSD.shape[1]))
      sig = CSD[chan,idx0:idx1]
      afterT = (maxT-minT) * (idx1 - idx0) / (right - left + 1)
      tt = linspace(maxT,maxT+afterT,len(sig)) + alignoffset
      ax = self.lax[gdx+0]
      ax.plot(tt,sig,'k',linewidth=lw)
      if MUA is not None:
        ax = self.lax[gdx+1]
        ax.plot(tt,MUA[chan+1,idx0:idx1],'k',linewidth=lw)
      print(tt[0],tt[-1])
    # reset xlim on spectrogram
    xl = ax.get_xlim()
    ax = self.lax[0]
    ax.set_xlim((xl))
  def drawavgevent (self,vidx,clr='b',freqmin=0.5,freqmax=250.0,freqstep=0.5,lw=1,align='bywaveletpeak',ylspec=None,ylavgCSD=None,ylavgMUA=None,xl=None,dravgwspec=False):
    # draw average spec,csd,mua from events specified in vidx
    if dravgwspec:
      self.drawavgwaveformspec(vidx,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,align=align,ylspec=ylspec)
    else:
      self.drawavgspec(vidx,align=align,ylspec=ylspec)
    self.drawavgcsd(vidx,clr=clr,lw=lw,align=align)
    if self.MUA is not None: self.drawavgmua(vidx,clr=clr,lw=lw,align=align)
    if ylavgCSD is not None: self.set_ylim(ylavgCSD,2)
    if ylavgMUA is not None and self.MUA is not None: self.set_ylim(ylavgMUA,3)
    if xl is not None: self.set_xlim(xl)
  def savetopdf (self,prefix,suffix,basedir='gif/',ylspec=(1,150),clr='r',drawavg=False,align='bywaveletpeak',lw=1,clravg='r',xl=None,ylavgCSD=None,ylavgMUA=None,dravgwspec=False,freqmin=1.0,freqmax=250.0,freqstep=1.0,drawfilt=False,filtclr='b'):
    # save a set of events to a single pdf file
    bdp = basedir+prefix
    for i,idx in enumerate(self.dframe.index): 
      self.clf()
      self.draw(idx,ylspec=ylspec,clr=clr,align=align,lw=lw,drawfilt=drawfilt,filtclr=filtclr)
      if xl is not None: self.set_xlim(xl)
      if drawavg:
        self.drawavgcsd(self.dframe.index,clr=clravg,lw=lw+1,align=align)
        self.drawavgmua(self.dframe.index,clr=clravg,lw=lw+1,align=align)
      savefig(bdp + str(int(idx)) + suffix + '_TMP_.pdf')
    if drawavg: # draw average spec,csd,mua by itself in a separate image/page
      self.clf()
      self.drawavgevent(self.dframe.index,clravg,freqmin,freqmax,freqstep,lw+1,align,ylspec,ylavgCSD,ylavgMUA,xl,dravgwspec)
      savefig(bdp + str(int(idx+1)) + suffix + '_TMP_.pdf')
    cmd = 'pdftk ' + bdp+'*'+suffix+'_TMP_.pdf cat output '+bdp+'_OUT_'+suffix+'.pdf; rm '+bdp+'*'+suffix+'_TMP_.pdf'
    print('cmd:',cmd)
    os.system(cmd)
    return cmd
    
#
def plotCV2Bands (dout,cvlim=(-0.25,5),winsz=10.0):
  for i,b in zip([0,1,2,3,4,5],['delta','theta','alpha','beta','gamma','hgamma']):
    subplot(6,2,i*2+1)
    title(b)
    plot([0,len(dout[b]['CV'])],[1,1],'k--',linewidth=1)
    plot(dout[b]['CV'])
    ylim(cvlim)
    ylabel(r'$CV^2$')
    subplot(6,2,i*2+2)
    arr = dout[b]['CV']
    N = round(mean(dout[b]['Count'])/winsz,2)
    st = b + ': ERate:' + str(N) + ' Hz' # title string
    if len(arr) > 0:
      st += r' , $CV^2$'
      st +=' mean:' + str(round(mean(arr),2))
      st += ' median:'+str(round(median(arr),2))
      hist(arr,density=True)
    title(st)
    xlabel(r'$CV^2$');
    xlim(cvlim)

#    
def formCV2ByBandLayer (din,lband = ['delta','theta','alpha','beta','gamma','hgamma']):
  dout = {}  
  for b in lband:
    dout[b] = {'avg':[],'err':[],'chan':[],'med':[]}
    lk = sort([x for x in din.keys() if type(x)==int]) # since now have lsidx in din (list of start indices)
    for k in lk:
      arr = din[k][b]['CV']
      if len(arr) > 0:
        dout[b]['chan'].append(k)
        dout[b]['med'].append(median(arr))
        dout[b]['avg'].append(mean(arr))
        dout[b]['err'].append(np.std(arr) / sqrt(len(arr)))
  return dout

#
def plotCV2BandsByLayer (din,bymean=True,cvlim=(-0.25,5),lband=['delta','theta','alpha','beta','gamma','hgamma'],lclr=['k','r','g','b','c','m']):
  minchan,maxchan=1e9,-1e9
  for c,b in zip(lclr,lband):
    lchan = din[b]['chan']
    print('lchan:',lchan)
    if len(lchan) > 0:
      minchan,maxchan = min(minchan,min(lchan)),max(maxchan,max(lchan))
  plot([minchan,maxchan],[1.0,1.0],'--',color='gray',linewidth=1)
  for c,b in zip(lclr,lband):
    lchan = din[b]['chan']
    if bymean:
      plot(lchan,np.array(din[b]['avg'])-din[b]['err'],c+'--',linewidth=1)
      plot(lchan,din[b]['avg'],c)
      plot(lchan,np.array(din[b]['avg'])+din[b]['err'],c+'--',linewidth=1)
    else:
      plot(lchan,din[b]['med'],c)
  ylim(cvlim); xlim((minchan-1,maxchan+1))
  if bymean:
    ylabel(r'Average $CV^2$')
  else:
    ylabel(r'Median $CV^2$')
  xlabel('Channel')
  gca().legend(handles=[mpatches.Patch(color=c,label=s) for c,s in zip(lclr,lband)])
    
#
def getEventT (din,chan,band):
  lsidx = din['lsidx']
  lwT = []
  for sidx,levent in zip(lsidx, din[chan][band]['levent']):
    lwT.append([])
    for event in levent:
      for T in linspace(event.minT,event.maxT,event.right-event.left+1):
        lwT[-1].append(T)
    lwT[-1].sort()
  return lwT

# histogram bin optimization
def hist_bin_opt (x, minbin=20, maxbin=600, spacing=10, N_trials=1):
  """ Returns optimal number of bins for histogram, using algorithm in Shimazaki and Shinomoto, Neural Comput, 2007
  x is input array
  """
  bin_checks = np.arange(minbin, maxbin, spacing)
  # bin_checks = np.linspace(150, 300, 16)
  costs = np.zeros(len(bin_checks))
  i = 0
  # this might be vectorizable in np
  for n_bins in bin_checks:
    # use np.histogram to do the numerical minimization
    pdf, bin_edges = np.histogram(x, n_bins)
    # calculate bin width
    # some discrepancy here but should be fine
    w_bin = np.unique(np.diff(bin_edges))
    if len(w_bin) > 1: w_bin = w_bin[0]
    # calc mean and var
    kbar = np.mean(pdf)
    kvar = np.var(pdf)
    # calc cost
    costs[i] = (2.*kbar - kvar) / (N_trials * w_bin)**2.
    i += 1
  # find the bin size corresponding to a minimization of the costs
  bin_opt_list = bin_checks[costs.min() == costs]
  bin_opt = bin_opt_list[0]
  return bin_opt

#
def modind (x):
  num = len(x)
  sigma = 0.0
  tot = sum(x)
  if tot <= 0.: return 0.
  for xi in x:
    if xi > 0:
      sigma -= (xi/tot)*log2(xi/tot) # entropy
  return 1.0 - sigma / log2(float(num)) # uniform distrib entropy is log(N), so this is diff from uniform entropy

#
def getdlmaw (dframe, dlms, lchan):
  dlmaw = {}; ddlphs = OrderedDict({chan:{} for chan in lchan})
  lfreq1 = arange(0.5,5.0,0.5)
  lfreq2 = arange(dbands['alpha'][0],dbands['gamma'][1]+2.0,2.0)
  for chan in lchan:
    for f1 in lfreq1:
      ddlphs[chan][f1]=OrderedDict()
      for f2 in lfreq2:
        ddlphs[chan][f1][f2]=[]
    lmaw = []
    for fdx1,freq1 in enumerate(lfreq1):
      ma = []
      colother = 'codelta'
      if freq1 >= dbands['theta'][0]: colother = 'cotheta'
      for freq2 in lfreq2:
        for windowidx in range(len(dlms[chan])):
          dfs = dframe[(dframe.peakF>=freq2-1.0) & (dframe.peakF<=freq2+1.0) & (dframe.Foct<1.5) & (dframe.chan==chan) & (dframe.windowidx==windowidx) & (dframe.colother==1)]
          if len(dfs) > 0:
            for idx in dfs.index:
              ddlphs[chan][freq1][freq2].append(dlms[chan][windowidx].PHS[fdx1,dframe.at[idx,'WaveletPeakIDX']])
        ma.append(modind(np.histogram(ddlphs[chan][freq1][freq2],bins=20,range=(-pi,pi))[0]))
        lmaw.append(ma)
    dlmaw[chan] = np.array(lmaw).T
  return dlmaw,ddlphs

# human ecog data
def IsHECoG (fn): return fn.count('NS')>0 and fn.count('hecog')>0

# get the output file paths stored in a dictionary
def getoutfilepaths (fn, basedir, getbipolar, winsz, medthresh, overlapth, useDynThresh, freqmin, freqmax, freqstep, \
                     dolaggedcoh, docfc, dolaggedcohnoband, dosim):
  if IsHECoG(fn):
    pass
  elif IsCortex(fn):
    basedir = os.path.join(basedir,'A1')
  else:
    basedir = os.path.join(basedir,'Thal')
  fbase = os.path.join(basedir,os.path.basename(fn))
  if dolaggedcoh or dolaggedcohnoband or docfc or dosim:
    pass
  else:
    fbase += '_bipolar_'+str(getbipolar)+'_winsz_'+str(winsz)+'_medthresh_'+str(medthresh)+'_overlapth_'+str(overlapth)
    fbase += '_useDynThresh_'+str(useDynThresh)+'_freqminmaxstep_'+str(freqmin)+'_'+str(freqmax)+'_'+str(freqstep)
  fout = {}
  fout['IEI'] = fbase + '_IEI.pkl'
  fout['ddcv2'] = fbase + '_ddcv2.pkl'
  fout['laggedcoh'] = fbase + '_laggedcoh.pkl'
  fout['laggedcohnoband'] = fbase + '_laggedcohnoband.pkl'  
  fout['cfc'] = fbase + '_cfc.pkl'
  fout['dcycprop'] = fbase + '_dcycprop.pkl'
  fout['dframe'] = fbase + '_dframe.pkl'
  return fout

# get lagged coherence dictionary
def getlaggedcohd (dat, sampr, lband, lchan, f_step=0.5, lwinsz = [3*3*2/dbands[b][0] for b in lband]):
  dllc = OrderedDict({chan:{} for chan in lchan})
  for chan in lchan:
    for b,w in zip(lband,lwinsz):
      print(chan,b,w)
      dllc[chan][b] = []
      for sidx in arange(0,dat.shape[1],int(w*sampr)):
        sig = dat[chan,sidx:sidx+int(w*sampr)]
        sig -= mean(sig)
        sig = sps.detrend(sig)
        dllc[chan][b].append(lagged_coherence(sig,(dbands[b][0],dbands[b][1]),sampr,f_step=f_step))
  return dllc

#
def getlaggedcohdnoband (dat, sampr, lchan, freqmin=0.5,freqmax=100.0,freqstep=1.0, ffctr=18.0):
  lfreq,lfreqwidth = getlfreqwidths(freqmin,freqmax,freqstep)
  ddllc = {'lchan':lchan}
  ddllc['lfreq']=lfreq
  ddllc['lfreqwidth']=lfreqwidth
  ddllc['ffctr']=ffctr
  for chan in lchan:
    ddllc[chan] = dllc = OrderedDict({f:([],f-fs/2.0,f,f+fs/2.0,fs) for f,fs in zip(lfreq,lfreqwidth)})
    for f,fs in zip(lfreq,lfreqwidth):
      w = ffctr/f
      print('chan:',chan,f-fs/2.,f,f+fs/2.,w)
      for sidx in arange(0,dat.shape[1],int(w*sampr)):
        sig = dat[chan,sidx:sidx+int(w*sampr)]
        sig -= mean(sig)
        sig = sps.detrend(sig)
        dllc[f][0].append(lagged_coherence(sig,(f-fs/2.,f+fs/2.),sampr,f_step=fs/2.0))
    dm = np.array([mean(dllc[f][0]) for f in lfreq])
    ds = np.array([std(dllc[f][0])/sqrt(len(dllc[f][0])) for f in lfreq])
    dllc['dm']=dm; dllc['ds']=ds
  return ddllc  

# get true,false positive rates -- intended for ERP detection
def getTPFP (xt, xr):
  TP = FP = FN = TN = 0.0
  for t,r in zip(xt,xr):
    if t and r: TP += 1
    if t and not r: FP += 1
    if not t and r: FN += 1
    if not t and not r: TN += 1
  TPR = TP / (TP+FN)
  FPR = FP / (FP+TN)      
  return TPR, FPR

#
def drawwavebycol (dframe, xsel, scol, sampr, gap=20., scaley=5, xl=None, yl=None,ylab='Num Cycles',alsofiltsigcor=True):
  if alsofiltsigcor:
    x = xsel.sort_values(by=[scol,'filtsigcor'],ascending=[True,False]) # xsel is a dataframe, with selection already performed; dframe is original data frame
  else:
    x = xsel.sort_values(scol) # xsel is a dataframe, with selection already performed; dframe is original data frame    
  dxpos = {} # current x position
  dcnt = {} # number of entries
  for idx in x.index:  
    ncycle = round(dframe.at[idx,scol]) # number of cycles (or scol)
    csdsig = dframe.at[idx,'CSDwvf'] # raw CSD signal
    filtsig = dframe.at[idx,'filtsig'] # filtered signal
    if ncycle not in dxpos: dxpos[ncycle] = 0.
    if ncycle not in dcnt: dcnt[ncycle] = 1
    tt = linspace(dxpos[ncycle], dxpos[ncycle]+1e3*len(csdsig)/sampr, len(csdsig))
    curry = ncycle * scaley # current y position
    if xl is not None and tt[-1] > xl[1]: continue
    plot(tt, normarr(csdsig) + curry,'r',linewidth=1)
    plot(tt,normarr(filtsig) + curry,'b')
    dxpos[ncycle] += 1e3*len(csdsig)/sampr + gap
  if xl is not None: xlim(xl)
  if yl is not None: ylim(yl)  
  ax=gca()
  ax.set_yticks(arange(0,scaley*max(list(dxpos.keys())),5))
  print(ax.get_yticks())
  ax.set_yticklabels([str(int(x)) for x in ax.get_yticks()])
  lytick = ax.get_yticklabels()
  print(lytick,lytick[0],lytick[1],lytick[2])
  ax.set_yticklabels([str(int(float(ytick.get_text()))/scaley) for ytick in lytick if len(ytick.get_text())>0])
  ylabel(ylab)
  xlabel('Time (ms)')

def stderr (x):
  try:
    return np.std(x)/sqrt(len(x))
  except:
    return 0.0

#  
def getburststats (eventt, lburstdur, ldf, timecheck, band):
  lcycnpeak,lcycnpeakS = [],[] # get some summary on num cycles
  lncycle,lncycleS = [],[]
  lpeakF,lpeakFS = [],[]
  lFoct,lFoctS = [],[]
  ldur,ldurS = [],[]
  lpow,lpowS = [],[]
  loscscore,loscscoreS = [],[]
  if timecheck:
    for burstdur,df in zip(lburstdur,ldf):
      dfs = df[(df.chan==0)&(df.band == band)]
      lcycnpeakT, lncycleT, lpeakFT, lFoctT, ldurT, lpowT, loscscoreT = [], [], [], [], [], [], []
      for t in eventt:
        dfs2 = dfs[(abs(dfs.minT-t)<100.0)]
        if len(dfs2) > 0:
          lcycnpeakT.append(mean(dfs2.cyc_npeak))
          lncycleT.append(mean(dfs2.ncycle))    
          lpeakFT.append(mean(dfs2.peakF))
          lFoctT.append(mean(dfs2.Foct))
          ldurT.append(mean(dfs2.dur))
          lpowT.append(mean(dfs2.avgpowevent))
          loscscoreT.append(mean(dfs2.OSCscore))
      print(burstdur,len(dfs2),mean(lpeakFT),mean(lncycleT),mean(ldurT),mean(lcycnpeakT),mean(lFoctT),mean(lpowT),mean(loscscoreT))
      lcycnpeak.append(mean(lcycnpeakT)); lcycnpeakS.append(stderr(lcycnpeakT))
      lncycle.append(mean(lncycleT)); lncycleS.append(stderr(lncycleT));
      lpeakF.append(mean(lpeakFT)); lpeakFS.append(stderr(lpeakFT))
      lFoct.append(mean(lFoctT)); lFoctS.append(stderr(lFoctT))
      ldur.append(mean(ldurT)); ldurS.append(stderr(ldurT))
      lpow.append(mean(lpowT)); lpowS.append(stderr(lpowT))
      loscscore.append(mean(loscscoreT)); loscscoreS.append(stderr(loscscoreT))
  else:
    for burstdur,df in zip(lburstdur,ldf):
      dfs = df[(df.chan==0)&(df.band == band)]
      print(burstdur,len(dfs),mean(dfs.peakF),mean(dfs.ncycle),mean(dfs.dur),mean(dfs.cyc_npeak),mean(dfs.Foct))
      lcycnpeak.append(mean(dfs.cyc_npeak)); lcycnpeakS.append(stderr(dfs.cyc_npeak))
      lncycle.append(mean(dfs.ncycle)); lncycleS.append(stderr(dfs.ncycle))
      lpeakF.append(mean(dfs.peakF)); lpeakFS.append(stderr(dfs.peakF))
      lFoct.append(mean(dfs.Foct)); lFoctS.append(stderr(dfs.Foct))
      ldur.append(mean(dfs.dur)); ldurS.append(stderr(dfs.dur))
      lpow.append(mean(dfs.avgpowevent)); lpowS.append(stderr(dfs.avgpowevent))
      loscscore.append(mean(dfs.OSCscore)); loscscoreS.append(stderr(dfs.OSCscore))
  dstat = {}
  dstat['lcycnpeak']=lcycnpeak; dstat['lcycnpeakS']=lcycnpeakS
  dstat['lncycle']=lncycle; dstat['lncycleS']=lncycleS
  dstat['lpeakF']=lpeakF; dstat['lpeakFS']=lpeakFS
  dstat['lFoct']=lFoct; dstat['lFoctS']=lFoctS
  dstat['ldur']=ldur; dstat['ldurS']=ldurS
  dstat['lpow']=lpow; dstat['lpowS']=lpowS
  dstat['loscscore']=loscscore; dstat['loscscoreS']=loscscoreS
  return dstat
                   
#
def noiseburstdetect (sampr=2e3,sigdur=21e3,burstfreq=10,burstamp=1.0,noiseamp=3.0,winsz=10,medthresh=4,overlapth=0.5,\
                      lburstdur=linspace(0.1, 1.5, 15), eventt=[1000, 4000, 7000, 11000, 14000, 17000],band='alpha',\
                      smooth=True,raiseamp=0.25,\
                      freqmin=0.25,freqmax=100.0,freqstep=0.25,\
                      usevoss=False,timecheck=True,usegauss=False,bgsig=None,\
                      mspecwidth=7.0):
  lchan = [0]
  ldf = [] # dataframe
  ldat = [] # data (signal)
  ldout = [] # ieistats out
  ldlms = [] # morlet spec output
  for burstdur in lburstdur:
    print('burstdur is',burstdur)
    times, sig = makeburstysig(sampr,sigdur,burstfreq,burstdur,burstamp=burstamp,noiseamp=noiseamp,eventt=eventt,\
                                smooth=smooth,raiseamp=raiseamp,usevoss=usevoss,usegauss=usegauss,bgsig=bgsig)
    dat = np.array([sig,sig])
    dout = getIEIstatsbyBand(dat,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,None,overlapth,getphase=True,savespec=True,mspecwidth=mspecwidth)
    df = GetDFrame(dout,sampr, dat, None, alignby='bywaveletpeak', haveMUA=False)
    addOSCscore(df)
    dlms={chan:dout[chan]['lms'] for chan in lchan};
    ldf.append(df)
    ldat.append(dat)
    ldout.append(dout)
    ldlms.append(dlms)
  dstat = getburststats(eventt, lburstdur, ldf, timecheck, band)
  return times,ldf,ldat,ldout,ldlms,dstat

#
def triangdetect (sampr=2e3,sigdur=21e3,triangamp=1.0,noiseamp=3.0,winsz=10,medthresh=4,overlapth=0.5,\
                  eventt=[1000, 4000, 7000, 11000, 14000, 17000],band=None,\
                  freqmin=0.25,freqmax=100.0,freqstep=0.25,usevoss=False,ltriangdur=[150.]):
  lchan = [0]
  ldf = [] # dataframe
  ldat = [] # data (signal)
  ldout = [] # ieistats out
  ldlms = [] # morlet spec output  
  for triangdur in ltriangdur:
    print('triangdur is',triangdur)
    times,sig = placetriang(sampr, sigdur, triangdur, eventt, noiseamp=noiseamp)
    dat = np.array([sig,sig])
    dout = getIEIstatsbyBand(dat,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,None,overlapth,getphase=True,savespec=True)
    df = GetDFrame(dout,sampr, dat, None, alignby='bywaveletpeak', haveMUA=False)
    dlms={chan:dout[chan]['lms'] for chan in lchan};
    ldf.append(df)
    ldat.append(dat)
    ldout.append(dout)
    ldlms.append(dlms)
  #
  lcycnpeak = [] # get some summary on num cycles
  lncycle = []
  lpeakF = []
  lFoct = []
  ldur = []
  for triangdur,df in zip(ltriangdur,ldf):
    if band is None:
      dfs = df[(df.chan==0)&(df.band == getband(1e3/triangdur))]
    else:
      dfs = df[(df.chan==0)&(df.band == band)]      
    print(triangdur,len(dfs),mean(dfs.peakF),mean(dfs.ncycle),mean(dfs.dur),mean(dfs.cyc_npeak),mean(dfs.Foct))
    lcycnpeak.append(mean(dfs.cyc_npeak))
    lncycle.append(mean(dfs.ncycle))
    lpeakF.append(mean(dfs.peakF))
    lFoct.append(mean(dfs.Foct))
    ldur.append(mean(dfs.dur))
  return times,ldf,ldat,ldout,ldlms,lcycnpeak,lncycle,lpeakF,lFoct,ldur


if __name__ == "__main__":
  narg = len(sys.argv)
  basedir = 'data/spont/oscout/'
  useCSD = True
  getbipolar = False
  winsz = 10
  medthresh = 4.0
  overlapth = 0.5
  useDynThresh = 0
  dorun = False
  doquit = False
  freqmin = 0.25 # 0.5
  freqmax = 250.0
  freqstep = 0.25 # 0.5
  dolaggedcoh = dolaggedcohnoband = False
  docfc = False
  mspecwidth = 7.0
  noiseamp = noiseampCSD # default is to use CSD, unless user specifies to use BIP
  dosim = 0
  if narg>=3: getbipolar = bool(int(sys.argv[2]))
  print(sys.argv)
  if narg>=17: dosim = int(sys.argv[16])  
  if narg>=2:
    fn = sys.argv[1]
    if dosim:
      samprds = 1e3
    else:
      samprds = getdownsampr(fn)
    print(fn,getorigsampr(fn),'samprds:',samprds)
    if getorigsampr(fn) < 30e3:
      print('skipping low sampr file', fn)
      quit()    
    if getbipolar:
      sampr,dat,dt,tt,CSD,MUA,BIP = loadfile(fn,samprds,getbipolar=getbipolar)
      dat = BIP
      noiseamp = noiseampBIP
    else:
      sampr,dat,dt,tt,CSD,MUA = loadfile(fn,samprds,getbipolar=getbipolar)
      dat = CSD
  if narg>=4: medthresh = float(sys.argv[3])
  if narg>=5: winsz = int(sys.argv[4])
  if narg>=6: overlapth = float(sys.argv[5])
  if narg>=7: freqmin = float(sys.argv[6])
  if narg>=8: freqmax = float(sys.argv[7])
  if narg>=9: freqstep = float(sys.argv[8])  
  if narg>=10: useDynThresh = int(sys.argv[9])
  if narg>=11: dorun = int(sys.argv[10])
  if narg>=12: doquit = int(sys.argv[11])
  if narg>=13: dolaggedcoh = int(sys.argv[12])
  if narg>=14: mspecwidth = float(sys.argv[13])
  if narg>=15: docfc = int(sys.argv[14])
  if narg>=16: dolaggedcohnoband = int(sys.argv[15])
  try:
    print('fn is',fn,'getbipolar:',getbipolar,'winsz:', winsz, 'medthresh:', medthresh,'useCSD:',useCSD,'useDynThresh:',useDynThresh,'dolaggedcoh',dolaggedcoh,'mspecwidth:',mspecwidth,'docfc:',docfc,'dolaggedcohnoband:',dolaggedcohnoband,'dosim:',dosim)
  except:
    pass
  if dorun:
    ar = getAreaCode(fn)
    lchan = []
    if ar == 1:
      dbpath = 'data/spont/A1/19apr4_A1_spont_LayersForSam.csv'
      if fn.count('ERP'): dbpath = 'data/ERP/A1/19may9_A1_ERP_Layers.csv'
      s2,g,i1 = getflayers(fn,dbpath=dbpath,abbrev=True)
      if s2 == -1: print('channels unknown!')
      else: lchan = [s2,g,i1]
    else: # for non cortical just pick the middle channel
      lchan = [int(dat.shape[0]/2)] # for
    if len(lchan) > 0:
      if dolaggedcoh: basedir = 'data/spont/laggedcoh'
      if dolaggedcohnoband: basedir = 'data/spont/laggedcohnoband'
      # if docfc: basedir = 'data/spont/cfc'
      fout = getoutfilepaths(fn, basedir, getbipolar, winsz, medthresh, overlapth, useDynThresh, freqmin, freqmax, freqstep, dolaggedcoh, docfc,dolaggedcohnoband,dosim)
      print('fout:',fout)
      if docfc:
        print('running CFC')
        if not os.path.isfile(fout['dframe']):
          print('did not find dframe',fout['dframe'])
          quit()
        dframe = pickle.load(open(fout['dframe'],'rb'))
        lchan = [s2,g,i1]; chan = s2  
        sig = CSD[chan,:]
        dlms={}; dlnoise={}; dlsidx={}; dleidx={}
        for chan in lchan:
          lms,lnoise,lsidx,leidx = getmorletwin(CSD[chan,:],int(winsz*sampr),sampr,freqmin=0.5,freqmax=6.5,freqstep=0.25,getphase=True,noiseamp=noiseamp)
          dlms[chan]=lms; dlnoise[chan]=lnoise; dlsidx[chan]=lsidx; dleidx[chan]=leidx
        dlmaw,ddlphs = getdlmaw(dframe,dlms,lchan)
        tmpout = {'dlmaw':dlmaw,'ddlphs':ddlphs}
        pickle.dump(tmpout,open(fout['cfc'],'wb'))
      elif dolaggedcoh:
        if not os.path.isfile(fout['laggedcoh']):
          print('running lagged coherence')
          #lwinsz = [3*3*2/dbands[b][0] for b in lband]
          lwinsz=[72.0, 32.0, 25.6, 10.7, 2.8, 1.2]
          #lwinsz = [72.0 for b in lband]
          dllc = getlaggedcohd(dat,sampr,lband,lchan,f_step=0.5,lwinsz=lwinsz)
          pickle.dump(dllc,open(fout['laggedcoh'],'wb'))
        else:
          print('already ran',fout['laggedcoh'])
      elif dolaggedcohnoband:
        if not os.path.isfile(fout['laggedcohnoband']):
          print('running lagged coherence without bands')
          ddllc = getlaggedcohdnoband(dat,sampr,lchan)
          pickle.dump(ddllc,open(fout['laggedcohnoband'],'wb'))
        else:
          print('already ran',fout['laggedcohnoband'])
      elif dosim: # run simulation - put alpha signals on top of CSD background        
        pass
      else: # this path is the main event/oscillation analyses
        if not os.path.isfile(fout['ddcv2']): # if did not already run/save this file
          if os.path.isfile(fout['IEI']):
            dout = pickle.load(open(fout['IEI'],'rb'))
            print('loaded IEI file',fout['IEI'])
          else:
            dout = getIEIstatsbyBand(dat,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,MUA,overlapth,getphase=True,savespec=False,useDynThresh=useDynThresh,useloglfreq=False,noiseamp=noiseamp)
            if ar == 1: dout['s2'],dout['g'],dout['i1']=s2,g,i1
            pickle.dump(dout,open(fout['IEI'],'wb'))
          if os.path.isfile(fout['dframe']):
            dframe = pickle.load(open(fout['dframe'],'rb'))
          else:
            dframe = GetDFrame(dout,sampr, CSD, MUA, alignby='bywaveletpeak', haveMUA=True)
            dframe.to_pickle(fout['dframe'])
          ddcv2={}
          for chan in lchan:
            print('chan is',chan,'for dCV2')
            ddcv2[chan] = getvarwindCV2(dframe,chan,lwinsz=[72.0, 32.0, 25.6, 10.7, 2.8, 1.2])
          pickle.dump(ddcv2,open(fout['ddcv2'],'wb'))
        else:
          print('already ran all files in ', fout)        
    else:
      print('no channels used; exiting')
    if doquit: quit()
    
