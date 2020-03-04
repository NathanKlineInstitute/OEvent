from filter import lowpass,bandpass
import numpy as np

# lowpass filter the items in lfps. lfps is a list or numpy array of LFPs arranged spatially by column
def getlowpass (lfps,sampr,maxf):
  datlow = []
  for i in range(len(lfps[0])): datlow.append(lowpass(lfps[:,i],maxf,df=sampr,zerophase=True))
  datlow = np.array(datlow)
  return datlow

# bandpass filter the items in lfps. lfps is a list or numpy array of LFPs arranged spatially by column
def getbandpass (lfps,sampr,minf=0.05,maxf=300):
  datband = []
  for i in range(len(lfps[0])): datband.append(bandpass(lfps[:,i],minf,maxf,df=sampr,zerophase=True))
  datband = np.array(datband)
  return datband

# Vaknin correction for CSD analysis (MS implementation)
# Allows CSD to be performed on all N contacts instead of N-2 contacts
# See Vaknin et al (1989) for more details
def Vaknin (x):
  # Preallocate array with 2 more rows than input array
  x_new = np.zeros((x.shape[0]+2, x.shape[1]))
  # print x_new.shape
  # Duplicate first and last row of x into first and last row of x_new
  x_new[0, :] = x[0, :]
  x_new[-1, :] = x[-1, :]
  # Duplicate all of x into middle rows of x_neww
  x_new[1:-1, :] = x
  return x_new

#
def removemean (x, ax=1):
  mean = np.mean(x, axis=ax, keepdims=True)
  x -= mean
  print(np.mean(x, axis=ax, keepdims=True))

# get CSD - first do a lowpass filter. lfps is a list or numpy array of LFPs arranged spatially by column
# spacing_um is electrode's contact spacing in units of micron
# returns CSD in units of mV/mm**2 (assuming lfps are in mV)
def getCSD (lfps,sampr,spacing_um=100.0,minf=0.05,maxf=300,norm=True,vaknin=False):
  datband = getbandpass(lfps,sampr,minf,maxf)
  #datband = getlowpass(lfps,sampr,maxf)
  if datband.shape[0] > datband.shape[1]: # take CSD along smaller dimension
    ax = 1
  else:
    ax = 0
  # can change to run Vaknin on bandpass filtered LFPs before calculating CSD, that
  # way would have same number of channels in CSD and LFP (but not critical, and would take more RAM);
  # also might want to subtract mean of each channel before calculating the diff(diff) ?
  #
  if vaknin: datband = Vaknin(datband)
  if norm: removemean(datband,ax=ax)
  # when drawing CSD make sure that negative values (depolarizing intracellular current) drawn in red,
  # and positive values (hyperpolarizing intracellular current) drawn in blue
  spacing_mm = spacing_um/1000 # spacing in mm
  CSD = -np.diff(datband,n=2,axis=ax)/spacing_mm**2 # now each column (or row) is an electrode -- CSD along electrodes
  """
  CSD = Vaknin(CSD)
  """  
  return CSD

# get bipolar waveforms - first do a lowpass filter. lfps is a list or numpy array of LFPs arranged spatially by column
# spacing_um is electrode's contact spacing in units of micron
# returns bipolar signal in units of mV/mm (assuming lfps are in mV)
def getBipolar (lfps,sampr,spacing_um=100.0,minf=0.05,maxf=300,norm=True,vaknin=False):
  datband = getbandpass(lfps,sampr,minf,maxf)
  if datband.shape[0] > datband.shape[1]: # take CSD along smaller dimension
    ax = 1
  else:
    ax = 0
  if vaknin: datband = Vaknin(datband)    
  if norm: removemean(datband,ax=ax)    
  # can change to run Vaknin on bandpass filtered LFPs before calculating bipolar waveforms, that
  # way would have same number of channels in bipolar and LFP (but not critical, and would take more RAM);
  # also might want to subtract mean of each channel before calculating the diff ?
  spacing_mm = spacing_um/1000 # spacing in mm
  bipolar = -np.diff(datband,n=1,axis=ax)/spacing_mm # now each column (or row) is an electrode -- bipolar along electrodes
  return bipolar

# get MUA - first do a bandpass filter then rectify. 
#  lfps is a list or numpy array of LFPs arranged spatially by column
def getMUA (lfps,sampr,minf=300,maxf=5000):
  datband = getbandpass(lfps,sampr,minf,maxf)
  return np.abs(datband)

# get a signal that estimates depolarization using csd and mua (method has not been validated)
def getdepcsdmua (csd,mua):
  out = zeros((len(csd),))
  dm = diff(mua)
  out[1:] = csd[1:] * dm
  C = csd[1:]
  out1 = out[1:]
  out = np.where((C<0)&(dm>0),-1*out1,0)
  out = list(out)
  out.insert(0,0)
  return np.array(out)

# get a signal that estimates hyperpolarization using csd and mua (method has not been validated)
def gethypcsdmua (csd,mua):
  out = zeros((len(csd),))
  dm = diff(mua)
  out[1:] = csd[1:] * dm
  C = csd[1:]
  out1 = out[1:]
  out = np.where((C>0)&(dm<0),out1,0)
  out = list(out)
  out.insert(0,0)
  return np.array(out)
