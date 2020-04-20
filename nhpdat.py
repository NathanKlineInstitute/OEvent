import sys
import os
import h5py
import numpy as np
from csd import *
from filter import downsample
from collections import OrderedDict

# frequency band ranges for monkeys
def makedbands ():
  dbands = OrderedDict()
  gapHz = 1
  dbands['delta'] = [0.5,3.0 + gapHz]
  dbands['theta'] = [4,8 + gapHz]
  dbands['alpha'] = [9,14 + gapHz]
  dbands['beta'] = [15,28 + gapHz]
  dbands['gamma'] = [29,80 + gapHz]
  dbands['hgamma'] = [81,200 + gapHz]
  return dbands

dbands = makedbands()
lband = list(dbands.keys())

#
def getband (f):
  for k in ['delta','theta','alpha','beta','gamma','hgamma']:
    if f >= dbands[k][0] and f < dbands[k][1]:
      return k
  return 'unknown'

if int(sys.version[0])==2:
  # channel/layer info - applies to all recordings?
  def makeDLayers ():
    dlyrL,dlyrR={},{}
    dlyrL['supra'] = [4,5,6,7,8,9]
    dlyrL['gran'] = [12,13,14]
    dlyrL['infra'] = [16,17,18,19]
    dlyrR['supra'] = [5,6,7,8]
    dlyrR['gran'] = [10,11]
    dlyrR['infra'] = [13,14,15,16]
    for D in [dlyrL,dlyrR]:
      lk = D.keys()
      for k in lk:
        for c in D[k]:
          D[c] = k
    return dlyrL,dlyrR
  dlyrL,dlyrR = makeDLayers()

# if recording is from the left side - this is probably not generally applicable for all PL data
def leftname (fname):
  f = None
  if fname.count("/"):
    f = fname.split("/")[1]
  else:
    f = fname
  f = f.split("-")
  if int(f[0]) == 1: return True
  return False

# matching pair of data-files - this is probably not generally applicable for all PL data
def namepair (f1,f2):
  if f1.count("spont") != f2.count("spont"): return False
  if f1.count("1-") == f2.count("1-"): return False
  f1sub,f2sub = f1.split("-")[1],f2.split("-")[1]
  if f1sub[0:2] != f2sub[0:2]: return False
  num1,num2=int(f1sub[2:len(f1sub)]),int(f2sub[2:len(f2sub)])
  if abs( num1 - num2 ) != 1: return False
  lf1,lf2=leftname(f1),leftname(f2)
  if lf1 and num1 > num2: return False # left side should have lower number
  if lf2 and num2 > num1: return False # left side should have lower number
  return True

# return first line matching s if it exists in file fn
def grepstr (fn, s):
  try:
    fp = open(fn,'r')
    lns = fp.readlines()
    for ln in lns:
      if ln.count(s) > 0:
        fp.close()
        return ln.strip()
    fp.close()
  except:
    pass
  return False

# find the csv path with layer information if it's in the same dir as fn
def findcsvdbpath (fn):
  basedir = os.path.split(fn)[0]
  for f in os.listdir(basedir):
    if f.endswith('.csv') and f.count('Layers') or f.count('layers'):
      return os.path.join(basedir,f)
  return None

#
def monoinc (lx):
  if len(lx) < 2: return True
  for i,j in zip(lx,lx[1:]):
    if i > j:
      return False
  return True

# this function gets the CSD channel ranges for the .mat cortical recording:
# s1: supragranular source
# s2: supragranular sink
# g: granular sink
# i1: infragranular sink
# i2: infragranular source
# each of these values have a range, by default will pick the middle value as s1,s2,g,i1,i2
#
# note that indices in dbpath file are Matlab based so subtracts 1 first
# since not all files have layers determined, returns empty values (-1) when not found
# when abbrev==True, only get s2,g,i1
def getflayers (fn, dbpath='data/spont/A1/19apr4_A1_spont_LayersForSam.csv',getmid=True,abbrev=False):
  if dbpath is None or len(dbpath)==0: dbpath = findcsvdbpath(fn)
  s = grepstr(dbpath,os.path.split(fn)[-1])
  if s == False:
    if abbrev:
      return [-1 for i in range(3)]
    else:
      return [-1 for i in range(5)]
  ls = s.split(',')
  print(ls)
  try:
    lint = [int(x)-1 for x in ls[2:]]
    if not monoinc(lint):
      if abbrev:
        return [-1 for i in range(3)]
      else:
        return [-1 for i in range(5)]      
    s1low,s1high,s2low,s2high,glow,ghigh,i1low,i1high,i2low,i2high = lint
    if getmid:
      s1 = int((s1low+s1high)/2.0)
      s2 = int((s2low+s2high)/2.0)
      g = int((glow+ghigh)/2.0)
      i1 = int((i1low+i1high)/2.0)
      i2 = int((i2low+i2high)/2.0)
      print(s1low,s1high,s2low,s2high,glow,ghigh,i1low,i1high,i2low,i2high,s1,s2,g,i1,i2)
      if abbrev:
        return s2,g,i1
      else:
        return s1,s2,g,i1,i2
    else:
      return s1low,s1high,s2low,s2high,glow,ghigh,i1low,i1high,i2low,i2high
  except:
    if abbrev:
      return [-1 for i in range(3)]
    else:
      return [-1 for i in range(5)]

# get simple value from the hdf5 (mat) file
def gethdf5val (fn,key):
  fp = h5py.File(fn,'r') # open the .mat / HDF5 formatted data
  val = fp[key][0][0] # sampling rate
  fp.close()
  return val

# get original sampling rate for LFP in the .mat file
def getorigsampr (fn): return gethdf5val(fn,'craw/adrate')

# get the stimulus intensity
def getStimIntensity (fn): return gethdf5val(fn,'params/filedata/intensity')

# get type of stimulus applied
def getStimType (fn): return int(gethdf5val(fn,'params/filedata/stim'))

# check if broadband noise (BBN) stimulus was used
def hasBBNStim (fn): return gethdf5val(fn,'params/filedata/stim') == 1

# check if click stimulus was used
def hasClickStim (fn): return gethdf5val(fn,'params/filedata/stim') == 5

# get downsampling rate that would allow 5000  Hz for MUA
def getdownsampr (fn):
  origsampr = getorigsampr(fn)
  if int(origsampr) == 44000:
    return 11000.0
  elif int(origsampr) == 20000:
    return 10000.0
  else:
    return 0
  
# read the matlab .mat file and return the sampling rate and electrophys data
# note that the local field potential data is stored in microVolts in the .mat
# files but is converted to milliVolts before returning from this function
def rdmat (fn,samprds=0):  
  fp = h5py.File(fn,'r') # open the .mat / HDF5 formatted data
  sampr = fp['craw']['adrate'][0][0] # sampling rate
  print('fn:',fn,'sampr:',sampr,'samprds:',samprds)
  dt = 1.0 / sampr # time-step in seconds
  dat = fp['craw']['cnt'] # cnt record stores the electrophys data
  npdat = np.zeros(dat.shape)
  tmax = ( len(npdat) - 1.0 ) * dt # use original sampling rate for tmax - otherwise shifts phase
  dat.read_direct(npdat) # read it into memory; note that this LFP data usually stored in microVolt
  npdat *= 0.001 # convert microVolt to milliVolt here
  fp.close()
  if samprds > 0.0: # resample the LFPs
    dsfctr = sampr/samprds
    dt = 1.0 / samprds
    siglen = max((npdat.shape[0],npdat.shape[1]))
    nchan = min((npdat.shape[0],npdat.shape[1]))
    npds = [] # zeros((int(siglen/float(dsfctr)),nchan))
    # print dsfctr, dt, siglen, nchan, samprds, ceil(int(siglen / float(dsfctr))), npds.shape
    for i in range(nchan): 
      print('resampling channel', i)
      npds.append(downsample(npdat[:,i], sampr, samprds))
    npdat = np.array(npds)
    npdat = npdat.T
    sampr = samprds
  tt = np.linspace(0,tmax,len(npdat)) # time in seconds
  return sampr,npdat,dt,tt # npdat is LFP in units of milliVolt

#
def getHDF5values (fn,key):
  fp = h5py.File(fn,'r')
  hdf5obj = fp[key]
  x = np.array(fp[hdf5obj.name])
  val = [y[0] for y in fp[x[0,0]].value]
  fp.close()
  return val

# get analog stimulus trigger times
def getTriggerTimes (fn):
  fp = h5py.File(fn,'r')
  hdf5obj = fp['trig/anatrig']
  x = np.array(fp[hdf5obj.name])
  val = [y[0] for y in fp[x[0,0]].value]
  fp.close()
  return val  

# get stimulus identifiers
def getTriggerIDs (fn):
  fp = h5py.File(fn,'r')
  hdf5obj = fp['trig/ttype']
  x = np.array(fp[hdf5obj.name])
  val = fp[x[0,0]].value[0] 
  val = [int(x) for x in val]
  fp.close()
  return val

# area codes
def setupdArea ():
  dArea = {1: 'A1',
           2: 'Belt',
           3: 'MGB',
           4: 'LGN',
           5: 'MedialPulvinar',
           6: 'Pulvinar',
           7: 'TRN',
           8: 'Motor',
           9: 'Striatum',
           33:'MGBv'}
  dk = list(dArea.keys())
  for k in dk:
    if type(k)==int:
      dArea[dArea[k]]=k
  return dArea

dArea = setupdArea()

# 0 means thalamus, 1 means A1, 2 means belt
def getAreaCode (fn):
  fp = h5py.File(fn,'r')
  code = int(fp['params']['filedata']['area'][0][0])
  fp.close()
  return code

# return True iff file recorded from neocortex
def IsCortex (fn):
  try:
    ac = getAreaCode(fn)
    dc = dArea[ac]
    return dc == 'A1' or dc == 'Belt' or dc == 'Motor'
  except:
    return False

# return True iff file
def IsThal (fn):
  try:
    ac = getAreaCode(fn)
    dc = dArea[ac]
    return dc == 'MGB' or dc == 'MGBv' or dc == 'Pulvinar' or dc == 'LGN' or dc == 'TRN' or dc == 'MedialPulvinar'
  except:
    return False

# get best frequency from the recording (frequency which area responds to most strongly)
def getBestFreq (fn):
  fp = h5py.File(fn,'r')
  code = int(fp['params']['filedata']['bf'][0][0])
  fp.close()
  if code > 0:
    return A1bestf[code-1] # cortical best frequency
  return code # 0 means thalamus where best freq varies with electrode

#
def loadfile (fn,samprds,getbipolar=False,spacing_um=100.0):
  # load a data file and get the CSD (mV/mm**2); samprds is downsampling rate; spacing_um is contact spacing in microns
  sampr,dat,dt,tt=rdmat(fn,samprds=samprds) # LFP signals returned in milliVolt
  CSD = getCSD(dat,sampr,spacing_um=spacing_um) # why make loadfile depend on getCSD? one function call but then more dependencies ... 
  # Note: index 0 of CSD comes from index 0,1,2 of LFP; so add 1 to index of CSD to get index into MUA
  # MUA index to CSD index, sub 1
  MUA=getMUA(dat,sampr)
  #divby = getorigsampr(fn) / samprds
  #trigtimes = [int(round(x)) for x in np.array(getTriggerTimes(fn)) / divby] # div by 22 since downsampled by factor of 22
  #trigIDs = getTriggerIDs(fn)
  if getbipolar:
    BIP=getBipolar(dat,sampr)
    return sampr,dat,dt,tt,CSD,MUA,BIP
  else:
    return sampr,dat,dt,tt,CSD,MUA

#
def remaptrigIDs (x):
  remap = {1:1, 2:2, 9:3, 10:4, 3:5, 4:6, 11:7, 12:8, 5:9, 6:10, 13:11, 14:12, 7:13, 8:14, 15:15, 16:16}
  return [remap[v] for v in x]

# A1/Belt best frequencies (params.filedata.bf tells best frequency in Hz index into this array coded from 1-14)
A1bestf = [353.553390593274,500.000000000000,707.106781186547,1000.00000000000,1414.21356237309,2000.00000000000,2828.42712474619,4000.00000000000,5656.85424949238,8000.00000000000,11313.7084989848,16000.0000000000,22627.4169979695,32000.0000000000]

# get the monkey name from the file path
def getmonkeyname (fn): return fn.split(os.path.sep)[-1].split('-')[-1].split('@')[0][0:2]

# get the experiment number from the file path
def getexperimentnums (fn): return fn.split(os.path.sep)[-1].split('-')[-1].split('@')[0][2:]
  
# get the experiment number file code from the file path
def getexperimentnumfilecode (fn): return getexperimentnums(fn)[-3:]

# get the experiment number prefix from the file path
def getexperimentnumprefix (fn): return getexperimentnums(fn)[0:-3]

# find closest file in the database matching fntarg and return the match along with its filecode distance
def closestfile (fntarg, dbpath='data/spont/A1/19apr4_A1_spont_LayersForSam.csv', dbdir='data/spont/A1'):
  fp = open(dbpath,'r')
  lns = fp.readlines()
  fp.close()
  monkeyname = getmonkeyname(fntarg)
  expfilecode = getexperimentnumfilecode(fntarg)
  expnumprefix = getexperimentnumprefix(fntarg)
  arcode = getAreaCode(fntarg)
  lns = [s.strip() for s in lns]
  score = 1e9
  bestfn = ''
  for ln in lns:
    fndb = ln.split(',')[0] # filename in the csv database
    if getmonkeyname(fndb) == monkeyname and expnumprefix == getexperimentnumprefix(fndb) and \
       getAreaCode(os.path.join(dbdir,fndb)) == arcode:
      tmpscore = abs(int(expfilecode) - int(getexperimentnumfilecode(fndb)))
      if tmpscore < score:
        score = tmpscore
        bestfn = fndb
  return bestfn,score

