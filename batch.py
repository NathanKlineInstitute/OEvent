"""
OEvent: Oscillation event detection and feature analysis.
batch.py - runs analysis on a set of files
Written by Sam Neymotin (samuel.neymotin@nki.rfmh.org)
References: Taxonomy of neural oscillation events in primate auditory cortex
https://doi.org/10.1101/2020.04.16.045021
"""
from pylab import *
import sys,os,numpy,subprocess
from math import ceil
import multiprocessing
import matplotlib.gridspec as gridspec
import shutil
from subprocess import Popen, PIPE, call
import pickle

myhost = os.uname()[1]
defQSZ = 1 # default queue size for batch
if myhost == 'zn': defQSZ = 4 # if on zn, have more RAM so bigger qsz

# append line s to filepath fn
def appline (s,fn):
  fp = open(fn,"a"); fp.write(s + "\n"); fp.close()

# check that the batch dir exists
def checkdir (d):
  try:
    if not os.path.exists(d): os.mkdir(d)
    return True
  except:
    print("could not create directory :",d)
    return False

# make a list of the sims that have already had their output saved, can then
# pass it into batchRun to skip those sims
def getSkipList (whichParams):
  lsec,lopt,lval = whichParams()
  sidx,lskip = -1,[]
  for i in range(len(lopt[0])):
    if lopt[0][i] == 'simstr':
      sidx = i
      break
  if sidx == -1:
    print("no simstr found!")
    return None
  for i in range(len(lval)):
    if os.path.exists("./data/" + lval[i][sidx] + "_.npz"):
      lskip.append(i)
  return lskip

# run a batch using multiprocessing - which calls mpiexec - single simulation then split across nodes
#  based on http://www.bryceboe.com/2011/01/28/the-python-multiprocessing-queue-and-large-objects/
def batchRun (lmyargs,blog,skip=[],qsz=defQSZ,bdir="./batch",pyf="load.py"):
  if not checkdir(bdir): return False
  jobs = multiprocessing.Queue()
  shutil.copy(pyf, bdir) # make a backup copy of py file -- but use local copy since has dependencies
  def myworker (jobs):
    while True:
      scomm = jobs.get()
      if scomm == None: break
      print("worker starting : " , scomm)
      os.system(scomm) # worker function, invoked in a process.
  for i in range(len(lmyargs)):
    if i in skip: continue
    cmd = "python3 " + pyf + " ";
    args = lmyargs[i]
    for arg in args: cmd += arg + ' '
    print('command is',cmd)
    appline(cmd,blog)
    jobs.put(cmd)
  workers = []
  for i in range(qsz):
    jobs.put(None)
    tmp = multiprocessing.Process(target=myworker, args=(jobs,))
    tmp.start()
    workers.append(tmp)
  for worker in workers: worker.join()
  return jobs.empty()

#
def getfilesext (basedir,ext):
  lfn = os.listdir(basedir)
  lfn = [os.path.join(basedir,x) for x in lfn if x.endswith(ext)]
  return lfn

def eventbatch ():
  print('running batch')
  lmedthresh = [4.0]
  lwinsz = [10]
  loverlapth = [0.5]
  lbipolar = [0] # [0, 1]
  llarg = []
  lfnA = getfilesext('data/spont/A1','.mat')
  #lfnB = getfilesext('data/spont/Thal','.mat')
  lfn = [x for x in lfnA]
  #for x in lfnB: lfn.append(x)
  freqmin = 0.25 #0.5
  freqmax = 250.0
  freqstep = 0.25 #0.5
  useDynThresh = 0
  dorun = doquit = 1
  for overlapth in loverlapth:  
    for medthresh in lmedthresh:
      for winsz in lwinsz:
        for bipolar in lbipolar:
          for fn in lfn:
            larg = [fn,str(bipolar),str(medthresh),str(winsz),str(overlapth),\
                  str(freqmin),str(freqmax), str(freqstep), str(useDynThresh),\
                  str(dorun), str(doquit)]
            llarg.append(larg)
  batchRun(llarg,'batch.log')

def simbatch (): # not used currently - did not finish setup of load.py for this
  print('running batch')
  lmedthresh = [4.0]
  lwinsz = [10]
  loverlapth = [0.5]
  lbipolar = [0]
  llarg = []
  lfnA = getfilesext('data/spont/A1','.mat')
  lfn = [x for x in lfnA]
  freqmin = 0.25 
  freqmax = 250.0
  freqstep = 0.25 
  useDynThresh = 0
  dorun = doquit = 1
  for overlapth in loverlapth:  
    for medthresh in lmedthresh:
      for winsz in lwinsz:
        for bipolar in lbipolar:
          for fn in lfn:
            larg = [fn,str(bipolar),str(medthresh),str(winsz),str(overlapth),\
                  str(freqmin),str(freqmax), str(freqstep), str(useDynThresh),\
                  str(dorun), str(doquit)]
            llarg.append(larg)
  batchRun(llarg,'batch.log')
  

def laggedcohbatch ():
  medthresh = 4.0
  winsz = 10
  overlapth = 0.5
  llarg = []
  lfnA = getfilesext('data/spont/A1','.mat')
  lfnB = getfilesext('data/spont/Thal','.mat')
  lfn = [x for x in lfnA]
  for x in lfnB: lfn.append(x)
  freqmin = 0.5
  freqmax = 250.0
  freqstep = 0.5
  useDynThresh = 0
  dorun = doquit = dolaggedcoh = 1
  bipolar = 0
  for fn in lfn:
    larg = [fn,str(bipolar),str(medthresh),str(winsz),str(overlapth),\
            str(freqmin),str(freqmax), str(freqstep), str(useDynThresh),\
            str(dorun), str(doquit), str(dolaggedcoh)]
    llarg.append(larg)
  batchRun(llarg,'batch.log',qsz=defQSZ) 

def laggedcohnobandbatch ():
  medthresh = 4.0
  winsz = 10
  overlapth = 0.5
  llarg = []
  lfnA = getfilesext('data/spont/A1','.mat')
  lfnB = getfilesext('data/spont/Thal','.mat')
  lfn = [x for x in lfnA]
  for x in lfnB: lfn.append(x)
  freqmin = 0.5
  freqmax = 250.0
  freqstep = 0.5
  useDynThresh = 0
  dorun = doquit = 1
  dolaggedcoh = 0
  mspecwidth = 7.0
  docfc = 0
  dolaggedcohnoband = 1
  bipolar = 0
  for fn in lfn:
    larg = [fn,str(bipolar),str(medthresh),str(winsz),str(overlapth),\
            str(freqmin),str(freqmax), str(freqstep), str(useDynThresh),\
            str(dorun), str(doquit), str(dolaggedcoh),str(mspecwidth),str(docfc),str(dolaggedcohnoband)]
    llarg.append(larg)
  batchRun(llarg,'batch.log',qsz=int(defQSZ*1.5))
  
#
def loadddcv2 (skipcsd=False,skipbipolar=False,lar=['A1','STG']):
  from nhpdat import getflayers
  ddcv2={}
  for ar in lar:
    ddcv2[ar]={}
    if ar == 'A1' or ar == 'Thal':
      bdir = 'data/spont/oscout/'+ar
    else:
      bdir = 'data/hecog/spont/oscout/'
    lfn = os.listdir(bdir)
    for fn in lfn:
      if fn.endswith('ddcv2.pkl'):
        if skipbipolar and fn.count('bipolar_True') > 0: continue
        if skipcsd and fn.count('bipolar_False') > 0: continue
        if ar == 'A1':
          fnorig = 'data/spont/'+ar + '/' + fn.split('_bipolar')[0]
          #print(fnorig)
          s2,g,i1 = getflayers(fnorig,abbrev=True)
          if s2 == -1: continue
        ddcv2[ar][fn] = pickle.load(open(bdir+'/'+fn,'rb'))
  return ddcv2

#
def plotddcv2byband (ddcv2,ar,dkey,skipbipolar=True,clr='k',bins=30,xlab=r'$CV^2$',xl=(0,3),histtype='bar',lw=4):
  lband = ['delta','theta','alpha','beta','gamma','hgamma']
  lval = []
  for bdx,b in enumerate(lband):
    v = []
    for k in ddcv2[ar].keys():
      if type(k)==str:
        if skipbipolar and k.count('bipolar_True') > 0: continue
      dcv2 = ddcv2[ar][k]
      lchan = list(dcv2.keys())
      lchan.sort()
      for c in lchan:
        if type(dcv2[c][b][dkey])==list:
          if len(dcv2[c][b][dkey])>0 and type(dcv2[c][b][dkey][0])==list:
            for l in dcv2[c][b][dkey]:
              for x in l:
                if not isnan(x):
                  v.append(x)
          else:
            for x in dcv2[c][b][dkey]:
              if not isnan(x):
                v.append(x)
        else:
          if not isnan(dcv2[c][b][dkey]):
            v.append(dcv2[c][b][dkey])
    ax = subplot(3,2,bdx+1)
    hist(v,density=True,bins=bins,color=clr,histtype=histtype,linewidth=lw)
    s = ar + ' ' + b + '\nmedian:' + str(round(median(v),2))+ ' mean:' + str(round(mean(v),2))
    title(s)#,fontsize=45)
    if xl is not None: xlim(xl)
    mv = mean(v)
    plot([mv,mv],[0,ax.get_ylim()[1]],'r--')
    md = median(v)
    plot([md,md],[0,ax.get_ylim()[1]],'b--')    
    if b == 'gamma' or b == 'hgamma': xlabel(xlab)#,fontsize=45)
    lval.append(v)
  return lval

#
def plotddcv2bybandchan (ddcv2,ar,dkey,skipbipolar=True,clr='k',bins=30,xlab=r'$CV^2$',xl=(0,3),histtype='bar',lw=4):
  lband = ['delta','theta','alpha','beta','gamma','hgamma']
  for bdx,b in enumerate(lband):
    v = []
    print(ddcv2[ar].keys())
    for chan in ddcv2[ar].keys():
      dcv2 = ddcv2[ar][chan]
      print(b,chan,dkey,dcv2.keys())
      if type(dcv2[b][dkey])==list:
        for x in dcv2[b][dkey]:
          if not isnan(x):
            v.append(x)
      else:
        if not isnan(dcv2[b][dkey]):
          v.append(dcv2[b][dkey])
    subplot(3,2,bdx+1)
    hist(v,normed=True,bins=bins,color=clr,histtype=histtype,linewidth=lw)
    s = ar + ' ' + b + ' median:' + str(round(median(v),2))+ ' mean:' + str(round(mean(v),2))
    title(s)
    xlim(xl)
    if b == 'gamma' or b == 'hgamma': xlabel(xlab)

    
#    
def loaddframebyarband (lcol,skipbipolar=True,skipcsd=False,FoctTH=1.5,ERPscoreTH=0.8,ERPDurTH=[75,300]):
  lar = ['A1', 'Thal']
  based = 'data/spont/oscout/'
  ddf = {'A1':{'s2':{},'g':{},'i1':{}},'Thal':{'Th':{}}}
  for ar,lschan in zip(lar,[['s2','g','i1'],['Th']]):
    for schan in lschan:
      for b in lband:
        ddf[ar][schan][b]={k:[] for k in lcol}
  for ar in lar:      
    for fn in os.listdir(based+ar):
      if getorigsampr('data/spont/'+ar+'/'+fn.split('_')[0]) != 44e3: continue
      if not fn.endswith('dframe.pkl'): continue
      if skipbipolar and fn.count('bipolar_True')>0: continue
      if skipcsd and fn.count('bipolar_False')>0: continue
      df = pickle.load(open(based+ar+'/'+fn,'rb'))
      print(fn)
      lchan = list(set(df['chan']))
      lchan.sort()
      if ar == 'A1':
       s2,g,i1 = lchan
       lschan = ['s2','g','i1']
      else:
        th = lchan[0]
        lschan = ['Th']
      for band in lband:
        for chan,schan in zip(lchan,lschan):
          dfs = df[(df.band==band) & (df.Foct<FoctTH) & (df.chan==chan) & ((df.ERPscore<ERPscoreTH)|(df.dur<ERPDurTH[0])|(df.dur>ERPDurTH[1]))]
          for k in lcol:
            lx = dfs[k]
            for x in lx: ddf[ar][schan][band][k].append(x)
  return ddf

# plot 
def plotdframebyarband (ddf,kcol,lband=['delta','theta','alpha','beta','gamma','hgamma'],\
                        lar=['A1','STG'],llschan=[['s2','g','i1'],['104']],\
                        llclr=[['r','g','b'],['c']],\
                        llab=['A1 supragran','A1 gran','A1 infragran','Human STG'],lcflat=['r','g','b','c'],drawlegend=True,ylab=None,msz=40):
  import matplotlib.patches as mpatches
  dtitle = {b:'' for b in lband}
  dlm = {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # 'A1':{'s2':[],'g':[],'i1':[]},'Thal':{'Th':[]}}
  dls = {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)}
  from nhpdat import dbands
  xfreq = [(dbands[k][1]+dbands[k][0])/2. for k in dbands.keys()]
  for ar,lsch,lclr in zip(lar,llschan,llclr):
    for schan in lsch:
      for bdx,b in enumerate(lband):
        dlm[ar][schan].append(mean(ddf[ar][schan][b][kcol]))
        dls[ar][schan].append(std(ddf[ar][schan][b][kcol])/sqrt(len(ddf[ar][schan][b][kcol])))  
  for ar,lsch,lclr in zip(lar,llschan,llclr):
    for schan,clr in zip(lsch,lclr):
      plot(xfreq,np.array(dlm[ar][schan])-dls[ar][schan],clr+'--')
      plot(xfreq,np.array(dlm[ar][schan])+dls[ar][schan],clr+'--')
      plot(xfreq,dlm[ar][schan],clr)
      plot(xfreq,dlm[ar][schan],clr+'o',markersize=msz)
  xlabel('Frequency (Hz)');
  if ylab is None:
    ylabel(kcol)
  else:
    ylabel(ylab)
  ax=gca()
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(lcflat,llab)]
  if drawlegend: ax.legend(handles=lpatch,handlelength=1)  
  return dlm,dls

# plot 
def plotdframebyarbandhist (ddf,kcol,lband=['delta','theta','alpha','beta','gamma','hgamma'],xl=None,xlab=None,ylab=None,\
                            lar=['A1','Thal'],llschan=[['s2','g','i1'],['Th']],\
                            llclr=[['r','g','b'],['c']],\
                            llab=['A1 supragran','A1 gran','A1 infragran','Thal'],lcflat=['r','g','b','c'],bins=20):
  import matplotlib.patches as mpatches
  dtitle = {b:'' for b in lband}
  dlm = {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # mean
  dls = {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # standard error
  dlmin =  {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # min
  dlmax  =  {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # max
  dlmed =  {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # median
  dlN =  {ar:{ch:[] for ch in lsch} for ar,lsch in zip(lar,llschan)} # median  
  from nhpdat import dbands
  xfreq = [(dbands[k][1]+dbands[k][0])/2. for k in dbands.keys()]
  for ar,lsch,lclr in zip(lar,llschan,llclr):
    for schan,clr in zip(lsch,lclr):
      for bdx,b in enumerate(lband):
        subplot(3,2,bdx+1); title(b)
        hist(ddf[ar][schan][b][kcol],density=True,histtype='step',linewidth=10,color=clr,bins=bins)
        if xl is not None: xlim(xl)
        if xlab is not None: xlabel(xlab)
        if ylab is not None: ylabel(ylab)
        dlm[ar][schan].append(mean(ddf[ar][schan][b][kcol]))
        dls[ar][schan].append(std(ddf[ar][schan][b][kcol])/sqrt(len(ddf[ar][schan][b][kcol])))
        dlmin[ar][schan].append(min(ddf[ar][schan][b][kcol]))
        dlmax[ar][schan].append(max(ddf[ar][schan][b][kcol]))
        dlmed[ar][schan].append(median(ddf[ar][schan][b][kcol]))
        dlN[ar][schan].append(len(ddf[ar][schan][b][kcol]))
        print(ar,schan,clr,b,kcol,dlN[ar][schan][-1],dlmin[ar][schan][-1],dlmax[ar][schan][-1],dlmed[ar][schan][-1],dlm[ar][schan][-1],dls[ar][schan][-1])
  ax=gca()
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(lcflat,llab)]
  ax.legend(handles=lpatch,handlelength=1)  
  return dlm,dls,dlmin,dlmax,dlmed,dlN

    
#
def loaddlcoh (lband = ['delta','theta','alpha','beta','gamma','hgamma'], skipbipolar = True,\
               ar='A1', bdir='data/spont/laggedcoh/A1',origdir='data/spont/A1/',lschan=['s2','g','i1']):
  from nhpdat import getorigsampr
  ddlcoh = {}
  ddlcoh[ar] = {}
  lfn = os.listdir(bdir)
  for fn in lfn:
    if skipbipolar and fn.count('bipolar_True') > 0: continue
    origfn = origdir+fn.split('_')[0]
    if ar == 'A1' and getorigsampr(origfn) < 44e3: continue
    if fn.endswith('.pkl'): ddlcoh[ar][fn] = pickle.load(open(bdir+'/'+fn,'rb'))
  dlcoh = {ar:{schan:{} for schan in lschan}}
  for c in lschan:
    for b in lband:
      dlcoh[ar][c][b]=[]
  for k in ddlcoh[ar].keys():
    for chan,schan in zip(ddlcoh[ar][k].keys(),lschan):
      for b in lband:
        for x in ddlcoh[ar][k][chan][b]: dlcoh[ar][schan][b].append(x)        
  return ddlcoh,dlcoh

# plot lagged coherence output as line plot
def plotdlcoh (dlcoh,lband=['delta','theta','alpha','beta','gamma','hgamma'],\
               ar='A1',lschan=['s2','g','i1'],lclr=['r','g','b'],dolegend=True):
  import matplotlib.patches as mpatches
  dlm = {ar:{schan:[] for schan in lschan}}
  dls = {ar:{schan:[] for schan in lschan}}
  from nhpdat import dbands
  xfreq = [(dbands[k][1]+dbands[k][0])/2. for k in dbands.keys()]
  for ar,lsch,lclr in zip([ar],[lschan],[lclr]):
    for schan in lsch:
      for bdx,b in enumerate(lband):
        dlm[ar][schan].append(mean(dlcoh[ar][schan][b]))
        dls[ar][schan].append(std(dlcoh[ar][schan][b])/sqrt(len(dlcoh[ar][schan][b])))      
  for ar,lsch,lclr in zip([ar],[lschan],[lclr]):
    for schan,clr in zip(lsch,lclr):
      plot(xfreq,np.array(dlm[ar][schan])-dls[ar][schan],clr+'--')
      plot(xfreq,np.array(dlm[ar][schan])+dls[ar][schan],clr+'--')
      plot(xfreq,dlm[ar][schan],clr)
      plot(xfreq,dlm[ar][schan],clr+'o',markersize=40)
  xlabel('Frequency (Hz)',fontsize=45); ylabel('Lagged Coherence',fontsize=45)
  if dolegend:
    ax=gca()
    lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(lclr,['NHP A1 supragranular','NHP A1 granular','NHP A1 infragranular'])]
    ax.legend(handles=lpatch,handlelength=1)  
  return dlm,dls

if __name__ == "__main__":
  batchty = 0
  if len(sys.argv) > 1: batchty = int(sys.argv[1])
  if batchty == 0:
    print('eventbatch')
    eventbatch()
  elif batchty == 1:
    print('laggedcohbatch')
    laggedcohbatch()
  elif batchty == 2:
    print('laggedcohnobandbatch')
    laggedcohnobandbatch()
  elif batchty == 3:
    print('simbatch')
    simbatch()
  
