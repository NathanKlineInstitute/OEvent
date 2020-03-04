import numpy
from scipy.stats.stats import pearsonr

# gets population correlation vectors. lfps is column vectors. samples are rows.
def getpcorr (lfps, winsz):
  idx,jdx,n,sz = 0,0,len(lfps[0]),len(lfps)
  pcorr = numpy.zeros((int(math.ceil(sz/winsz)+1),n*(n-1)/2))
  for sidx in range(0,sz,winsz):
    if idx % 10 == 0: print(idx)
    eidx = sidx + winsz
    if eidx >= sz: eidx = sz - 1
    jdx = 0
    for i in range(len(lfps[0])):
      v1 = lfps[sidx:eidx,i]
      for j in range(i+1,len(lfps[0]),1):
        v2 = lfps[sidx:eidx,j]
        pcorr[idx][jdx] = pearsonr(v1,v2)[0]
        jdx += 1
    idx += 1
  return pcorr

#
def getpco (pcorr):
  sz = len(pcorr)
  pco = numpy.zeros((sz,sz))
  for i in range(sz):
    pco[i][i]=1.0
    for j in range(sz):
      pco[i][j] = pco[j][i] = pearsonr(pcorr[i,:],pcorr[j,:])[0]
  return pco
