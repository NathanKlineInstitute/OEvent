from neuron import h

#
def getnqgc (dalphabin,S1toG,GtoS1,S1toI1,I1toS1,GtoI1,I1toG,offset=-1):
  nqgc = h.NQS('gdx','s1on','gon','i1on','S1toG','GtoS1','S1GDir','S1toI1','I1toS1','S1I1Dir','GtoI1','I1toG','GI1Dir')
  for tdx in range(1,len(dalphabin[s1]),1):
    s1on = dalphabin[s1][tdx]
    gon = dalphabin[g][tdx]
    i1on = dalphabin[i1][tdx]
    gdx = tdx + offset
    s1tog = sum(S1toG[gdx][1:15]) # [1:50])+sum(S1toG[gdx][70:115])
    gtos1 = sum(GtoS1[gdx][1:15]) # [1:50])+sum(GtoS1[gdx][70:115])
    s1gdir = s1tog - gtos1
    s1toi1 = sum(S1toI1[gdx][1:15]) # [1:50])+sum(S1toI1[gdx][70:115])
    i1tos1 = sum(I1toS1[gdx][1:15]) # [1:50])+sum(I1toS1[gdx][70:115])
    s1i1dir = s1toi1 - i1tos1
    gtoi1 = sum(GtoI1[gdx][1:15]) # [1:50])+sum(GtoI1[gdx][70:115])
    i1tog = sum(I1toG[gdx][1:15]) # [1:50])+sum(I1toG[gdx][70:115])
    gi1dir = gtoi1 - i1tog
    nqgc.append(gdx,s1on,gon,i1on,s1tog,gtos1,s1gdir,s1toi1,i1tos1,s1i1dir,gtoi1,i1tog,gi1dir)
  return nqgc

#
def getnqTE (dbandbin,CSD,tdur,sampr,lch,dlpr,offset=-1):
  teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
  teCalc = teCalcClass()
  teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
  teCalc.initialise(1) # Use history length 1 (Schreiber k=1)
  teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
  nqte = h.NQS('gdx','s2on','gon','i2on','S2toG','GtoS2','S2GDir','S2toI2','I2toS2','S2I2Dir','GtoI2','I2toG','GI2Dir','s2alpha','galpha','i2alpha')
  s2,g,i2 = lch
  sz = len(dbandbin[s2])
  for tdx in range(1,sz,1):
    if tdx % 10 == 0: print(tdx, ' of ' , sz)
    s2alpha, galpha, i2alpha = dlpr[(s2,'alpha')][tdx],dlpr[(g,'alpha')][tdx],dlpr[(i2,'alpha')][tdx]
    s2on = dbandbin[s2][tdx]
    gon = dbandbin[g][tdx]
    i2on = dbandbin[i2][tdx]
    gdx = tdx + offset
    # short duration from CSD for the calculation
    csds2 = CSD[s2][int(tdx*tdur*sampr):int(tdx*(tdur+1)*sampr)]
    csdg = CSD[g][int(tdx*tdur*sampr):int(tdx*(tdur+1)*sampr)]
    csdi2 = CSD[i2][int(tdx*tdur*sampr):int(tdx*(tdur+1)*sampr)]
    # calculate TE from S2 -> G
    teCalc.initialise() 
    teCalc.setObservations(csds2, csdg)
    s2tog = max(0.0,teCalc.computeAverageLocalOfObservations())
    # calculate TE from G -> S2
    teCalc.initialise() 
    teCalc.setObservations(csdg, csds2)
    gtos2 = max(0.0,teCalc.computeAverageLocalOfObservations())
    # overall direction
    s2gdir = s2tog - gtos2
    # calculate TE from S2 -> I2
    teCalc.initialise() 
    teCalc.setObservations(csds2, csdi2)
    s2toi2 = max(0.0,teCalc.computeAverageLocalOfObservations())
    # calculate TE from I2 -> S2
    teCalc.initialise() 
    teCalc.setObservations(csdi2, csds2)
    i2tos2 = max(0.0,teCalc.computeAverageLocalOfObservations())
    # overall direction
    s2i2dir = s2toi2 - i2tos2
    # calculate TE from G -> I2
    teCalc.initialise() 
    teCalc.setObservations(csdg, csdi2)
    gtoi2 = max(0.0,teCalc.computeAverageLocalOfObservations())
    # calculate TE from I2 -> G
    teCalc.initialise() 
    teCalc.setObservations(csdi2, csdg)
    i2tog = max(0.0,teCalc.computeAverageLocalOfObservations())
    # overall direction
    gi2dir = gtoi2 - i2tog
    nqte.append(gdx,s2on,gon,i2on,s2tog,gtos2,s2gdir,s2toi2,i2tos2,s2i2dir,gtoi2,i2tog,gi2dir,s2alpha,galpha,i2alpha)
  return nqte

#
def getphtrigTE (fn,dsfctr,minf,maxf):
  sampr,dat,dt,tt=rdmat(fn); 
  s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])
  samprds,CSDds,ttds = getCSDds(fn,dsfctr=dsfctr)
  dalpha = {}
  for ch in [s2,g,i2]: dalpha[ch] = bandpass(CSDds[ch,:],minf,maxf,df=samprds,zerophase=True)
  dminmax = {}
  for ch in dalpha.keys(): dminmax[ch] = getpowlocalMinMax(dalpha[ch])
  sz = int(sampr * 0.5) + 1
  szds = int(samprds * 0.5) + 1
  dphmin,dphmax = {},{}
  dphminalph,dphmaxalph = {},{}
  teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
  teCalc = teCalcClass()
  teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
  teCalc.initialise(1) # Use history length 1 (Schreiber k=1)
  teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
  dTEcsdmin,dTEcsdmax={},{}
  #for ch1 in [s2,g,i2]:
  for ch1 in [s2]:
    (lpowMIN,lpowMAX,minx,miny,maxx,maxy) = dminmax[ch1]
    #for ch2 in [s2,g,i2]:
    for ch2 in [g]:
      if ch1 == ch2: continue      
      dTEcsdmin[ch1,ch2] = []; dTEcsdmax[ch1,ch2] = []
      CSD = CSDds[ch2,:]; csdsz = len(CSD)
      print(ch1,ch2,len(minx))
      for x in minx:
        t0 = x*1.0/samprds
        sidx,eidx = int((t0-0.25)*samprds), int((t0+0.25)*samprds + 1)
        if eidx < csdsz and sidx >= 0:
          teCalc.initialise() 
          teCalc.setObservations(CSDds[ch1,sidx:sidx+szds], CSD[sidx:sidx+szds])
          dTEcsdmin[ch1,ch2].append(max(0.0,teCalc.computeAverageLocalOfObservations()))
      for x in maxx:
        t0 = x*1.0/samprds
        sidx,eidx = int((t0-0.25)*samprds), int((t0+0.25)*samprds + 1)
        if eidx < csdsz and sidx >= 0:
          teCalc.initialise() 
          teCalc.setObservations(CSDds[ch1,sidx:sidx+szds], CSD[sidx:sidx+szds])
          dTEcsdmax[ch1,ch2].append(max(0.0,teCalc.computeAverageLocalOfObservations()))
  return dTEcsdmin,dTEcsdmax

# get an nqs with sample entropy entries - lts is a list of time-series
#  sampenM = epoch size, sampenR = error tolerance, sampenN = normalize, sampenSec = seconds to use
#  slideR = whether to use a sliding tolerance
def getnqsampen (lts,sampr,scale=1,sampenM=2,sampenR=0.2,sampenN=0,sampenSec=1,slideR=1):
  if h.INSTALLED_sampen == 0.0: h.install_sampen()
  nq = NQS("t","sampen","chid")
  vec,vs,vt,vch = Vector(), Vector(),Vector(),Vector()
  sampenWinsz = sampenSec * sampr # size in samples
  if sampenWinsz < 100 and sampenSec > 0:
    print("getnqsampen WARNING: sampenWinsz was : ", sampenWinsz, " set to 100.")
    sampenWinsz = 100
    sampenSec = sampenWinsz / sampr # reset sampenSec
    nq.clear( (len(lts[0]) / sampenWinsz + 1) * len(lts) )
  else:
    nq.clear(len(lts))
  chid = 0 # channel ID
  for ts in lts:
    vec.from_python(ts)
    if sampenSec > 0:
      print("chid : " , chid, " of " , len(lts))
      vs.resize( vec.size() / sampenWinsz + 1); vs.fill(0)
      vec.vsampenvst(sampenM,sampenR,sampenN,sampenWinsz,vs,slideR)
      if vt.size() < 1:
        vt.indgen(0,vs.size()-1,1); vt.mul(sampenSec); vt.add(sampenSec / 2.0)
        vch.resize(vt.size());
      vch.fill(chid)
      nq.v[0].append(vt); nq.v[1].append(vs); nq.v[2].append(vch)
    else: # single value for the time-series on the channel
      nq.append(0,vec.vsampen(sampenM,sampenR,sampenN),chid)
    chid += 1
  return nq

# calculates/saves sampen from the mat file (fname)
def savenqsampen (fname,ldsz=[200],csd=False,scale=1,sampenM=2,sampenR=0.2,sampenN=0,sampenSec=1,slideR=1):
  print(' ... ' + fname + ' ... ')
  sampr,dat,dt,tt=None,None,None,None
  try:
    sampr,dat,dt,tt = rdmat(fname)
  except:
    print('could not open ' , fname)
    return False
  print(dat.shape)
  maxf=300; datlow=getlowpass(dat,sampr,maxf); # lowpass filter the data
  del dat
  dat = datlow # reassign dat to lowpass filtered data
  if csd:
    CSD,F,T,lsp=getCSDspec(dat,sampr,window=1,maxfreq=maxf,logit=True)
    del dat
    if len(ldsz) > 0:
      for dsz in ldsz:
        V = downsamplpy(CSD,dsz)
        nq = getnqsampen(V,sampr/dsz,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
        nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_CSD_dsz_"+str(dsz)+"_sampen.nqs")
        del V
        nqsdel(nq)
    else:
      nq = getnqsampen(CSD,sampr/dsz,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
      nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_CSD_sampen.nqs")      
    del CSD,F,T,lsp
  else:
    lts = dat # transpose
    print('lts.shape = ', lts.shape)
    if len(ldsz) > 0:
      for dsz in ldsz:
        print('dsz : ', dsz)
        V = downsamplpy(lts,dsz)
        nq = getnqsampen(V,sampr/dsz,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
        nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_dsz_"+str(dsz)+"_sampen.nqs")
        nqsdel(nq)
        del V
    else:
      nq = getnqsampen(lts,sampr,scale,sampenM,sampenR,sampenN,sampenSec,slideR)
      nq.sv("/u/samn/plspont/data/sampen/"+fname.split("/")[1]+"_sampen.nqs")
      print('nq.gethdrs():')
      nq.gethdrs()
  del tt
  return True

# run sampen on files in lf (list of file paths)
def sampenbatch (lf,nproc=10,ldsz=[200],csd=False,exbbn=True,\
                 scale=1,sampenM=2,sampenR=0.2,sampenN=0,sampenSec=1,slideR=1):
  #pool = Pool(processes=nproc)
  #args = ((fn,dsz,csd,scale,sampenM,sampenR,sampenN,sampenSec,slideR) for fn in lf)
  #print 'args : ' , args
  #pool.map_async(savenqsampen,args)
  #pool.close(); pool.join()  
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    savenqsampen(fn,ldsz,csd,scale,sampenM,sampenR,sampenN,sampenSec,slideR)

# lf is list of files, exbbn == exclude broadband noise files
def rdsampenbatch (lf,dsz=200,csd=False,exbbn=True):
  dnq = {}
  for fn in lf:
    if exbbn and fn.count("spont") < 1: continue
    fnq = '/u/samn/plspont/data/sampen/'+fn.split('/')[1]
    if csd: fnq += '_CSD'
    if dsz > 0: fnq += '_dsz_'+str(dsz)
    fnq += '_sampen.nqs'
    try:
      dnq[fn]=NQS(fnq)
      if dnq[fn].m[0] < 2:
        dnq.pop(fn,None) # get rid of it
    except:
      print('could not open ' , fnq)
  return dnq

#
def grangerVST (ts1,ts2,sampr,WINS,INCS,order=30,maxfreq=125):
  winsz = int(WINS*sampr); incsz = int(INCS*sampr)
  sidx,eidx = 0,winsz;
  Gx2y,Gy2x = [],[];
  maxsz = len(ts1)
  idx=0
  while sidx < maxsz and eidx < maxsz:
    if idx % 10 == 0: print(idx)
    F,pp,cohe,Fx2y,Fy2x,Fxy=granger(ts1[sidx:eidx]-mean(ts1[sidx:eidx]),ts2[sidx:eidx]-mean(ts2[sidx:eidx]),order=order,rate=sampr,maxfreq=maxfreq)
    Gx2y.append(Fx2y); Gy2x.append(Fy2x);
    sidx += incsz; eidx += incsz;
    idx += 1
  return F,numpy.array(Gx2y),numpy.array(Gy2x)

#
def getGrangerVSTPairs (CSD,sampr,lch,lt,WINS=1,INCS=1,order=30):
  dout={}
  for i in range(len(lch)):
    for j in range(i+1,len(lch),1):
      F,Gx2y,Gy2x=grangerVST(CSD[lch[i]],CSD[lch[j]],sampr,WINS,INCS,order)
      dout[lt[i]+'_'+lt[j]]=Gx2y
      dout[lt[j]+'_'+lt[i]]=Gy2x
      dout['F']=F
  return dout

#
def grangerVSTBatch (lf,WINS=1,INCS=1,order=30):
  for fn in lf:
    print('running granger for ' , fn)
    try:
      samprds,CSDds,ttds = getCSDds(fn)
    except:
      print('could not load:' , fn)
      continue
    s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])
    lch=[s2,g,i2]; lt=['s2','g','i2']; 
    foutbase = '/u/samn/plspont/data/granger/13oct4_A_' + fn.split('data/')[1] 
    d = getGrangerVSTPairs(CSDds,samprds,lch,lt,WINS,INCS,order)
    fout = foutbase + '_granger.npz';
    numpy.savez(fout,F=d['F'],s2_g=d['s2_g'],g_s2=d['g_s2'],\
                  s2_i2=d['s2_i2'],i2_s2=d['i2_s2'],\
                  g_i2=d['g_i2'],i2_g=d['i2_g'])
    del d,CSDds,ttds

#
def TEBatch (lf,tdur,dsfctr,stdth):
  for fn in lf:
    print('running TE for ' , fn)
    try:
      samprds,CSDds,ttds = getCSDds(fn,dsfctr=dsfctr)
    except:
      print('could not load:' , fn)
      continue
    s1,s2,g,i1,i2=getflayers(fn.split('data/')[1])
    lfreq,lfwidth,lbwcfc = getlfreqwidths(step=0.5)
    dspec = getdspec(CSDds,samprds,[s1,g,i1])
    dlpr = getchanrelpowinrange(dspec,lfreq,[s1,g,i1],tdur,samprds)
    dspk = getdspk(dlpr,[s1,g,i1],'alpha',stdth)
    dbandbin = getbandbin([s1,g,i1],dspk,dlpr,'alpha')
    nqte = getnqTE(dbandbin,CSDds[:,int(samprds*3):-int(samprds*3)],tdur,samprds,[s1,g,i1],0)
    foutdir = '/u/samn/plspont/data/TE/15oct26/'
    foutbase = foutdir + fn.split('data/')[1] + '_TE_dsf_' + str(dsfctr) + '_tdur_' + str(tdur) 
    print(foutdir, foutbase)
    nqte.sv(foutbase + '_nqte_.nqs') 
    #pickle.dump(dspec,open(foutbase+'_dspec.pkl','w'))
    pickle.dump(dlpr,open(foutbase+'_dlpr.pkl','w'))
    pickle.dump(dspk,open(foutbase+'_dspk.pkl','w'))
    pickle.dump(dbandbin,open(foutbase+'_dbandbin.pkl','w'))
    del dspec,dlpr,dspk,dbandbin,CSDds,ttds

# lf is list of files, exbbn == exclude broadband noise files
def rdTEBatch (lf,tdur,dsfctr):
  nqo = None; fidx = 0
  for fn in lf:
    findir = '/u/samn/plspont/data/TE/15oct26/'
    finbase = findir + fn.split('data/')[1] + '_TE_dsf_' + str(dsfctr) + '_tdur_' + str(tdur) 
    fnq = finbase + '_nqte_.nqs'
    print(fidx,fnq)
    if not os.path.exists(fnq):
      print(fnq , ' does not exist')
      continue
    try:
      nq=NQS(fnq); nq.resize('fidx'); nq.pad(); nq.getcol('fidx').fill(fidx); fidx+=1
      if nqo is None:
        nqo = nq
      else:
        nqo.append(nq)
        h.nqsdel(nq)
    except: 
      print('could not open ' , fnq)
  return nqo

#
def gethist (lT, maxt, binsz, nlevel=0):
  vec = h.Vector()
  vT = h.Vector()
  vT.from_python(lT)
  vec.hist(vT,0,(maxt+binsz-1)/binsz,binsz)
  if nlevel > 0:
    vd=h.Vector(vec.size())
    vec.getdisc(vd,nlevel)
    return vd.to_python()
  return vec.to_python()

#
def normte (a1, a2, nshuf=30):
  h1,h2,ho=h.Vector(),h.Vector(),h.Vector(len(a1)+nshuf+1)
  h1.from_python(a1)
  h2.from_python(a2)
  return max(h1.tentropspks(h2,ho,nshuf),0.0)

