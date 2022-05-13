# example of oevent use - email samuel.neymotin@nki.rfmh.org for questions
from oevent import * # import oevent code

# make sure to download s2samp.npy from https://drive.google.com/file/d/1hhdKog80aEWFfgWEq-mlxTLPRhjBgQtD/view?usp=sharing

try:
  dat = np.load('s2samp.npy') # example 20 second CSD from supragranular sink channel; 11 kHz sampling rate
except:
  print('ERROR: could not load n2samp.npy, make sure to download from https://drive.google.com/file/d/1hhdKog80aEWFfgWEq-mlxTLPRhjBgQtD/view?usp=sharing')
  quit()

# parameters for OEvent
medthresh = 4.0 # median threshold
sampr = 11000 # sampling rate
winsz = 10 # 10 second window size
freqmin = 0.25 # minimum frequency (Hz)
freqmax = 250.0  # maximum frequency (Hz)
freqstep = 0.25 # frequency step (Hz)
overlapth = 0.5 # overlapping bounding box threshold (threshold for merging event bounding boxes)
chan = 0 # which channel to use for event analysis
lchan = [chan] # list of channels to use for event analysis
MUA = None # multiunit activity; not required

print('Calculating wavelets & extracting oscillation events. . .')
#dat = CSD # use current source density signals (CSD); can also use LFP or MUA
dout = getIEIstatsbyBand(dat,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,MUA,overlapth,getphase=True,savespec=True)
df = GetDFrame(dout,sampr, dat, MUA, haveMUA=False) # convert the oscillation event data into a pandas dataframe

print('Normalizing wavelet spectrograms. . .')
for ms in dout[chan]['lms']: ms.TFR = mednorm(ms.TFR) # this is for drawing normalized spectrograms

dlms={chan:dout[chan]['lms'] for chan in lchan}; # dictionary of morlet spectrograms (one per channel)

print('Creating event viewer. . .')
tt = np.linspace(0,len(dat[0])/sampr,len(dat[0])) # time in units of seconds
evv = eventviewer(df,dat,None,tt,sampr,winsz,dlms) # create an event viewer to examine the oscillation events
evv.specrange = (0,10) # spectrogram color range (in median normalized units)

# select and view a beta event
dfs = df[(df.band == 'beta') & (df.filtsigcor>0.5) & (df.Foct<1.5) & (df.ncycle>3)] 
evv.draw(dfs.index[0],clr='red',drawfilt=True,filtclr='b',ylspec=(0.25,45),lwfilt=3,lwbox=3,verbose=False)
evv.cbaxes = evv.fig.add_axes([0.925, 0.5+.135, 0.0125, 0.15]); colorbar(evv.specimg, cax=evv.cbaxes)

