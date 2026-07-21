#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:57:42 2026

@author: max
"""
import copy
import numpy as np
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim import sim_oscillation

import matplotlib.pyplot as plt

import oevent

#%% Plotting function
def plot_events_ts(df_band, sig, time, fs):
    
    T_min = df_band.minT.values
    T_max = df_band.maxT.values

    # Pick an event
    lfp_bg = copy.copy(sig)

    plt.figure()

    for i in range(len(T_min)):
        
        sidx_ev = round(T_min[i]/1e3 * fs)
        eidx_ev = round(T_max[i]/1e3 * fs)
        
        lfp_bg[0, sidx_ev:eidx_ev] = np.nan
        
        plt.plot(time[sidx_ev:eidx_ev], sig_sim[0, sidx_ev:eidx_ev], 'r', linewidth=1)    

    plt.plot(time, lfp_bg.T, 'k', linewidth=1)
    
    plt.xlim([time[0], time[-1]])
    plt.xlabel('Time [s]')
    
#%% Simulation
n_cyc = 10

sig_cyc = np.empty(n_cyc, dtype=object)

for c in range(n_cyc):
    
    n_sec_osc = 5
    n_sec_ap = 5
    fs = 512
 
    sig_ap = sim_powerlaw(n_sec_ap, fs, exponent=-2.3, f_range=[2, None])

    sig_ap_osc = sim_powerlaw(n_sec_osc, fs, exponent=-2.3, f_range=[2, None])
    sig_osc = sim_oscillation(n_sec_osc, fs, 10)
    
    sig_osc += sig_ap_osc
    
    sig_cyc[c] = np.hstack((sig_ap, sig_osc))
    
sig_sim = np.hstack(sig_cyc)

time = np.arange(len(sig_sim)) / fs

#%% Compute the wavelent spectrogram

winsz = len(sig_sim) / fs # 10 second window size
freqmin = 1. # minimum frequency (Hz)
freqmax = 250.  # maximum frequency (Hz)
freqstep = 2 # frequency step (Hz)
overlapth = 0.1 # overlapping bounding box threshold (threshold for merging event bounding boxes)

# Get spectrum with oevent
lms,lnoise,lsidx,leidx = oevent.getmorletwin(sig_sim, int(winsz*fs), fs, 
                                             freqmin=freqmin, freqmax=freqmax, 
                                             freqstep=freqstep,
                                             noiseamp=200.)

# Plot signal and spectrum
plt.figure()
plt.plot(time, sig_sim, linewidth=1)
plt.xlim([time[0], time[-1]])
plt.xlabel('Time [s]')

plt.figure()
plt.plot(lms[0].f, np.log(np.median(lms[0].TFR, axis=1)), linewidth=1)
plt.xlim([lms[0].f[0], lms[0].f[-1]])
plt.xlabel('Freq [Hz]')
plt.ylabel('Log Power')

plt.figure()
plt.imshow(np.log(lms[0].TFR), 
           extent=[time[0], time[-1], lms[0].f[-1], lms[0].f[0]], 
           aspect='auto')
plt.xlabel('Time [s]')
plt.ylabel('Freq [Hz]')
plt.colorbar(label='Log Power')

#%% OEvent oscillation detection 

# parameters for OEvent
medthresh = 4 # median threshold
lchan = [0]

sig_sim = np.expand_dims(sig_sim, 0)
events_1f = oevent.getIEIstatsbyBand(sig_sim,winsz,fs,freqmin,freqmax,freqstep,
                                     medthresh,lchan,None,overlapth,getphase=True,
                                     savespec=True,normop=oevent.one_over_f_norm,
                                     neighborhood=np.ones((10,1500)))
events_med = oevent.getIEIstatsbyBand(sig_sim,winsz,fs,freqmin,freqmax,freqstep,
                                      medthresh,lchan,None,overlapth,getphase=True,
                                      savespec=True,normop=oevent.mednorm,
                                      neighborhood=np.ones((10,1500)))

df_1f = oevent.GetDFrame(events_1f,fs, sig_sim, None, haveMUA=False) # convert the oscillation event data into a pandas dataframe
df_med = oevent.GetDFrame(events_med,fs, sig_sim, None, haveMUA=False)

df_alpha_1f = df_1f[(df_1f.peakF>5) & (df_1f.peakF<15) & (df_1f.dur>1.5*fs)] 
df_alpha_med = df_med[(df_med.peakF>5) & (df_med.peakF<15) & (df_med.dur>1.5*fs)] 

# Plot normalized spectrograms
plt.figure()
plt.imshow(events_1f[0]['lms'][0].TFR, 
           extent=[time[0], time[-1], events_1f[0]['lms'][0].f[-1], events_1f[0]['lms'][0].f[0]], 
           aspect='auto', vmax=10)
plt.xlabel('Time [s]')
plt.ylabel('Freq [Hz]')
plt.colorbar(label='Power')
plt.title('1/f normalization')

plt.figure()
plt.imshow(events_med[0]['lms'][0].TFR, 
           extent=[time[0], time[-1], events_1f[0]['lms'][0].f[-1], events_med[0]['lms'][0].f[0]], 
           aspect='auto', vmax=10)
plt.xlabel('Time [s]')
plt.ylabel('Freq [Hz]')
plt.colorbar(label='Power')
plt.title('Median normalization')

# Plot histogram in frequency of interest
idx_f = events_med[0]['lms'][0].f == 9 

plt.figure()
plt.subplot(2,2,1)
plt.hist(events_1f[0]['lms'][0].TFR[idx_f,:].T, 50)
ylim = plt.ylim()
plt.plot(medthresh*np.ones((2,1)), ylim, 'r', linewidth=1)
plt.ylim(ylim)
plt.xlim([0, 40])
plt.xlabel('Power [norm.]')
plt.ylabel('#')
plt.title('9Hz - 1/f normalization')

plt.subplot(2,2,3)
plt.hist(events_med[0]['lms'][0].TFR[idx_f,:].T, 50)
ylim = plt.ylim()
plt.plot(medthresh*np.ones((2,1)), ylim, 'r', linewidth=1)
plt.ylim(ylim)
plt.xlim([0, 10])
plt.xlabel('Power [norm.]')
plt.ylabel('#')
plt.title('9Hz - Median normalization')

idx_f = events_med[0]['lms'][0].f == 149

plt.subplot(2,2,2)
plt.hist(events_1f[0]['lms'][0].TFR[idx_f,:].T, 200)
ylim = plt.ylim()
plt.plot(medthresh*np.ones((2,1)), ylim, 'r', linewidth=1)
plt.ylim(ylim)
plt.xlim([0, 20])
plt.xlabel('Power [norm.]')
plt.ylabel('#')
plt.title('149Hz - 1/f normalization')

plt.subplot(2,2,4)
plt.hist(events_med[0]['lms'][0].TFR[idx_f,:].T, 200)
ylim = plt.ylim()
plt.plot(medthresh*np.ones((2,1)), ylim, 'r', linewidth=1)
plt.ylim(ylim)
plt.xlim([0, 20])
plt.xlabel('Power [norm.]')
plt.ylabel('#')
plt.title('149Hz - Median normalization')

plt.tight_layout()

# Plot detected events
plot_events_ts(df_alpha_1f, sig_sim, time, fs)
plt.title('1/f normalization')

plot_events_ts(df_alpha_med, sig_sim, time, fs)
plt.title('Median normalization')