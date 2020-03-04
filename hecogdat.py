"""
Created on Fri Apr 13 10:55:13 2018

@author: sbickel, samn
"""
import hdf5storage # easier to use for Matlab 7.3+ files than h5py

# This imports a matlab mat file, which includes a variable called ecog with the data.
# 
# Ecog has several fields specifying seizure onset channels etc.
# The main field is called ecog.ftrip which contains:
# For time domain data:
# ecog.ftrip.fsample
# ecog.ftrip.nChans
# ecog.ftrip.label (channel labels)
# ecog.ftrip.trial (contains a matrix nchan x time)
# ecog.ftrip.time  (time in seconds)
#
# Below is importing the spectral data, but it should work for time domain data too.
# ecog.psd.freqs (frequencies)
# ecog.psd.pwr   (matrix with nchan x freq)

# read the ecog data from the matlab .mat file and return the sampling rate and electrophys data
def rdecog (fn):
  d = {}
  d['dat'] = hdf5storage.read(path='ecog/ftrip/trial',filename=fn)[0][0]
  d['nchan'] = d['dat'].shape[0]
  label = hdf5storage.read(path='ecog/ftrip/label',filename=fn)
  d['label'] = [l[0][0][0] for l in label]
  d['sampr'] = hdf5storage.read(path='ecog/ftrip/fsample',filename=fn)[0][0]
  d['tt'] = hdf5storage.read(path='ecog/ftrip/time',filename=fn)[0][0][0,:]
  return d
