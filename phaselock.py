import numpy as np
import scipy.signal as sps
import ctypes
from ctypes import cdll, c_double, c_void_p
import os
import numpy.ctypeslib as npct

# calculates a phase locking value between 2 time series via morlet wavelets
class PhaseLock():
    """ Based on 4Dtools (deprecated) MATLAB code
        Might be a newer version in fieldtrip
    """
    def __init__ (self, tsarray1, tsarray2, sampr, f_max=60.):
        # Save time-series arrays as self variables
        # ohhhh. Do not use 1-indexed keys of a dict!
        self.ts = {
            1: tsarray1,
            2: tsarray2,
        }
        # Set frequecies over which to sort
        self.f = 1. + np.arange(0., f_max, 1.)
        # Set width of Morlet wavelet (>= 5 suggested)
        self.width = 7.
        # sampling frequency
        self.fs = sampr 
        self.data = self.traces2PLS()
    def traces2PLS (self):
        # Not sure what's going on here...
        # nshuffle = 200;
        nshuffle = 1;
        # Construct timevec
        tvec = np.arange(1, self.ts[1].shape[1]) / self.fs
        # Prellocated arrays
        # Check sizes
        B = np.zeros((self.f.size, len(self.ts[1])))
        Bstat = np.zeros((self.f.size, len(self.ts[1])))
        Bplf = np.zeros((self.f.size, len(self.ts[1])))
        # Do the analysis
        for i, freq in enumerate(self.f):
            print('%i Hz' % freq)
            # Get phase of signals for given freq
            # Check sizes
            B1 = self.phasevec(freq, num_ts=1)
            B2 = self.phasevec(freq, num_ts=2)
            # Potential conflict here
            # Check size
            B[i, :] = np.mean(B1 / B2, axis=0)
            B[i, :] = abs(B[i, :])
            # Randomly shuffle B2
            for j in range(0, nshuffle):
                # Check size
                idxShuffle = np.random.permutation(B2.shape[0])
                B2shuffle = B2[idxShuffle, :]
                Bshuffle = np.mean(B1 / B2shuffle, axis=0)
                Bplf[i, :] += Bshuffle
                idxSign = (abs(B[i, :]) > abs(Bshuffle))
                Bstat[i, idxSign] += 1
        # Final calculation of Bstat, Bplf
        Bstat = 1. - Bstat / nshuffle
        Bplf /= nshuffle
        # Store data
        return {
            't': tvec,
            'f': self.f,
            'B': B,
            'Bstat': Bstat,
            'Bplf': Bplf,
        }
    def phasevec (self, f, num_ts=1):
        """ should num_ts here be 0, as an index?
        """
        dt = 1. / self.fs
        sf = f / self.width
        st = 1. / (2. * np.pi * sf)
        # create a time vector for the morlet wavelet
        t = np.arange(-3.5*st, 3.5*st+dt, dt)
        m = self.morlet(f, t)
        y = np.array([])
        for k in range(0, self.ts[num_ts].shape[0]):
            if k == 0:
                s = sps.detrend(self.ts[num_ts][k, :])
                y = np.array([sps.fftconvolve(s, m)])
            else:
                # convolve kth time series with morlet wavelet
                # might as well let return valid length (not implemented)
                y_tmp = sps.fftconvolve(self.ts[num_ts][k, :], m)
                y = np.vstack((y, y_tmp))
        # Change 0s to 1s to avoid division by 0
        # l is an index
        # y is now complex, so abs(y) is the complex absolute value
        l = (abs(y) == 0)
        y[l] = 1.
        # normalize phase values and return 1s to zeros
        y = y / abs(y)
        y[l] = 0
        y = y[:, np.ceil(len(m)/2.)-1:y.shape[1]-np.floor(len(m)/2.)]
        return y
    def morlet (self, f, t):
        """ Calculate the morlet wavelet
        """
        sf = f / self.width
        st = 1. / (2. * np.pi * sf)
        A = 1. / np.sqrt(st*np.sqrt(np.pi))
        y = A * np.exp(-t**2./(2.*st**2.)) * np.exp(1.j*2.*np.pi*f*t)
        return y

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

def loadfunc ():
  from os import system
  try:
    phslock = npct.load_library("libphslock", ".")
  except:
    print('compiling phslock.c')
    os.system('gcc -Wall -fPIC -c phslock.c')
    #os.system('gcc -shared -Wl, -soname,libphslock.so -o libphslock.so phslock.o')
    os.system('gcc -shared -o libphslock.so phslock.o')  
    phslock = npct.load_library("libphslock", ".")
  # setup the return types and argument types
  phslock.phslockv.restype = c_double
  phslock.phslockv.argtypes = [array_1d_double, array_1d_double, ctypes.c_int]
  phslock.phslockvshuf.restype = c_double
  phslock.phslockvshuf.argtypes = [array_1d_double, array_1d_double, ctypes.c_int, ctypes.c_int]
  return phslock

phslock = loadfunc()

#
def getphaselockv(a, b, nshuf=0):
  if not type(a) == np.ndarray: a = np.array(a)
  if not type(b) == np.ndarray: b = np.array(b)
  if not a.flags.c_contiguous: a = np.ascontiguousarray(a)
  if not b.flags.c_contiguous: b = np.ascontiguousarray(b)
  if nshuf > 0:
    return phslock.phslockvshuf(a, b, len(a), nshuf)
  else:
    return phslock.phslockv(a, b, len(a))
  
