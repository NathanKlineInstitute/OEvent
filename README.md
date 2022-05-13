# OEvent:
Neural oscillation event detection and feature analysis.

## Requirements

Python 3 with scipy/numpy/pandas

## Nonhuman primate data

We have provided a recording (s2samp.npy) from the Lakatos lab at Nathan Kline Institute here:
 https://drive.google.com/file/d/1hhdKog80aEWFfgWEq-mlxTLPRhjBgQtD/edit
The data consists of 20 seconds of current-source density (CSD) signal, from a supragranular sink channel.
The sampling rate is 11 kHz. The data is stored in the numpy format.

## Example Use in examp.py

make sure you've downloaded the data file to the same directory as examp.py

Then, run the example with:
python -i examp.py

This will load oevent and use it to calculate wavelet spectrograms, then extract the oscillation events.
Finally, an "event viewer" will be created which will display a beta oscillation event (wavelet spectrogram
in top panel and CSD signal in the bottom panel (red is the raw signal during event, and blue is the filtered
signal).

### Contact:
For questions/comments email samuel.neymotin@nki.rfmh.org

### References:
Taxonomy of neural oscillation events in primate auditory cortex
SA Neymotin, I Tal, A Barczak, MN O'Connell, T McGinnis, N Markowitz, E Espinal, E Griffith, H Anwar, S Dura-Bernal, CE Schroeder, WW Lytton, SR Jones, S Bickel, P Lakatos
https://doi.org/10.1101/2020.04.16.045021
Under review at eNeuro

