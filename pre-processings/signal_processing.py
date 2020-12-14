## Debug
import  sys, os, glob
sys.path.append (os.path.join (os.path.dirname (__file__), '..', 'oscilloscopes', ))

import numpy          as np
import scipy.signal
import ctypes

# import matplotlib as plt
## to avoid bug when it is run without graphic interfaces
import matplotlib, sys
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

################################################################################
def unpackData (dataFile, device):
################################################################################
# unpackData
# read data from oscilloscope files
#
# Args:
#  + dataFile: path to the data to read
#  + device: type of files ('i', 'hackrf')
#
# Returns:
#  + x: the traces stored in the given file
################################################################################

    if device == 'i':
        dataFileHandler = open (dataFile, mode = 'br')
        x = np.array ([ctypes.c_int8(i).value for i in bytearray (dataFileHandler.read ())])
        dataFileHandler.close ()
        return x
    elif device == 'hackrf':
        return np.fromfile (dataFile, dtype=np.complex64)
    else: # pico
        MAX_VALUE = 32512.
        return np.fromfile (dataFile, np.dtype ('int16'))/MAX_VALUE


################################################################################
def stft (X, F, window, overlap, verbose = False):
################################################################################
# stft
# compute the stft (short-time fourrier transform) of the given signal X recorded
# at sampling rate F.
#
# Args:
#  + X: trace
#  + F: frequency (Hz)
#  + window: size of the window in number of time samples
#  + overlap: size of the overlap in number of time samples
#  + verbose: to display the spectrogram (default value is false)
#
# Returns:
#  + f: frequency axis
#  + t: time axis
#  + Zxx: absolute value of the spectrogram
################################################################################
    # f, t, Zxx = scipy.signal.stft (X, F, window = 'triang', nperseg =
    #                                window, noverlap = overlap, return_onesided= False)

    if (np.any (np.iscomplex (X))):
        f, t, Zxx = scipy.signal.stft (X, F, window = 'triang', nperseg =
                                       window, noverlap = overlap,
                                       return_onesided = False)
    else:
        f, t, Zxx = scipy.signal.stft (X, F, window = 'triang', nperseg =
                                       window, noverlap = overlap,
                                       return_onesided = True)

    if (verbose):
        fig, axs = plt.subplots (figsize = (16, 9), sharex=True, sharey=True)
        im = axs.imshow (Zxx, cmap = 'Reds', interpolation ='none', aspect='auto',
                         origin ='lower',
                         extent = [t.min (), t.max (), f.min (), f.max ()])

        axs.set_ylabel('Frequency [Hz]')
        axs.set_xlabel('Time [s]')

        plt.show ()

    return t, f, np.abs (Zxx)


################################################################################
def butter_bandpass(lowcut, highcut, fs, order=5):
################################################################################
# butter_bandpass
# generate filter for passband or low/hight-pass
#
# sources:
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
#
# inputs:
#  + lowcut/highcut: threshold in Hz,
#  + fs: frequency in Hz,
#  + order: 5 by default
################################################################################
    nyq = 0.5 * fs
    if (lowcut == 0):
        high = highcut / nyq

        b, a = scipy.signal.butter (order, high, btype='lowpass')
        return b, a

    elif (highcut == 0):
        low = lowcut / nyq

        b, a = scipy.signal.butter (order, low, btype='highpass')
        return b, a

    else:
        low = lowcut / nyq
        high = highcut / nyq

        b, a = scipy.signal.butter (order, [low, high], btype='band')
        return b, a


################################################################################
def butter_bandpass_filter (X, F, lowcut, highcut, order=5, verbose = False):
################################################################################
# butter_bandpass_filter
# apply filter (passband or low/hight-pass) on the given trace X.
#
# sources:
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
#
# inputs:
#  + X: trace to filter
#  + F: sampling rate of the recorded traces (in Hz)
#  + lowcut/highcut: threshold in Hz,
#  + order: 5 by default
#  + verbose: to display the spectrogram (default value is false)
################################################################################
    b, a = butter_bandpass (lowcut, highcut, F, order=order)

    y = scipy.signal.lfilter (b, a, X)

    if (verbose):
        plt.figure ()
        plt.plot (X, label = 'raw data')
        plt.plot (y, label = 'filtered [lc/hc = %s/%s]'%(lowcut, highcut))
        plt.xlabel ('time (sample)')
        plt.legend ()
        plt.show ()

    return y


################################################################################
def multi_butter_bandpass_filter (X, F, lowcuts, highcuts, order=5, verbose = False):
################################################################################
# multi_butter_bandpass_filter
# apply multiple passband
#
# sources:
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
#
# inputs:
#  + X: trace to filter
#  + F: sampling rate of the recorded traces (in Hz)
#  + lowcut/highcut: thresholds in Hz (list of thresholds),
#  + order: 5 by default
#  + verbose: to display the spectrogram (default value is false)
#
# returns:
#  - y: filtered signal
################################################################################

    for i in range (len (lowcuts)):
        b, a = butter_bandpass (lowcuts [i], highcuts [i], F, order=order)

        if (i == 0):
            y = scipy.signal.lfilter (b, a, X)
        else:
            y += scipy.signal.lfilter (b, a, X)

    if (verbose):
        plt.figure ()
        plt.plot (X, label = 'raw data')
        plt.plot (y, label = 'filtered [lc/hc = %s/%s]'%(lowcuts, highcuts))
        plt.xlabel ('time (sample)')
        plt.legend ()
        plt.show ()

    return y
