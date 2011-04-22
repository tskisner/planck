from __future__ import division
import exceptions
import numpy as np
from itertools import *
import ephem
import datetime
import private
import os
try:
    from bitstring import ConstBitArray
except exceptions.ImportError:
    print('pix2od requires bitstring: easy_install bitstring')
    sys.exit(1)

OBTSTARTDATE = datetime.datetime(1958,1,1,0,0,0)
LAUNCH = datetime.datetime(2009, 5, 13, 13, 11, 57, 565826)
SECONDSPERDAY = 3600 * 24

def pix2od(ch, pixels):
    """ nside 512 NEST pixel number to OD [91-563 excluding 454-455] for Planck channels
    it returns a list of sets.
    Each set contains the ODs hit by each pixel
    """
    filename = os.path.join(private.PIX2ODPATH, 'od_by_pixel_%s.bin' % ch.replace('M','S'))
    pixels_by_od = ConstBitArray(filename = filename)
    odrange = [91, 563+1]
    num_ods = odrange[1] - odrange[0]
    NSIDE = 512
    NPIX = 12 * NSIDE**2
    tot_ods = list()
    if np.any(np.array(pixels) >= NPIX):
        raise exceptions.ValueError('ERROR: input pixels must be NSIDE 512, RTFM!')
    for pix in pixels:
        ods = set(np.array(list(pixels_by_od[num_ods*pix:num_ods*pix+num_ods].findall([True]))) + odrange[0])
        tot_ods.append(ods)
    return tot_ods

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return izip(*[chain(iterable, repeat(padvalue, n-1))]*n)

def ahfdate2obt(ahfdate):
     jd = ephem.Date(ahfdate.replace('T',' ').replace('-','/')) - ephem.Date('-4713/1/1 12:00:0')
     return jd2obt(jd)

def time2sample(freq, time):
    if freq == 30:
        return int((time - 1621174818.021514892578125) * 32.5079365079365)
    elif freq == 44:
        return int((time - 1621174818.012481689453125) * 46.5454545454545)
    elif freq == 70:
        return int((time - 1621174818.008087158203125) * 78.7692307692308)
    else:
        return None

def ndsample2time(freq, sample):
    if freq == 30:
        time_flat = 1621174818.021514892578125 + (sample / 32.5079365079365)
        BREAK98 = 1629377124.0625305
        BREAK98LEN = 3.573974609375
        afterbreak, = np.where(time_flat >= BREAK98)
        correction = np.zeros_like(time_flat)
        correction[afterbreak] = BREAK98LEN
        time_corrected = time_flat + correction
        return time_corrected

def sample2time(freq, sample):
    if freq == 30:
        return 1621174818.021514892578125 + (sample / 32.5079365079365)
    elif freq == 44:
        return 1621174818.012481689453125 + (sample / 46.5454545454545)
    elif freq == 70:
        return 1621174818.008087158203125 + (sample / 78.7692307692308)
    else:
        return None

def timedelta2seconds(diff):
    return diff.days * 24 * 3600 + diff.seconds + diff.microseconds * 1e-3

def obt2utc(obt):
    '''Convert OBT (s) to UTC'''
    return OBTSTARTDATE + datetime.timedelta(0,obt)

def utc2obt(utc):
    '''Convert UTC (datetime object) to OBT (s)'''
    return timedelta2seconds(utc - OBTSTARTDATE)

def approxod2utc(od):
    return LAUNCH + datetime.timedelta(od)

def utc2approxod(utc):
    return timedelta2seconds(utc - LAUNCH) / SECONDSPERDAY

def approxod2obt(od):
    return utc2obt(approxod2utc(od))

def obt2approxod(obt):
    return utc2approxod(obt2utc(obt))

def powerspectrum(s, Fs):
    import matplotlib.pyplot as plt
    Pxx, freqs = plt.mlab.psd(s, NFFT=len(s), Fs = Fs)
    return freqs, Pxx

def whitenoise(lenght):
    return np.random.standard_normal(size=lenght)
