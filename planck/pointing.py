from __future__ import division

import math
import operator
import pyfits
import logging as l 
import csv
import glob
from itertools import *
from Quaternion import Quat as quat
import numpy as np
from LFI import LFI
#from IPython.Debugger import Tracer; debug_here = Tracer()
import re
import cmb.quaternionarray as qarray
from utils import grouper
import planck
import private
from cgkit.cgtypes import *
    
QECL2GAL = np.array((-0.37382079227204573, 0.33419217216073838, 0.64478939348298625, 0.57690575088960561))

def ahf_limits(odrange = range(90,481 +1), folder = private.AHF_limits):
    from IPython.kernel import client
    tc = client.TaskClient()
    files = [glob.glob(folder + '/%04d/att_hist_high*' % od)[0] for od in odrange]
    lims = tc.map(get_ahf_lim, files)
    outfile = open('/u/zonca/p/remix/AHF-limits.txt','w')
    out = csv.writer(outfile)
    for f,l in zip(files,lims):
        out.writerow([f,'%f' % l[0],'%f' % l[1]])
    outfile.close()

def get_ahf_lim(AHF):
    import pyfits
    f = pyfits.open(AHF)
    return f[1].data.field('OBT_SPL')[[0,-1]]/2.**16

class Siam(object):

    def __init__(self):
        siamfile = private.siam
        l.debug('using SIAM %s' % siamfile)
        f = open(siamfile)
        lines = f.readlines()
        self.siam = {}
        for line in grouper(4,lines[1:]):
            chtag = line[0].split()[0]
            m = np.matrix(';'.join(line[1:])).T
            self.siam[chtag] = m
    def get(self, ch):
        if ch.inst.name == 'HFI':
            return self.siam[ch.tag]
        else:
            l.warning('For LFI using instrument DB angles')
            return SiamAngles().get(ch)

class SiamAngles(object):

    SPIN2BORESIGHT = 85.0
    
    def __init__(self):
        pass

    def get(self, ch):
        mat_spin2boresight=mat3.rotation(np.pi/2-self.SPIN2BORESIGHT/180.*np.pi,vec3(0,1,0))
        theta = ch.rimo['THETA_UV']/180.*np.pi
        phi = ch.rimo['PHI_UV']/180.*np.pi
        psi = ch.rimo['PSI_UV']/180.*np.pi
        mat_theta_phi = mat3.rotation(theta,vec3(-math.sin(phi),math.cos(phi),0))
        mat_psi = mat3.rotation(psi,vec3(0,0,1))
        # detector points to X axis
        total = mat_spin2boresight * (mat_theta_phi * mat_psi)
        total_mat = np.matrix(np.split(np.array(total.toList(rowmajor=True)),3))
        # siam is defined as pointing to Z axis
        return total_mat * np.matrix([[0,0,1],[0,1,0],[1,0,0]])

def AHF_filename2od(filename):
        return int(re.findall('/(\d{4})/',filename)[0])

def get_AHF_limits(filenames = True):
        limitsfile = private.limitsfile
        if filenames:
            dt=np.dtype({'names':['filename','start','end'],'formats':['S100',np.float,np.float]})
            return np.loadtxt(open(limitsfile),delimiter=',',dtype=dt)
        else:
            dt=np.dtype({'names':['od','start','end'],'formats':[np.int,np.float,np.float]})
            return np.loadtxt(open(limitsfile),delimiter=',',dtype=dt, converters={0: AHF_filename2od})

def AHF_btw_OBT(obt):

        limits = get_AHF_limits()

        first_file_index = limits['start'].searchsorted(obt[0]) - 1
        last_file_index = limits['end'].searchsorted(obt[-1])
        if first_file_index == -1:
            first_file_index = 0
        if last_file_index > len(limits) -1:
            last_file_index -= 1

        files = limits['filename'][first_file_index:last_file_index + 1]
        l.debug('Opening %s' % files)
        return files

def generate_repointing_flag(obt):
    flag = np.zeros_like(obt)
    files = [pyfits.open(file)[1].data for file in AHF_btw_OBT(obt)]

    files[-1] = files[-1][:(files[-1].field('OBT_SPL')/2.**16).searchsorted(obt[-1])+1]
    files[0] = files[0][(files[0].field('OBT_SPL')/2.**16).searchsorted(obt[0])-1:]
    AHF = np.concatenate(files)

    i_start_repointing, = np.nonzero(np.diff(AHF['OBT_BEG']))
    start_repointing = AHF['OBT_SPL'][i_start_repointing+1]/2.**16
    end_repointing = AHF['OBT_BEG'][i_start_repointing+1]/2.**16
    for start, end in zip(start_repointing,end_repointing):
        flag[obt.searchsorted(start):obt.searchsorted(end)] = 1
    return flag

def quaternion_ecl2gal(qsat):
    '''Convert array of quaternions from Ecliptic to Galactic'''
    l.info('Rotating to Galactic frame')
    qsatgal = qarray.mult(QECL2GAL ,qsat)
    # renormalizing to unity
    qsatgal = qarray.norm(qsatgal)
    return qsatgal

def vector_ecl2gal(vecl):
    '''Convert arrays from Ecliptic to Galactic'''
    l.info('Rotating to Galactic frame')
    return qarray.rotate(QECL2GAL ,vecl)

class Pointing(object):
    '''Pointing interpolation and rotation class
    
    usage:
    >>> ch= Planck()['100-1a']
    >>> pnt = Pointing(obt, coord='G') #interpolates AHF to obt
    >>> vec = pnt.get(ch) #rotates to detector frame and gives x,y,z vector
    >>> pix = pnt.get_pix(ch, 2048, nest=True) #healpix pixel number nside 2048
    '''

    def __init__(self,obt,coord='G', AHF_d=None, nointerp=False):
        '''AHF_d is the pyfits AHF data if already loaded in the main file
        nointerp to use the AHF OBT stamps'''
        l.warning('Pointing setup, coord:%s' % coord)
        #get ahf limits

        if  AHF_d is None:
            AHF_data_iter = (pyfits.open(file)[1].data for file in AHF_btw_OBT(obt))
        else:
            AHF_data_iter = [AHF_d]

        ahfobt = np.array([])
        qsat = None
        for AHF_data in AHF_data_iter:

            obt_spl = AHF_data.field('OBT_SPL')/2.**16
            i_start = max(obt_spl.searchsorted(obt[0])-1,0)
            i_end = min(obt_spl.searchsorted(obt[-1])+1,len(obt_spl)-1)
            AHF = AHF_data[i_start:i_end]

            allquat = np.hstack([AHF.field('QUATERNION_X')[:,np.newaxis], AHF.field('QUATERNION_Y')[:,np.newaxis], AHF.field('QUATERNION_Z')[:,np.newaxis], AHF.field('QUATERNION_S')[:,np.newaxis]])

            if qsat is None:
                qsat = allquat
            else:
                qsat = np.vstack([qsat,allquat])
            ahfobt = np.concatenate([ahfobt, AHF.field('OBT_SPL')/2.**16])

        if coord == 'E':
            qsatgal = qsat
        elif coord == 'G':
            hfobt = np.array([])
            qsatgal = quaternion_ecl2gal(qsat)

        if nointerp:
            self.qsatgal_interp = qsatgal 
        else:
            l.info('Interpolating quaternions')
            #nlerp
            self.qsatgal_interp = qarray.nlerp(obt, ahfobt, qsatgal)

        l.info('Quaternions interpolated')
        self.siam = Siam()

        self.ahfobt = ahfobt
        self.obt = obt

    def interp_get(self, rad):
        '''Interpolation after rotation to gal frame'''
        from Quaternion import Quat
        l.info('Rotating to detector %s' % rad)
        siam_quat = Quat(self.siam.get(rad)).q
        totquat = qarray.mult(self.qsatgal_interp, siam_quat)
        totquat_interp = qarray.nlerp(self.obt, self.ahfobt, totquat)
        x = np.array([[1, 0, 0]]).T
        x = np.array(x).flatten()
        vec = qarray.rotate(totquat_interp, x)
        l.info('Rotated to detector %s' % rad)
        return vec

    def get(self, rad):
        l.info('Rotating to detector %s' % rad)
        x = self.siam.get(rad) * np.array([[1, 0, 0]]).T
        x = np.array(x).flatten()
        vec = qarray.rotate(self.qsatgal_interp, x)
        l.info('Rotated to detector %s' % rad)
        return vec

    def get_pix(self, rad, nside=1024, nest=True):
        from healpy import vec2pix
        vec = self.get(rad)
        return vec2pix(nside, vec[:,0], vec[:,1], vec[:,2], nest)

    def get_ang(self, rad):
        from healpy import vec2ang
        vec = self.get(rad)
        return vec2ang(vec)
