from __future__ import print_function, division
import sys
import numpy as np
import copy
import os
import glob
import sqlite3
from planck.ringdb import RingDB
from collections import OrderedDict

import planck.private as private
from planck.Planck import parse_channels, EXCLUDED_CH, group_by_horn
from collections import defaultdict

import logging as l
from pytoast.core import Run, ParMap

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

l.basicConfig(level=l.INFO)

sim_noise2_nbin = 14345
#sim_noise2_nbin = 7172

def find_AHF(dir, od):
    """Searches the supplied directory for an AHF file with velocity information appended"""
    pattern = '{}/{:04}/vel_att_hist_high_*{:04}*fits'.format( dir, od, od )
    files = glob.glob( pattern )
    if len(files) == 0:
        raise Exception('Found no AHF file matching {}'.format( pattern ))
    
    return sorted( files )[-1]


def find_EFF(dir, od, freq, efftype):
    """Searches the supplied directory for an Exchange File Format file with right frequency and type"""
    pattern = '{}/{:04}/?{:03}?{:04}?{}*fits'.format( dir, od, freq, od, efftype )
    files = glob.glob( pattern )
    if len(files) == 0:
        raise Exception('Found no EFF file matching {}'.format( pattern ))
    
    return sorted( files )[-1]


def get_pair( det ):
    """If the supplied detector shares a horn with another one, return the name of the other detector"""
    pair1 = det.replace('a','b').replace('M','S').replace('A','B')
    pair2 = det.replace('b','a').replace('S','M').replace('B','A')

    if pair1 != det: return pair1
    elif pair2 != det: return pair2
    else: return None


def get_eff_od(file_path):
    return int(os.path.basename(file_path).split('-')[1])


def strconv(f):
    """Formatting for the xml"""
    return "%.16g" % f


def pairsplit(det):
    pix = ""
    first = ""
    second = ""
    if det[-1] not in 'MSab':
        pix = det
    elif ( det.endswith('a') ):
        pix = det.rstrip('a')
        first = "a"
        second = "b"
    elif ( det.endswith('M') ):
        pix = det.rstrip('M')
        first = "M"
        second = "S"
    return pix, first, second

def hornindx( hornkey ):
    if ( hornkey == 27 ):
        return 0
    elif ( hornkey == 28 ):
        return 1
    elif ( hornkey == 24 ):
        return 0
    elif ( hornkey == 25 ):
        return 1
    elif ( hornkey == 26 ):
        return 2
    elif ( hornkey == 18 ):
        return 0
    elif ( hornkey == 19 ):
        return 1
    elif ( hornkey == 20 ):
        return 2
    elif ( hornkey == 21 ):
        return 3
    elif ( hornkey == 22 ):
        return 4
    elif ( hornkey == 23 ):
        return 5
    else:
        hornst = str ( hornkey )
        lastdigit = int( hornst[-1:] )
        if hornst[:3] == '217': lastdigit -= 4 # Bundle SWBs into 2 virtual horns
        if hornst[:3] == '353': lastdigit -= 2 # Bundle SWBs into one virtual horn
        return str( lastdigit - 1 )


def Params(dic=None):
    """Creates a Toast ParMap from a python dictionary"""
    params = ParMap()
    if not dic is None:
        for k,v in dic.items():
            if isinstance(v, float):
                v = strconv(v)
            else:
                v = str(v)
            params[k] = v
    return params

DEFAULT_OBTMASK = {'LFI':1, 'HFI':1}
DEFAULT_FLAGMASK = {'LFI':255, 'HFI':1}

class ToastConfig(object):
    """Toast configuration class"""

    def __init__(self, ringdb, fpdb, 
                 time_range=None, click_range=None, lfi_ring_range=None, hfi_ring_range=None, od_range=None,
                 # Detector information
                 channels=None, extend_143=False, extend_545=False, extend_857=True,
                 psd=None, channel_psd=None, simu_psd=None,
                 output_xml='toastrun.xml', log_level=l.INFO,
                 # Map params
                 nside=1024, ordering='RING', coord='E', components='IQU',
                 nside_des=None,
                 iqus=False, iqus_des=False, extra=None, 
                 # Data directories
                 exchange_folder=None, ahf_folder=None,
                 remote_exchange_folder=None, remote_ahf_folder=None,
                 exchange_weights=None, eff_is_for_flags=False,
                 efftype=None,
                 # Flag selection
                 use_SCM=True, use_HCM=False, use_OCM=False,
                 obtmask=None, flagmask=None, flag_HFI_bad_rings=False,
                 pairflags=None,
                 # pointing corrections
                 ptcorfile=None, no_wobble=False, wobble_high=False, deaberrate=True,
                 calibration_file=None, dipole_removal=False, zodi_removal=False,
                 # Baseline subtraction
                 baseline_file=None, 
                 # Noise simulation
                 noise_tod=False, noise_tod_weight=None,
                 horn_noise_tod=None, horn_noise_weight=None, horn_noise_psd=None, observation_is_interval=False,
                 # Beamsky params
                 beamsky=None, beamsky_weight=None, interp_order=5,
                 sum_diff=False):
        """TOAST configuration:

            ringdb: AHF ring definition database file
            fpdb: Planck Focalplane Database
            channels: one of integer frequency, channel string, list of channel strings
            od_range: list of start and end OD, AHF ODS, i.e. with whole pointing periods as the DPC is using
            lfi_ring_range: first and last LFI pointing ID to include
            hfi_ring_range: first and last HFI ring number to include
            use_SCM(True) : Include Science scans
            use_HCM(False) : Include repointing maneuvers
            use_OCM(False) : Include Orbital Control Maneuvers
            obtmask and flagmask: default LFI 1,255 HFI 1,1
            exchange_folder: either a string or a container or strings listing locations for Exchange Format data
            exchange_weights: list of scaling factors to apply to Exchange data prior to coadding
            eff_is_for_flags : only use Exchange data for flags
            remote_exchange_folder, remote_ahf_folder: they allow to run toast.py in one environment using ahf_folder and exchange_folder and then replace the path with the remote folders
            calibration_file: path to a fits calibration file, with first extension OBT, then one extension per channel with the calibration factors
            baseline_file : path to a Madam or TOAST baseline file to subtract presolved baselines
            dipole_removal: dipole removal is performed ONLY if calibration is specified
            zodi_removal(False) : enable zodiacal light removal
            noise_tod: boolean = Add simulated noise TODs. two-element container, weights for regular and phase-binned noise.
            noise_tod_weight: scaling factor to apply to noise_tod
            flag_HFI_bad_rings: If a valid file, use that as input.
            pairflags(None): Explicitly enable or disable symmetric flagging within horns. Default is LFI(True), HFI(False)
            psd : templated name of ASCII or FITS PSD files. Tag CHANNEL will be replaced with the appropriate channel identifier.
            channel_psd : templated name of PSD file to link to the channel information and supply to client codes
            simu_psd : templated name of PSD file to draw random realizations from when simulating noise
            horn_noise_tod : False
            horn_noise_weight : None
            horn_noise_psd : templated name of an ASCII PSD files. Tag HORN will be replaced with the appropriate identifier.
            deaberrate : Correct pointing for aberration
            extend_143 : Whether or not to include the RTS bolometer, 143-8 in processing
            extend_545 : Whether or not to include the RTS bolometer, 545-3 in processing
            extend_857 : Whether or not to include the RTS bolometer, 857-4 in processing
            no_wobble : Disable all flavors of wobble angle correction
            wobble_high : Use 8Hz wobble correction rather than per ring
            beamsky : templated name of the beamsky files for OTF sky convolution. Tag CHANNEL will be replaced with the appropriate channel identifier.
            beamsky_weight : scaling factor to apply to the beamsky
            interp_order : beamsky interpolation order defines number of cartesian pixels to interpolate over
            sum_diff : If True, build a run containing sum and difference timestreams.
            nside : Healpix NSIDE to use for primary sky representation (default = 1024)
            nside_des : Healpix NSIDE to use for destriping sky representation (default same as nside)
            iqus : If True, generate output map with one spurious map per horn (default False). If 'full', additional spurious maps are added between horns.
            iqus_des : If True, destripe on a map with one spurious map per horn (default False)
            extra : If nonzero, set the number of extra IQUS maps rather than use internal logic

            additional configuration options are available modifying:
            .config
            and Data Selector configuration:
            .data_selector.config
            dictionaries before running .run()
        """
        l.root.level = log_level
        self.extend_143 = extend_143
        if self.extend_143:
            if '143-8' in EXCLUDED_CH: EXCLUDED_CH.remove('143-8')
        self.extend_545 = extend_545
        if self.extend_545:
            if '545-3' in EXCLUDED_CH: EXCLUDED_CH.remove('545-3')
        self.extend_857 = extend_857
        if self.extend_857:
            if '857-4' in EXCLUDED_CH: EXCLUDED_CH.remove('857-4')
        self.channels = parse_channels(channels)
        self.f = self.channels[0].f
        if not os.path.isfile( ringdb ):
            raise Exception('Sorry, unable to open ring database: {}'.format(str(ringdb)))
        self.ringdb = RingDB( ringdb, self.f.freq, time_range=time_range, click_range=click_range, lfi_ring_range=lfi_ring_range, hfi_ring_range=hfi_ring_range, od_range=od_range,
                              use_SCM=use_SCM, use_HCM=use_HCM, use_OCM=use_OCM)
        self.ringdb.apply_breaks()
        self.ringdb.exclude_od( private.bad_ods )
        self.ringdb.exclude_ring( private.bad_rings )
        self.observation_is_interval = observation_is_interval
        # Disabled to allow remote build
        #if not os.path.isfile( fpdb ):
        #    raise Exception('Sorry, unable to open focalplane database: {}'.format(str(fpdb)))
        self.fpdb = fpdb
        self.nside = nside
        if ( nside_des == None ):
            self.nside_des = nside
        else:
            self.nside_des = nside_des
        self.coord = coord
        self.ordering = ordering
        self.iqus = iqus
        self.iqus_des = iqus_des
        self.extra = extra
        if channels == None: raise Exception('Must define which channels to include')
        self.output_xml = output_xml
        self.ptcorfile = ptcorfile
        if no_wobble and wobble_high: raise Exception('no_wobble and wobble_high are mutually exclusive')
        self.no_wobble = no_wobble
        self.wobble_high = wobble_high
        self.sum_diff = sum_diff
        self.deaberrate = deaberrate
        self.noise_tod = noise_tod
        self.noise_tod_weight = noise_tod_weight

        self.psd = psd
        if channel_psd == None:
            self.channel_psd = psd
        else:
            self.channel_psd = channel_psd
        if simu_psd == None:
            self.simu_psd = psd
        else:
            self.simu_psd = simu_psd
        if self.channel_psd == None or self.channel_psd.lower() == 'rimo': self.channel_psd = 'RIMO'
        if self.simu_psd == None or self.simu_psd.lower() == 'rimo': self.simu_psd = 'RIMO'

        self.horn_noise_tod = horn_noise_tod
        self.horn_noise_weight = horn_noise_weight
        self.horn_noise_psd = horn_noise_psd
        if self.horn_noise_tod and not self.horn_noise_psd:
            raise Exception('Must specify horn_noise_psd template name when enabling horn_noise')
        self.beamsky = beamsky
        self.beamsky_weight = beamsky_weight
        self.interp_order = interp_order
        self.rngorder = {
            'LFI18M' : 0,
            'LFI18S' : 1,
            'LFI19M' : 2,
            'LFI19S' : 3,
            'LFI20M' : 4,
            'LFI20S' : 5,
            'LFI21M' : 6,
            'LFI21S' : 7,
            'LFI22M' : 8,
            'LFI22S' : 9,
            'LFI23M' : 10,
            'LFI23S' : 11,
            'LFI24M' : 12,
            'LFI24S' : 13,
            'LFI25M' : 14,
            'LFI25S' : 15,
            'LFI26M' : 16,
            'LFI26S' : 17,
            'LFI27M' : 18,
            'LFI27S' : 19,
            'LFI28M' : 20,
            'LFI28S' : 21,
            '100-1a' : 22,
            '100-1b' : 23,
            '100-2a' : 24,
            '100-2b' : 25,
            '100-3a' : 26,
            '100-3b' : 27,
            '100-4a' : 28,
            '100-4b' : 29,
            '143-1a' : 30,
            '143-1b' : 31,
            '143-2a' : 32,
            '143-2b' : 33,
            '143-3a' : 34,
            '143-3b' : 35,
            '143-4a' : 36,
            '143-4b' : 37,
            '143-5'  : 38,
            '143-6'  : 39,
            '143-7'  : 40,
            '143-8'  : 41,
            '217-5a' : 42,
            '217-5b' : 43,
            '217-6a' : 44,
            '217-6b' : 45,
            '217-7a' : 46,
            '217-7b' : 47,
            '217-8a' : 48,
            '217-8b' : 49,
            '217-1'  : 50,
            '217-2'  : 51,
            '217-3'  : 52,
            '217-4'  : 53,
            '353-3a' : 54,
            '353-3b' : 55,
            '353-4a' : 56,
            '353-4b' : 57,
            '353-5a' : 58,
            '353-5b' : 59,
            '353-6a' : 60,
            '353-6b' : 61,
            '353-1'  : 62,
            '353-2'  : 63,
            '353-7'  : 64,
            '353-8'  : 65,
            '545-1'  : 66,
            '545-2'  : 67,
            '545-3'  : 68,
            '545-4'  : 69,
            '857-1'  : 70,
            '857-2'  : 71,
            '857-3'  : 72,
            '857-4'  : 73,
            'LFI18' : 74,
            'LFI19' : 75,
            'LFI20' : 76,
            'LFI21' : 77,
            'LFI22' : 78,
            'LFI23' : 79,
            'LFI24' : 80,
            'LFI25' : 81,
            'LFI26' : 82,
            'LFI27' : 83,
            'LFI28' : 84,
            '100-1' : 85,
            '100-2' : 86,
            '100-3' : 87,
            '100-4' : 88,
            '143-1' : 89,
            '143-2' : 90,
            '143-3' : 91,
            '143-4' : 92,
            '217-5' : 93,
            '217-6' : 94,
            '217-7' : 95,
            '217-8' : 96,
            '353-3' : 97,
            '353-4' : 98,
            '353-5' : 99,
            '353-6' : 100,
            }

        self.config = {}
        if self.f.inst.name == 'HFI' and pairflags != True:
            self.config['pairflags'] = False
        else:
            self.config['pairflags'] = True

        if efftype is None:
            self.efftype ='R'
            if self.f.inst.name == 'LFI' and (not calibration_file is None):
                self.efftype ='C'
        else:
            self.efftype = efftype

        if not os.path.isdir(str(ahf_folder)):
            raise Exception('No such AHF folder: {}'.format( str(ahf_folder) ) )
        self.ahf_folder = ahf_folder
        self.remote_ahf_folder = remote_ahf_folder
        
        if exchange_folder:
            if isinstance(exchange_folder, str):
                exchange_folder = [exchange_folder]
            for d in exchange_folder:
                if not os.path.isdir(str(d)):
                    raise Exception('No such EFF folder: {}'.format( str(d) ) )
        self.exchange_folder = exchange_folder
        
        if remote_exchange_folder:
            if isinstance(remote_exchange_folder, str):
                remote_exchange_folder = [remote_exchange_folder]
        self.remote_exchange_folder = remote_exchange_folder
        
        self.exchange_weights = exchange_weights

        self.wobble = private.WOBBLE
        self.components = components

        if obtmask != None:
            self.obtmask = obtmask
        else:
            self.obtmask = DEFAULT_OBTMASK[self.f.inst.name]
            
        if flagmask != None:
            self.flagmask = flagmask
        else:
            self.flagmask = DEFAULT_FLAGMASK[self.f.inst.name]

        self.calibration_file = calibration_file
        self.baseline_file = baseline_file
        self.dipole_removal = dipole_removal
        self.zodi_removal = zodi_removal

        # Disable bad ring list check to allow building runs for other systems
        #if flag_HFI_bad_rings and not os.path.isfile(str(flag_HFI_bad_rings)):
        #    raise Exception('Sorry, unable to open HFI bad ring list: {}'.format(str(flag_HFI_bad_rings)))
        self.bad_rings = flag_HFI_bad_rings

        self.eff_is_for_flags = eff_is_for_flags

        self.fptab = None


    def parse_fpdb(self, channel, key):
        if self.fptab == None:
            fptab = pyfits.getdata(self.fpdb, 1)
        ind = (fptab.field('DETECTOR') == channel).ravel()
        if np.sum(ind) != 1:
            raise Exception('Error matching {} in {}: {} matches.'.format(channel, self.fpdb, np.sum(ind)))
        return fptab.field(key)[ind][0]


    def run(self, write=True):
        """Call the python-toast bindings to create the xml configuration file"""
        self.conf = Run()

        if self.noise_tod or self.horn_noise_tod:
            self.conf.variable_add ( "rngbase", "native", Params({"default":"0"}) )
          
        sky = self.conf.sky_add ( "sky", "native", ParMap() )

        hornmaps = 0
        hornlist = []
        for ch in self.channels:
            if ch.tag[-1] not in 'MSab': continue # Only list polarized horns
            h = hornindx ( ch.horn )
            if h not in hornlist:
                hornlist.append( h )
                hornmaps += 1

        # number of virtual horns for iqus mapmaking
        ndet = len(self.channels)
        nvhorn = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11}[ndet]
        nhorn = {30:2, 44:3, 70:6, 100:4, 143:4, 217:4, 353:4, 545:0, 857:0}[self.f.freq]

        extra = 0
        if ( self.iqus ):
            if self.iqus == 'full':
                extra = nvhorn
                if self.f.freq == 30:
                    hornlist = [0,1,6]
                elif self.f.freq == 44:
                    hornlist = [0,1,2,6,9]
                elif self.f.freq == 100:
                    hornlist = [0,1,2,3,6,7,9]
                elif self.f.freq == 143:
                    hornlist = [0,1,2,3,4,6,7,8,9,10]
                elif self.f.freq == 545:
                    hornlist = [0,6]
                elif self.f.freq == 857:
                    hornlist = [0,1,6]
                else:
                    hornlist = range(nvhorn)
            else:
                extra = nhorn # hornmaps

        if self.extra: extra = self.extra

        mapset = sky.mapset_add ( '_'.join(['healpix',self.components, self.ordering]), "healpix", 
            Params({
                "stokes"  : self.components,
                "order"  : self.ordering,
                "coord"  : self.coord,
                "nside"  : str(self.nside),
                "extra"  : str(extra)
            }))

        if ( self.nside != self.nside_des ):

            des_sky = self.conf.sky_add ( "destripe_sky", "native", ParMap() )

            extra = 0
            if self.iqus_des:
                if self.iqus_des == 'full':
                    extra = nvhorn
                else:
                    extra = nhorn # hornmaps

            mapset_des = des_sky.mapset_add ( '_'.join(['destripe_healpix',self.components, self.ordering]), "healpix", 
            Params({
                "stokes"  : self.components,
                "order"  : self.ordering,
                "coord"  : self.coord,
                "nside"  : str(self.nside_des),
                "extra"  : str(extra)
            }))

        if self.no_wobble:
            teleparams = {
                "wobble_ahf_high":"FALSE",
                "wobble_ahf_obs":"FALSE",
                "wobblepsi2dir":""
                }
        else:
            if self.wobble_high:
                wobble_obs = 'FALSE'
                wobble_high = 'TRUE'
            else:
                wobble_obs = 'TRUE'
                wobble_high = 'FALSE'
            
            teleparams = {  
                #"wobblepsi2dir":self.wobble["psi2_dir"],
                "wobblepsi2_ref":self.wobble["psi2_ref"],
                "wobblepsi1_ref":self.wobble["psi1_ref"],
                "wobble_ahf_obs":wobble_obs,
                "wobble_ahf_high":wobble_high
                #"wobblepsi2_offset":wobble_offset
                }

        if self.ptcorfile:
            teleparams['ptcorfile'] = self.ptcorfile

        if self.deaberrate != None:
            if self.deaberrate:
                teleparams['deaberrate'] = 'TRUE'
            else:
                teleparams['deaberrate'] = 'FALSE'

        tele = self.conf.telescope_add ( "planck", "planck", 
            Params(teleparams))

        #if ( hornmaps > 0 ):
        if ( len(hornlist) > 0 ):
            hornstrlist = [ str(x) for x in sorted ( hornlist ) ]
            hornstr = ",".join( hornstrlist )
            fp = tele.focalplane_add ( "FP_%s" % self.f.inst.name, "planck_rimo", Params({"path":self.fpdb, "freq":self.f.freq, "horns":hornstr}) )
        else:
            fp = tele.focalplane_add ( "FP_%s" % self.f.inst.name, "planck_rimo", Params({"path":self.fpdb, "freq":self.f.freq}) )

        self.add_pointing(tele)
        self.add_observations(tele)
        self.add_streams()
        self.add_eff_tods()
        self.add_noise()
        self.add_channels(tele)
        if write:
            self.write()


    def write(self):
        # write out XML
        self.conf.write ( self.output_xml )


    def add_pointing(self, telescope):
        # Add pointing files
        for i, od in enumerate( self.ringdb.ods ):
            path = find_AHF( self.ahf_folder, od )
            if self.remote_ahf_folder: path = path.replace(self.ahf_folder, self.remote_ahf_folder)
                
            telescope.pointing_add ( "%04d" % i, "planck_ahf", Params({"path": path}))


    def add_observations(self, telescope):
        """Each observation is an OD as specified in the AHF files"""
        # Add streamset
        self.strset = telescope.streamset_add ( self.f.inst.name, "native", Params() )
          
        # Add observations

        # For Planck, each observation must cache the timestamps for all days included
        # in the observation, which is very expensive.  For this reason, the number of
        # ODs in a single observation should be the minimum necessary for the size of
        # the interval being considered.  For applications which do not care about noise
        # stationarity or which are only dealing with ring-by-ring quantities, one OD
        # per observation is fine.  If you wish to consider intervals longer than a day
        # as a single stationary interval, then each observation will need to include
        # multiple ODs.

        # For "planck_exchange" format, if no "start" or "stop" times are specified,
        # then the observation will span the time range of the EFF data specified in
        # the "times1", "times2", "times3", etc parameters.

        # Observations are same for all datasets but now there may be
        # several exchange folders and raw streams to add. Therefore
        # tod_name_list and tod_par_list are lists of dictionaries.
        
        self.tod_name_list = []
        self.tod_par_list = []
        for i in range(len(self.exchange_folder)):
            self.tod_name_list.append( defaultdict(list) )
            self.tod_par_list.append( defaultdict(list) )
            
        # Add observations

        for iobs, observation in enumerate(self.ringdb.observations):
            params = {'start':observation['start_time'], 'stop':observation['stop_time']}

            eff_files = OrderedDict()
            eff_ods = self.ringdb.get_eff_ods(observation['ods']) # dictionary of dictionaries
            print('Observation {} ods: '.format(observation['id']), end='')
            for od in observation['ods']:
                print(' {}'.format(od), end='')
            print(' EFF ods: ', end='')
            for eff_od in eff_ods.keys():
                print(' {}'.format(eff_od), end='')
            print()
            earliest = 1e30
            latest = -1e30
            for eff_od, oddict in eff_ods.items():
                earliest = min( earliest, oddict['start_time'] )
                latest = max( latest, oddict['stop_time'] )
                eff_files[ find_EFF(self.exchange_folder[0], eff_od, self.f.freq, self.efftype) ] = True
            print(eff_files.keys())
            eff_files = eff_files.keys()
            for i, eff in enumerate( eff_files ):
                if self.remote_exchange_folder:
                    params[ "times%d" % (i+1) ] = eff.replace(self.exchange_folder[0], self.remote_exchange_folder[0])
                else:
                    params[ "times%d" % (i+1) ] = eff

            if 'nrow' in observation:
                params['latest'] = observation['stop_time']
            else:
                params['latest'] = latest

            if 'start_row' in observation:
                params['earliest'] = observation['start_time']
            else:
                params['earliest'] = earliest
                    
            obs = self.strset.observation_add ( observation['id'] , "planck_exchange", Params(params) )

            if not self.observation_is_interval:
                for interval in observation['intervals']:
                    obs.interval_add( interval['id'], 'native', Params({'start':interval['start_time'], 'stop':interval['stop_time']}) )

            for ix, exchange_folder in enumerate(self.exchange_folder):
                eff_files = OrderedDict()
                for eff_od, oddict in eff_ods.items():
                    path = find_EFF(exchange_folder, eff_od, self.f.freq, self.efftype)
                    if self.remote_exchange_folder:
                        path = path.replace(exchange_folder, self.remote_exchange_folder[ix])
                    oddict['path'] = path
                    
                for ch in self.channels:
                    for i, oddict in enumerate(eff_ods.values()):
                        eff_od = oddict['eff_od']
                        file_path = oddict['path']
                        # Add TODs for this stream
                        params = {}
                        params[ 'start_time' ] = oddict[ 'start_time' ]
                        params[ 'stop_time' ] = oddict[ 'stop_time' ]
                        params[ 'rows' ] = oddict[ 'nrow' ]
                        params[ 'flagmask' ] = self.flagmask
                        params[ 'obtmask' ] = self.obtmask
                        params[ 'hdu_name' ] = ch.eff_tag
                        params[ 'hdu' ] = private.hdu_numbers[ ch.tag ]
                        if self.config[ 'pairflags' ]:
                            params[ 'pairflags' ] = 'TRUE'
                            pairtag = get_pair( ch.tag )
                            paireff_tag = get_pair( ch.eff_tag )
                            if pairtag and paireff_tag:
                                params[ 'pair_hdu_name' ] = paireff_tag
                                params[ 'pair_hdu' ] = private.hdu_numbers[ pairtag ]
                                
                        #if self.remote_exchange_folder:
                        #    params[ 'path' ] = file_path.replace(self.exchange_folder[0], self.remote_exchange_folder[ix])
                        #else:
                        #    params[ 'path' ] = file_path.replace(self.exchange_folder[0], self.exchange_folder[ix])
                        params[ 'path' ] = file_path
                            
                        tag = ''
                        if 'eff_od' in observation and observation['eff_od'] == eff_od:
                            # Only use part of the EFF file due to break
                            if 'nrow' in observation:
                                params['rows'] = observation['nrow']
                                params['stop_time'] = observation['stop_time']
                                tag = 'a'
                            if 'start_row' in observation:
                                params['startrow'] = observation['start_row']
                                params['start_time'] = observation['start_time']
                                params['rows'] -= observation['start_row']
                                tag = 'b'
                        name = "%s_%d%s" % (ch.tag, eff_od, tag)
                        if name not in self.tod_name_list[ix][ch.tag]:
                            print('add ' + name)
                            self.tod_name_list[ix][ch.tag].append(name)
                            self.tod_par_list[ix][ch.tag].append(params)
                        else:
                            print("skip " + name)


    def add_streams(self):
        # Add streams for data components
        basename = "@rngbase@"

        self.strm = {}

        for ch in self.channels:
            stack_elements = []

            # Noise TOD oversample factors:
            if self.f.freq < 400:
                oversample = 2
            else:
                oversample = 4

            # add simulated noise stream
            if self.noise_tod:
                rngstream = self.rngorder[ ch.tag ] * 100000

                if self.channel_psd != self.simu_psd:
                    noisename = "/planck/" + self.f.inst.name + "/noise_simu_" + ch.tag
                else:
                    noisename = "/planck/" + self.f.inst.name + "/noise_" + ch.tag
                self.strm["simnoise_" + ch.tag] = self.strset.stream_add( "simnoise_" + ch.tag, "native", Params( ) )
                suffix = ''
                if self.noise_tod_weight != None and self.noise_tod_weight != 1:
                    suffix += ',PUSH:$' + strconv(self.noise_tod_weight) + ',MUL'
                stack_elements.append( "PUSH:simnoise_" + ch.tag + suffix)
                # one noise tod per pointing period
                # Must retrieve all pp_boundaries to properly increment the stream index
                # From now on, we always combine HCM and SCM into a single noise tod object
                hcm = None
                for interval in self.ringdb.detector_intervals:
                    if 'H' in interval['id']:
                        if hcm != None or interval == self.ringdb.detector_intervals[-1]:
                            if hcm == None: hcm = interval
                            # write out orphan HCM
                            nn = 0
                            try:
                                nn = len( self.noise_tod )
                                if nn != 2: raise Exception()
                                # if we made it this far, user gave sim_noise2 weights
                                self.strm["simnoise_" + ch.tag].tod_add ( "nse_%s_%05d" % (ch.tag, hcm['index']), "sim_noise2", Params({
                                            "noise" : noisename,
                                            "base" : basename,
                                            "start" : hcm['start_time'],
                                            "stop" : hcm['stop_time'],
                                            "offset" : rngstream + hcm['index'],
                                            "oversample" : oversample,
                                            "weight1" : self.noise_tod[0],
                                            "weight2" : self.noise_tod[1],
                                            "nbin"   : sim_noise2_nbin,
                                        }))
                            except:
                                if nn == 2: raise
                                # User wants regular noise, not phase binned
                                self.strm["simnoise_" + ch.tag].tod_add ( "nse_%s_%05d" % (ch.tag, hcm['index']), "sim_noise", Params({
                                            "noise" : noisename,
                                            "base" : basename,
                                            "start" : hcm['start_time'],
                                            "stop" : hcm['stop_time'],
                                            "offset" : rngstream + hcm['index'],
                                            "oversample" : oversample,
                                        }))
                            hcm = None
                        hcm = interval
                    else:
                        if hcm == None: hcm = interval
                        nn = 0
                        try:
                            nn = len( self.noise_tod )
                            if nn != 2: raise Exception()
                            # if we made it this far, user gave sim_noise2 weights
                            self.strm["simnoise_" + ch.tag].tod_add ( "nse_%s_%05d" % (ch.tag, interval['index']), "sim_noise2", Params({
                                        "noise" : noisename,
                                        "base" : basename,
                                        "start" : hcm['start_time'],
                                        "stop" : interval['stop_time'],
                                        "offset" : rngstream + interval['index'],
                                        "oversample" : oversample,
                                        "weight1" : self.noise_tod[0],
                                        "weight2" : self.noise_tod[1],
                                        "nbin"   : sim_noise2_nbin,
                                    }))
                        except:
                            if nn == 2: raise
                            self.strm["simnoise_" + ch.tag].tod_add ( "nse_%s_%05d" % (ch.tag, interval['index']), "sim_noise", Params({
                                        "noise" : noisename,
                                        "base" : basename,
                                        "start" : hcm['start_time'],
                                        "stop" : interval['stop_time'],
                                        "offset" : rngstream + interval['index'],
                                        "oversample" : oversample,
                                        }))
                        hcm = None

            # add simulated noise stream common to each horn
            if self.horn_noise_tod and ch.tag[-1] in 'MSab':
                horn = ch.tag[:-1] 
                rngstream = self.rngorder[ horn ] * 100000
                chan = ch.tag
                pair = get_pair( chan )
                if chan[-1] in 'Ma':
                    horn = chan + '_' + pair
                else:
                    horn = pair + '_' + chan

                noisename = "/planck/" + self.f.inst.name + "/noise_" + horn
                self.strm["simnoise_" + horn] = self.strset.stream_add( "simnoise_" + horn, "native", Params( ) )
                suffix = ''
                if self.horn_noise_weight and self.horn_noise_weight != 1:
                    suffix += ',PUSH:$' + strconv(self.horn_noise_weight) + ',MUL'
                if len(stack_elements) != 0:
                    suffix += ',ADD'
                stack_elements.append( "PUSHDATA:simnoise_" + horn + suffix)
                # one noise tod per pointing period

                hcm = None
                for interval in self.ringdb.detector_intervals:
                    if 'H' in interval['id']:
                        if hcm != None or interval == self.ringdb.detector_intervals[-1]:
                            if hcm == None: hcm = interval
                            # write out orphan HCM
                            self.strm["simnoise_" + horn].tod_add ( "nse_%s_%05d" % (horn, hcm['index']), "sim_noise", Params({
                                        "noise" : noisename,
                                        "base" : basename,
                                        "start" : hcm['start_time'],
                                        "stop" : hcm['stop_time'],
                                        "offset" : rngstream + hcm['index'],
                                        "oversample" : oversample,
                                    }))
                            hcm = None
                        hcm = interval
                    else:
                        if hcm == None: hcm = interval
                        self.strm["simnoise_" + horn].tod_add ( "nse_%s_%05d" % (horn, interval['index']), "sim_noise", Params({
                                    "noise" : noisename,
                                    "base" : basename,
                                    "start" : hcm['start_time'],
                                    "stop" : interval['stop_time'],
                                    "offset" : rngstream + interval['index'],
                                    "oversample" : oversample,
                                    }))
                        hcm = None

            # add the beam sky
            if self.beamsky:
                epsilon = self.parse_fpdb(ch.tag, 'epsilon')
                #beamskyname = '/planck/' + self.f.inst.name + '/' + ch.tag
                beamskyname = '/planck/' + ch.tag
                self.strm["beamsky_" + ch.tag] = self.strset.stream_add( "beamsky_" + ch.tag, "planck_beamsky", Params({
                    'path' : self.beamsky.replace('CHANNEL', ch.tag),
                    'epsilon' : str(epsilon),
                    'interp_order' : str(self.interp_order),
                    'coord' : self.coord,
                    'channel' : beamskyname
                    }) )
                suffix = ''
                if self.beamsky_weight and self.beamsky_weight != 1:
                    suffix += ',PUSH:$' + strconv(self.beamsky_weight) + ',MUL'
                if len(stack_elements) != 0:
                    suffix += ',ADD'
                stack_elements.append( "PUSHDATA:beamsky_" + ch.tag + suffix)

            # add real data streams, either for flags or data plus flags
            for i, x in enumerate(self.exchange_folder):
                self.strm[ 'raw{}_{}'.format(i, ch.tag) ] = self.strset.stream_add( "raw{}_{}".format(i, ch.tag), "native", Params() )
                suffix = ''
                if self.eff_is_for_flags:
                    stack_elements.append( "PUSHFLAG:raw{}_{}{}".format(i, ch.tag, suffix) )
                else:
                    if self.exchange_weights and self.exchange_weights[i] != 1:
                        suffix = ',PUSH:$' + strconv(self.exchange_weights[i]) + ',MUL'
                    if len(stack_elements) != 0:
                        suffix += ',ADD'
                    stack_elements.append( "PUSH:raw{}_{}{}".format(i, ch.tag, suffix) )

            # add calibration stream
            if (not self.calibration_file is None):
                self.strm["cal_" + ch.tag] = self.strset.stream_add( "cal_" + ch.tag, "planck_cal", Params( {"hdu":ch.tag, "path":self.calibration_file } ) )
                stack_elements.append("PUSHDATA:cal_" + ch.tag + ",MUL")
            
            # add baseline stream
            if (not self.baseline_file is None):
                self.strm["baseline_" + ch.tag] = self.strset.stream_add( "baseline_" + ch.tag, "baseline", Params( {"row":-1, "rowname":ch.tag, "path":self.baseline_file.replace('CHANNEL',ch.tag) } ) )
                stack_elements.append("PUSH:baseline_" + ch.tag + ",SUB")
            
            # dipole subtract
            if not self.sum_diff:
                # Always add all three types of dipole streams, even if only one is included in the stack
                self.strm["orbital_dipole_" + ch.tag] = self.strset.stream_add( "orbital_dipole_" + ch.tag, "dipole", Params( {"channel":ch.tag, "coord":"E", "type":"ORBITAL"} ) )
                self.strm["solsys_dipole_" + ch.tag] = self.strset.stream_add( "solsys_dipole_" + ch.tag, "dipole", Params( {"channel":ch.tag, "coord":"E", "type":"SOLSYS"} ) )
                self.strm["dipole_" + ch.tag] = self.strset.stream_add( "dipole_" + ch.tag, "dipole", Params( {"channel":ch.tag, "coord":"E"} ) )
                if self.dipole_removal:
                    if self.dipole_removal == 'ORBITAL':
                        stack_elements.append("PUSHDATA:orbital_dipole_" + ch.tag + ",SUB")
                    if self.dipole_removal == 'SOLSYS':
                        stack_elements.append("PUSHDATA:solsys_dipole_" + ch.tag + ",SUB")
                    else:
                        stack_elements.append("PUSHDATA:dipole_" + ch.tag + ",SUB")

            # zodi subtract
            if self.zodi_removal and not self.sum_diff:
                self.strm["zodi_" + ch.tag] = self.strset.stream_add( "zodi_" + ch.tag, "zodi", Params( {"channel":ch.tag, "coord":"E", "emissivity":"1.0"} ) )
                stack_elements.append("PUSHDATA:zodi_" + ch.tag + ",SUB")

            # bad rings
            if self.bad_rings:
                self.strm["bad_" + ch.tag] = self.strset.stream_add ( "bad_" + ch.tag, "planck_bad", Params({'detector':ch.tag, 'path':self.bad_rings}) )
                stack_elements.append("PUSHFLAG:bad_" + ch.tag)

            # stack
            expr = ','.join([el for el in stack_elements])
            self.strm["stack_" + ch.tag] = self.strset.stream_add ( "stack_" + ch.tag, "stack", Params( {"expr":expr} ) )

            # add sum/diff stacks
            if ( self.sum_diff ):
                pix, first, second = pairsplit(ch.tag)
                if ( first != "" ):
                    if 'I' in self.components:
                        try:
                            # Optimize the weights to minimize P->T leakage
                            ipair = [ c.tag for c in self.channels ].index( pix+second )
                            chpair = self.channels[ ipair ]
                            eps1 = ch.rimo['EPSILON']
                            eps2 = chpair.rimo['EPSILON']
                            eta1 = ( 1 - eps1 ) / ( 1 + eps1 )
                            eta2 = ( 1 - eps2 ) / ( 1 + eps2 )
                            w1 = eta2 / ( eta1 + eta2 )
                            w2 = eta1 / ( eta1 + eta2 )
                        except:
                            w1, w2 = None, None
                        if w1 is not None:
                            expr = "PUSH:stack_{},PUSH:${},MUL,PUSH:stack_{},PUSH:${},MUL,ADD".format( pix+first, w1, pix+second, w2)
                        else:
                            expr = "PUSH:stack_%s,PUSH:stack_%s,ADD" % ( pix+first, pix+second)
                        # Always add the dipole streams, even if they are not included on the stack
                        self.strm["orbital_dipole_" + pix] = self.strset.stream_add( "orbital_dipole_" + pix, "dipole", Params( {"channel":pix+'_sum', "coord":"E", "type":"ORBITAL"} ) )
                        self.strm["solsys_dipole_" + pix] = self.strset.stream_add( "solsys_dipole_" + pix, "dipole", Params( {"channel":pix+'_sum', "coord":"E", "TYPE":"SOLSYS"} ) )
                        self.strm["dipole_" + pix] = self.strset.stream_add( "dipole_" + pix, "dipole", Params( {"channel":pix+'_sum', "coord":"E"} ) )
                        if self.dipole_removal:
                            if self.dipole_removal == 'ORBITAL':
                                expr += ",PUSHDATA:orbital_dipole_%s,SUB" % ( pix )
                            elif self.dipole_removal == 'SOLSYS':
                                expr += ",PUSHDATA:solsys_dipole_%s,SUB" % ( pix )
                            else:
                                expr += ",PUSHDATA:dipole_%s,SUB" % ( pix )
                        if self.zodi_removal:
                            self.strm["zodi_" + pix] = self.strset.stream_add( "zodi_" + pix, "zodi", Params( {"channel":pix+'_sum', "coord":"E", "emissivity":"1.0"} ) )
                            expr += ",PUSHDATA:zodi_%s,SUB" % ( pix )
                        self.strm["sum_" + pix] = self.strset.stream_add ( "sum_" + pix, "stack", Params( { 'expr' : expr } ) )
                    if 'QU' in self.components:
                        self.strm["diff_" + pix] = self.strset.stream_add ( "diff_" + pix, "stack", Params( { 'expr' : "PUSH:stack_%s,PUSH:stack_%s,SUB" % ( pix+first, pix+second) } ) )


    def add_eff_tods(self):
        """Add TOD files already included in tod_name_list and tod_par_list to the streamset"""
        for ix in range(len(self.exchange_folder)):
            # remove duplicate files on breaks
            for ch in self.channels:
                for name in [n for n in self.tod_name_list[ix][ch.tag] if n.endswith(('a','b'))]:
                    try:
                        delindex = self.tod_name_list[ix][ch.tag].index(name[:-1])
                        print('Removing %s because of breaks' % self.tod_name_list[ix][ch.tag][delindex])
                        del self.tod_name_list[ix][ch.tag][delindex]
                        del self.tod_par_list[ix][ch.tag][delindex]
                    except ValueError:
                        pass

            # add EFF to stream
            for ch in self.channels:
                for name, par in zip(self.tod_name_list[ix][ch.tag], self.tod_par_list[ix][ch.tag]):
                    self.strm[ "raw{}_{}".format(ix, ch.tag) ].tod_add ( name, "planck_exchange", Params(par) )


    def add_noise(self):
        # Add RIMO noise model (LFI) or mission average PSD (HFI)

        for ch in self.channels:

            if self.channel_psd != self.simu_psd: 
                noises = [self.strset.noise_add ( "noise_channel_" + ch.tag, "native", Params() )]
                noises.append( self.strset.noise_add ( "noise_simu_" + ch.tag, "native", Params() ) )
                psds = [self.channel_psd]
                psds.append( self.simu_psd )
            else:
                noises = [self.strset.noise_add ( "noise_" + ch.tag, "native", Params() )]
                psds = [self.channel_psd]

            # add PSD
            for noise, psd in zip( noises, psds ):
                if psd == None or psd.upper() == 'RIMO':
                    noise.psd_add ( "psd", "planck_rimo", Params({
                                "start" : self.strset.observations()[0].start(),
                                "stop" : self.strset.observations()[-1].stop(),
                                "path": self.fpdb,
                                "detector": ch.tag
                                }))
                else:
                    psdname = psd.replace('CHANNEL', ch.tag)
                    if 'fits' in psdname:
                        if False:
                            # One PSD for the entire span
                            noise.psd_add ( "psd", "fits", Params({
                                        "start" : self.strset.observations()[0].start(),
                                        "stop" : self.strset.observations()[-1].stop(),
                                        "path": psdname
                                        }))
                        else:
                            # Separate PSD object for every observation
                            for obs in self.strset.observations():
                                noise.psd_add ( "psd", "fits", Params({
                                            "start" : obs.start(),
                                            "stop" : obs.stop(),
                                            "path": psdname
                                            }))
                    else:
                        noise.psd_add ( "psd", "ascii", Params({
                                    "start" : self.strset.observations()[0].start(),
                                    "stop" : self.strset.observations()[-1].stop(),
                                    "path": psdname
                                    }))

            # add horn noise psd
            if self.horn_noise_tod and ch.tag[-1] in 'aM':
                horn = ch.tag[:-1]
                psdname = self.horn_noise_psd.replace('HORN', horn)
                #pix, first, second = pairsplit(ch.tag)
                chan = ch.tag
                pair = get_pair( chan )
                horn = chan + '_' + pair
                noise = self.strset.noise_add ( "noise_" + horn, "native", Params() )
                if 'fits' in psdname:
                    if False:
                        # One PSD for the entire span
                        noise.psd_add ( "psd", "fits", Params({
                                    "start" : self.strset.observations()[0].start(),
                                    "stop" : self.strset.observations()[-1].stop(),
                                    "path": psdname
                                    }))
                    else:
                        # Separate PSD object for every observation
                        for obs in self.strset.observations():
                            noise.psd_add ( "psd", "fits", Params({
                                        "start" : obs.start(),
                                        "stop" : obs.stop(),
                                        "path": psdname
                                        }))
                else:
                    # ASCII format only support single PSD
                    noise.psd_add ( "psd", "ascii", Params({
                        "start" : self.strset.observations()[0].start(),
                        "stop" : self.strset.observations()[-1].stop(),
                        "path": psdname
                        }))

            # for sum / difference case, we only use RIMO filters
            if ( self.sum_diff ):
                pix, first, second = pairsplit(ch.tag)
                if ( first != "" ):
                    noise = self.strset.noise_add ( "noise_" + pix + "_sum", "native", Params() )
                    wp = noise.psd_add ( "psd", "planck_rimo", Params( { 
                        "start" : self.strset.observations()[0].start(),
                        "stop" : self.strset.observations()[-1].stop(),
                        "path": self.fpdb,
                        "detector" : pix + "sum"
                        }))
                    noise = self.strset.noise_add ( "noise_" + pix + "_diff", "native", Params() )
                    wp = noise.psd_add ( "psd", "planck_rimo", Params( { 
                        "start" : self.strset.observations()[0].start(),
                        "stop" : self.strset.observations()[-1].stop(),
                        "path": self.fpdb,
                        "detector" : pix + "dif"
                        }))


            
    def add_channels(self, telescope):
        params = ParMap()
        params[ "focalplane" ] = self.conf.telescopes()[0].focalplanes()[0].name()
        for ch in self.channels:
            if ( self.sum_diff ) and ( ch.tag[-1] in 'Sb' ): continue

            pix, first, second = pairsplit(ch.tag)
            if ( self.sum_diff and ( first != "" ) ):
                # we have the first detector of a pair, and want the sum/diff
                if 'I' in self.components:
                    params[ "detector" ] = pix + "_sum"
                    params[ "stream" ] = "DEFAULT:/planck/%s/sum_%s" % (self.f.inst.name, pix) + ',ORBITAL_DIPOLE:/planck/{}/orbital_dipole_{}'.format(self.f.inst.name, pix) + ',SOLSYS_DIPOLE:/planck/{}/solsys_dipole_{}'.format(self.f.inst.name, pix) + ',DIPOLE:/planck/{}/dipole_{}'.format(self.f.inst.name, pix)
                    params[ "noise" ] = "/planck/%s/noise_%s" % (self.f.inst.name, pix + "_sum")
                    telescope.channel_add ( pix + "_sum", "native", params )
                if 'QU' in self.components:
                    params[ "detector" ] = pix + "_diff"
                    params[ "stream" ] = "/planck/%s/diff_%s" % (self.f.inst.name, pix)
                    params[ "noise" ] = "/planck/%s/noise_%s" % (self.f.inst.name, pix + "_diff")
                    telescope.channel_add ( pix + "_diff", "native", params )
            else:
                # we have a spiderweb or are not doing sum/diff
                params[ "detector" ] = ch.tag
                params[ "stream" ] = "DEFAULT:/planck/%s/stack_%s" % (self.f.inst.name, ch.tag) + ',ORBITAL_DIPOLE:/planck/{}/orbital_dipole_{}'.format(self.f.inst.name, ch.tag) + ',SOLSYS_DIPOLE:/planck/{}/solsys_dipole_{}'.format(self.f.inst.name, ch.tag) + ',DIPOLE:/planck/{}/dipole_{}'.format(self.f.inst.name, ch.tag)
                if self.channel_psd != self.simu_psd:
                    params[ "noise" ] = "/planck/%s/noise_channel_%s" % (self.f.inst.name, ch.tag)
                else:
                    params[ "noise" ] = "/planck/%s/noise_%s" % (self.f.inst.name, ch.tag)
                telescope.channel_add ( ch.tag, "native", params )


if __name__ == '__main__':

    # Default LFI run

    #conf = ToastConfig([740, 760], 30, nside=1024, ordering='RING', coord='E', efftype='R', output_xml='test_30_default.xml')
    conf = ToastConfig( 'db.db', 'LFI_RIMO_18092012_FFP6_v1_PTCOR6_hornpnt.fits',
                        od_range=[191,192], channels=['LFI28M'], ahf_folder='AHF', exchange_folder='EFF',
                        noise_tod=True, eff_is_for_flags=False)
    conf.ringdb.exclude_od( 191 )
    conf.run()

#    # LFI run with noise simulation and real data flags
#    
    #conf = ToastConfig([450, 460], 30, nside=1024, ordering='RING', coord='E', efftype='C', output_xml='test_30_simnoise.xml', noise_tod=True)
    #conf.run()
#
#    # LFI real data run with calibration and dipole subtraction
#
#    conf = ToastConfig([97, 101], 30, nside=1024, ordering='RING', coord='E', efftype='C', output_xml='test_30_dical.xml', calibration_file='/project/projectdirs/planck/data/mission/calibration/dx7/lfi/369S/C030-0000-369S-20110713.fits', dipole_removal=True)
#    conf.run()
#
#    # LFI noise simulation with real flags, calibration and dipole subtraction
#
#    conf = ToastConfig([97, 101], 30, nside=1024, ordering='RING', coord='E', efftype='C', output_xml='test_30_simdical.xml', calibration_file='/project/projectdirs/planck/data/mission/calibration/dx7/lfi/369S/C030-0000-369S-20110713.fits', dipole_removal=True, noise_tod=True)
#    conf.run()
#
#    # HFI default run
#
#    conf = ToastConfig([97, 101], 100, nside=2048, ordering='NEST', coord='G', output_xml='test_100_default.xml')
#    conf.run()
#
#    # HFI noise simulation
#
#    conf = ToastConfig([97, 101], 100, nside=2048, ordering='NEST', coord='G', output_xml='test_100_simnoise.xml', noise_tod=True)
#    conf.run()
#
#    # HFI real data run with calibration and dipole subtraction
#
#    conf = ToastConfig([97, 101], 100, nside=2048, ordering='NEST', coord='G', output_xml='test_100_dical.xml', calibration_file='/project/projectdirs/planck/data/mission/calibration/dx7/hfi/variable_gains_100GHz.fits', dipole_removal=True)
#    conf.run()
#
#    # HFI noise simulation with real flags, calibration and dipole subtraction
#
#    conf = ToastConfig([97, 101], 100, nside=2048, ordering='NEST', coord='G', output_xml='test_100_simdical.xml', calibration_file='/project/projectdirs/planck/data/mission/calibration/dx7/hfi/variable_gains_100GHz.fits', dipole_removal=True, noise_tod=True)
#    conf.run()
#
