import sys
import numpy as np
import os
import sqlite3
import copy

from collections import OrderedDict


class RingDB:
    def __init__(self, dbfile, freq, time_range=None, click_range=None,
                 lfi_ring_range=None, hfi_ring_range=None, od_range=None,
                 use_SCM=True, use_HCM=False, use_OCM=False):
        """Sets up the RingDB object to perform queries and return matching
        ring and operational day definitions"""

        if not os.path.isfile( dbfile ):
            raise Exception('ERROR: no such file: {}'.format( str(dbfile) ) )
        self.dbfile = dbfile

        if freq not in ( 30, 44, 70, 100, 143, 217, 353, 545, 857 ):
            raise Exception('ERROR: invalid frequency: {}'.format(str(freq)))
        self.freq = freq

        self.use_SCM = use_SCM
        self.use_HCM = use_HCM
        self.use_OCM = use_OCM

        conn = sqlite3.connect( self.dbfile )
        c = conn.cursor()

        # Convert the supplied range to canonical span definition

        # Perform the ring query here for all range types, even though it is
        # not strictly needed in all cases.

        cmd = None

        if freq < 100:
            extra = ''
        else:
            extra = ' AND start_time >= 1628777816.9397 AND stop_time <= 1705157618.21449'

        if time_range != None:
            cmd = 'select start_time, stop_time from rings where start_time >= {r[0]} and stop_time <= {r[1]}{e} order by start_time'.format( r=time_range, e=extra )
        elif click_range != None:
            cmd = 'select start_time, stop_time from rings where start_time >= {r[0]} and stop_time <= {r[1]}{e} order by start_time'.format( r=np.array(time_range)*2**-16, e=extra )
        elif lfi_ring_range != None:
            cmd = 'select start_time, stop_time from rings where LFI_ID >= {r[0]} and LFI_ID <= {r[1]}{e} order by start_time'.format( r=lfi_ring_range, e=extra )
        elif hfi_ring_range != None:
            cmd = 'select start_time, stop_time from rings where HFI_ID >= {r[0]} and HFI_ID <= {r[1]}{e} order by start_time'.format( r=hfi_ring_range, e=extra )
        elif od_range != None:
            cmd = 'select start_time, stop_time from rings where start_od >= {r[0]} and stop_od <= {r[1]}{e} order by start_time'.format( r=od_range, e=extra )
        else:
            print 'Warning: No range specified, mapping the entire database'
            cmd = 'select start_time, stop_time from rings order by start_time'
            
        query = c.execute( cmd )
        self.rings = query.fetchall()
        self.start = self.rings[0][0]
        self.stop = self.rings[-1][-1]
        self.range = [self.start, self.stop]



    def apply_breaks(self):
        """Breaks the observations and intervals according the the eff_breaks table"""
        conn = sqlite3.connect( self.dbfile )
        c = conn.cursor()

        if self.freq < 100:
            freq = self.freq
        else:
            freq = 100
            
        cmd = 'select eff_od, start_time, stop_time, start_row, stop_row, start_od, stop_od ' \
              'from eff_breaks where freq == {} and start_time < {} and stop_time > {}'.format(
            freq, self.stop, self.start)
            
        query = c.execute( cmd )
        for q in query:
            eff_od, start_time, stop_time, start_row, stop_row, start_od, stop_od = q

            # Apply breaks to self.intervals
            iint = 0
            while iint < len(self.intervals):
                interval = copy.deepcopy( self.intervals[iint] )
                if interval['start_time'] < stop_time and interval['stop_time'] > start_time:
                    print 'Breaking interval {}'.format(interval['id'])
                    # Break this interval
                    if start_time <= interval['start_time'] and stop_time >= interval['stop_time']:
                        # interval is contained in the break
                        del self.intervals[iint]
                        iint -= 1
                    elif start_time <= interval['start_time']:
                        # break covers the beginning of the interval
                        self.intervals[iint]['start_time'] = stop_time
                        break
                    elif stop_time >= interval['stop_time']:
                        # break covers the end of the interval
                        self.intervals[iint]['stop_time'] = start_time
                    else:
                        # the break is contained in the interval
                        interval['stop_time'] = start_time
                        interval['id'] += 'a'
                        self.intervals[iint]['start_time'] = stop_time
                        self.intervals[iint]['id'] += 'b'
                        self.intervals.insert(iint, interval)
                        break
                iint += 1
                
            # Apply the breaks to self.observations, reset the interval lists to be populated later
            iobs = 0
            while iobs < len(self.observations):
                observation = copy.deepcopy( self.observations[iobs] )
                if observation['start_time'] < stop_time and observation['stop_time'] > start_time:
                    # Break this observation
                    print 'Breaking observation {}'.format(observation['id'])
                    if start_time <= observation['start_time'] and stop_time >= observation['stop_time']:
                        # observation is contained in the break
                        del self.observations[iobs]
                        iobs -= 1
                    elif start_time <= observation['start_time']:
                        # break covers the beginning of the interval
                        self.observations[iobs]['start_time'] = stop_time
                        self.observations[iobs]['intervals'] = None
                        self.observations[iobs]['start_row'] = stop_row
                        self.observations[iobs]['eff_od'] = eff_od
                        break
                    elif stop_time >= observation['stop_time']:
                        # break covers the end of the interval
                        self.observations[iobs]['stop_time'] = start_time
                        self.observations[iobs]['intervals'] = None
                        self.observations[iobs]['nrow'] = start_row + 1
                        self.observations[iobs]['eff_od'] = eff_od
                    else:
                        # the break is contained in the observation
                        self.observations[iobs]['start_time'] = stop_time
                        self.observations[iobs]['start_row'] = stop_row
                        self.observations[iobs]['id'] += 'b'
                        self.observations[iobs]['intervals'] = None
                        self.observations[iobs]['eff_od'] = eff_od
                        observation['stop_time'] = start_time
                        observation['id'] += 'a'
                        observation['nrow'] = start_row + 1
                        observation['intervals'] = None
                        observation['eff_od'] = eff_od
                        self.observations.insert(iobs, observation)
                        break
                iobs += 1
                                
        # Now repopulate the empty interval lists

        for observation in self.observations:
            if observation['intervals'] == None:
                intervals = []
                for interval in self.intervals:
                    if interval['start_time'] < observation['stop_time'] and interval['stop_time'] > observation['start_time']:
                        intervals.append( interval )
                observation['intervals'] = intervals

        # Ensure detector intervals get rebuilt
        try:
            del self._detector_intervals
        except:
            pass


    def exclude_od( self, od ):
        """Remove intervals that overlap with given od from self.observations and self.intervals"""
        self.exclude( od=od )


    def exclude_ring( self, ring ):
        """Remove intervals that overlap with given ESA ring from self.observations and self.intervals"""
        self.exclude( ring=ring )


    def exclude( self, od=None, ring=None ):

        if od != None:
            if ring != None:
                raise Exception('Cannot exclude rings and ODs at the same time')
            key = 'od'
            value = od
            print 'Excluding OD == {}'.format(od)
        elif ring != None:
            key = 'ring'
            value = ring
            print 'Excluding ring == {}'.format(od)
        else:
            raise Exception('Must specify od or ring to exclude')

        # enforce the values to be in a list

        values = []
        if isinstance(value, str):
            values.append( value )
        else:
            try:
                for v in value:
                    values.append( v )
            except:
                values.append( value )

        for v in values:
            if key == 'od' and not isinstance(v, int):
                raise Exception('Excluded ODs must have integer IDs: {}'.format(v))
            if key == 'ring' and not isinstance(v, str):
                raise Exception('Excluded rings must have string IDs: {}'.format(v))

        # self.observations

        iobs = 0
        while iobs < len(self.observations):
            observation = self.observations[iobs]
            iint = 0
            while iint < len(observation['intervals']):
                interval = observation['intervals'][iint]
                if interval[key] in values:
                    del observation['intervals'][iint]
                else:
                    iint += 1

            if len(observation['intervals']) == 0:
                del self.observations[iobs]
            else:
                iobs += 1

        # self.intervals

        iint = 0
        while iint < len(self.intervals):
            interval = self.intervals[iint]
            if interval[key] in values:
                del self.intervals[iint]
            else:
                iint += 1
                    

    @property
    def observations(self):
        """Handle the possibility that the interval list may not be continuous"""
        try:
            return self._observations
        except:
            conn = sqlite3.connect( self.dbfile )
            c = conn.cursor()
            cmd = 'select od, start_time, stop_time from ods where start_time < {} and stop_time > {}'.format(self.stop, self.start)
            query = c.execute( cmd )
            self._observations = []
            for q in query:
                od = q[0]
                obs_intervals = []
                for interval in self.intervals:
                    if interval['od'] == od: obs_intervals.append(interval)
                if len(obs_intervals) > 0:
                    self._observations.append( {
                        'id':'{:04}'.format(od), 'ods':[q[0]],
                        'start_time':q[1], 'stop_time':q[2], 'intervals':obs_intervals
                        } )
            return self._observations


    def get_eff_ods(self, ods):
        """Return a list of EFF ODs that overlap with the specified AHF OD list"""
        freq = self.freq
        if self.freq > 70:
            freq = 100
        if freq not in [30, 44, 70, 100]: raise Exception('EFF information not available for frequency: {}'.format( freq ))
        conn = sqlite3.connect( self.dbfile )
        c = conn.cursor()
        eff_ods = OrderedDict()
        for od in ods:
            cmd = 'select eff_od, start_time, stop_time, nrow from eff_files where freq == {} and start_od <= {} and stop_od >= {}'.format(freq, od, od)
            query = c.execute( cmd )
            for q in query:
                id = q[0]
                if id not in eff_ods:
                    eff_ods[ id ] = {
                    'eff_od':q[0], 'start_time':q[1], 'stop_time':q[2], 'nrow':q[3]
                    }

        return eff_ods


    @property
    def ods(self):
        """Return a simple list of integers defining which AHF ODs are used"""
        try:
            return self._ods
        except:
            last_od = -1
            self._ods = []
            for interval in self.intervals:
                if interval['od'] != last_od:
                    last_od = interval['od']
                    self._ods.append(last_od)
            return self._ods
                    

    @property
    def intervals(self):
        """Return the split modes matching query"""
        try:
            return self._intervals
        except:
            conn = sqlite3.connect( self.dbfile )
            c = conn.cursor()
            cmd = 'select pointID_unique, ESA_ID, ACMS_mode, od, start_time, stop_time from split_ACMS_modes where start_time < {} and stop_time > {} AND ('.format(self.stop, self.start)
            if self.use_SCM: cmd += ' or ACMS_mode == "S"'
            if self.use_HCM: cmd += ' or ACMS_mode == "H"'
            if self.use_OCM: cmd += ' or ACMS_mode == "O"'
            cmd = cmd.replace('( or', '(')
            cmd += ' ) '
            query = c.execute( cmd )
            self._intervals = []
            for q in query:
                self._intervals.append( {'id':q[0].encode('ascii'), 'ring':q[1].encode('ascii'), 'mode':q[2].encode('ascii'), 'od':q[3], 'start_time':q[4], 'stop_time':q[5]} )

            return self._intervals

    @property
    def detector_intervals(self):
        """Return the detector intervals matching query"""
        try:
            return self._detector_intervals
        except:
            if self.freq < 100:
                tab = 'ring_times_lfi{}'.format( self.freq )
            else:
                tab = 'ring_times_hfi'
            conn = sqlite3.connect( self.dbfile )
            c = conn.cursor()
            cmd = 'select id, pointID_unique, start_time, stop_time from {} where start_time < {} and stop_time > {}'.format(tab, self.stop, self.start)
            query = c.execute( cmd )
            self._detector_intervals = []
            for q in query:
                self._detector_intervals.append( {'index':q[0], 'id':q[1].encode('ascii'), 'start_time':q[2], 'stop_time':q[3]} )

            # discard detector intervals without overlap with split intervals

            i = 0
            j = 0
            while i < len(self._detector_intervals):
                while j < len( self.intervals ):
                    interval = self.intervals[j]
                    if interval['stop_time'] < self._detector_intervals[i]['start_time']:
                        j += 1
                    elif interval['start_time'] > self._detector_intervals[i]['stop_time']:
                        overlaps = False
                        break
                    else:
                        overlaps = True
                        break
                if overlaps:
                    i += 1
                else:
                    del self._detector_intervals[i]

            return self._detector_intervals

    @property
    def AHF_files(self):
        try:
            return self._AHF_files
        except:
            self._AHF_files = None
    
    def get_EFF_files():
        pass


if __name__ == '__main__':
    
    print 'Testing RingDB'
    
    db = RingDB( 'db.db', 30, od_range=[98,98], use_SCM=True, use_HCM=True, use_OCM=False )

    print 'Intervals: '
    for interval in db.intervals: print interval

    print 'Observations: '
    for observation in db.observations:
        print observation['id']
        for interval in observation['intervals']: print interval

    print 'ODs: ', db.ods

    db.apply_breaks()

    print 'Intervals: '
    for interval in db.intervals: print interval

    print 'Observations: '
    for observation in db.observations:
        print observation['id']
        for interval in observation['intervals']: print interval

    print 'Done!'
