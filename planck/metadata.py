import glob
import sqlite3
import exceptions
import logging as l

from Planck import parse_channels, freq2inst

try:
    import private
except exceptions.ImportError:
    print('private.py is needed to use the planck module, this is available only to members of the Planck collaboration')
    raise

class DataSelector(object):
    """Planck data selector
    channels can be integer frequency, list of channel names (same frequency) or a single channel name string
    efftype is R for reduced, C for converted [uncal with dip]"""

    def __init__(self, channels=None, efftype='R'):
        self.config = {}
        self.config['database'] = private.database
        self.config['exchangefolder'] = private.exchangefolder
        self.channels = parse_channels(channels)
        self.f = self.channels[0].f
        self.efftype = efftype

    @property
    def ods(self):
        try:
            return self._ods
        except exceptions.AttributeError():
            self._ods = set()
            for obt_range in self.obt_ranges:
                self._ods.update(self.get_ODs(obt_range))
            self._ods = sorted(list(self._ods))
            return self.ods

    def by_od_range(self, od_range):
        """ods is a list of start and stop OD"""
        self.od_range = od_range
        self._ods = range(od_range[0], od_range[1])
        self.obt_ranges = [self.get_obt_range_from_od_range()]

    def by_obt(self, obt_ranges):
        """obt_ranges is a list of 2 element lists (or tuple) with start-stop obt"""
        self._obt_ranges = obt_ranges

    def by_rings(self, rings):
        """rings is a list of ring numbers"""
        #FIXME LFI or HFI?
        raise exceptions.NotImplementedError()

    def get_one_AHF(self, obt_range):
        conn = sqlite3.connect(self.config['database'])
        c = conn.cursor()
        values = (obt_range[0]*2**16, obt_range[-1]*2**16)
        query = c.execute('select file_path from ahf_files where endOBT>=? and startOBT<=?', values)
        files = [q[0] for q in query]
        c.close()
        return files

    def get_AHF(self):
        return [self.get_one_AHF(obt_range) for obt_range in self.obt_ranges]

    def get_EFF(self):
        return self.latest_exchange(self.f.freq, self.ods, None, self.efftype)

    def get_obt_range_from_ods(self, freq=None, ods=None):
        if ods is None:
            ods = self.ods
        if freq is None:
            freq = self.f.freq
        conn = sqlite3.connect(self.config['database'])
        c = conn.cursor()
        query = c.execute('select startOBT, endOBT from efdd_od_ranges where freq=? and od in ? order by od ASC', (freq, ods))
        c.close()
        obt_ranges = [[q[0]/2.**16, q[1]/2.**16] for q in query]
        return obt_ranges

    def get_obt_range_from_od_range(self, freq=None, od_range=None):
        if od_range is None:
            od_range = self.od_range
        if freq is None:
            freq = self.f.freq
        conn = sqlite3.connect(self.config['database'])
        c = conn.cursor()
        obt_range = []
        for obt_tag, od in zip(['startOBT','endOBT'], od_range):
            query = c.execute('select %s from efdd_od_ranges where freq=? and od==?' % obt_tag, (freq, od))
            obt_range.append(query.fetchone()[0]/2.**16)
        c.close()
        return obt_range

    def get_ODs(self, obtrange):
        """List of ODs within the obt range provided"""
        assert obtrange[1]>=obtrange[0]
        conn = sqlite3.connect(self.config['database'])
        c = conn.cursor()
        query = c.execute('select od from efdd_od_ranges where freq=? and endOBT>=? and startOBT<=?',(freq, obtrange[0]*2**16, obtrange[1]*2**16) )
        ods = [q[0] for q in query]
        c.close()
        return ods

    def latest_exchange(self, freq, ods, exchangefolder = None, type = 'R'):
        """Returns the latest version of an exchange format file"""
        single = False
        if isinstance(ods, int):
            ods = [ods]
            single = True
        if exchangefolder is None:
            exchangefolder = self.config['exchangefolder'][freq2inst(freq)]
        if glob.glob(exchangefolder + '/*.fits'):
            ods = [0]
        if type == 'K':
            type = ''
        EFF = []
        for od in ods:
            pattern = '/*%03d-%04d-%s*.fits' % (freq, od, type)
            if od:
                pattern = ('/%04d' % od) + pattern
            l.debug('Exchange format: %s' % (exchangefolder + pattern))
            allversions = glob.glob(exchangefolder + pattern)
            if not allversions:
                error_message = 'Cannot find file from pattern: %s' % (exchangefolder +         pattern)
                l.error(error_message)
                raise exceptions.IOError(error_message)
            EFF.append(max(allversions, key = lambda x: int(x[-13:-5])))
        if single:
            EFF = EFF[0]
        return EFF

if __name__ == '__main__':
    ds = DataSelector(channels=30)
    ds.by_od_range([100, 120])
    ds.get_EFF()
    ds.get_AHF()
