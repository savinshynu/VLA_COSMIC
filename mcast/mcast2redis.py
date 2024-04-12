"""
Collect metadata form the MCAST and 
send it out to a REDIS channel
"""


import logging
import evla_mcast
import redis
from evla_mcast.flag_server import FlagServer
import time
import sched
import numpy as np

# Set log format to DEBUG  level
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO)



class Coordinator_VLA(evla_mcast.Controller):
    """
    This class utilizes the evla_mcast.Controller (Software for multicasting VLA observation info) which  will receive the metadata of VLA  
    observations and publish it to certain channels using the redis system. 
    """
    def __init__(self, redis_host = 'localhost', redis_port = 6379):
         
       
        #redis port: the redis server to conect to publish metadata in channels
        
        # Inheriting from evla_mcast. Controller
        evla_mcast.Controller.__init__(self)

        # Require two Observation XML documents in order to get
        # the scan stop time as well as basic scan metadata:
        self.scans_require = ['obs','stop']
        
        # Assuming that a redis is server is already active and trying to connect to it.
        try: 
           self.red = redis.Redis(host = 'localhost', port = 6379, decode_responses=True)
           logging.info('Started connection with the Redis server')
        except redis.RedisError:
           logging.error('Connection to Redis server not successful')
           raise RuntimeError('Connection to Redis server not successful')

        
        self.flag = FlagServer() #Getting the live flagging info about the antennas

        self.sch = sched.scheduler(time.time, time.sleep) # Using a scheduler for starting and stopping scans
        
        self.chan_list = ['META1'] #list of channels to publish messages to


    def handle_config(self, scan):
                
        # This function get the scans with a complete metadata from evla_mcast
        logging.info('Handling scans with complete metadata')
        
        self.metadata = self.collect_metadata(scan) # collecting the metadata from the scan
        
        print (scan.datasetId, self.metadata['tstart'], self.metadata['tend'])
        
        #The evla_mcast sends out flagging of antenna information during the observations. A typical VLA scan includes
        # the telescope time on source plus the time telscopes used for slewing. self.flag = FlagServer() will output  a list of
        #flagged antennas during the observations. This requires the start time, stop time and datasetID of the scan. But it would be better to
        # do it many times during the scan as it would provide a better understanding of the flagging.
        # Here I am scheduling to get a list of antennas flagged every second during the observation (maybe there is a better way to do it and it may not be a good
        # idea to run it multiple times within a second during the scan) 
         
        #calculating the time difference between start and stop time in seconds 
        tdiff = (self.metadata['tend'] - self.metadata['tstart'])*24.0*3600.00
        # scheduling the collection of flagging data  at the end of observation and sending out all metadata to some channels.
        self.sch.enter(tdiff, 1, self.get_flagant, argument = (scan.startTime, scan.stopTime, scan.datasetId))
        self.sch.enter(tdiff, 2, self.pub_msg_channel, argument = (self.red, self.chan_list, self.metadata))
        self.sch.run()
   

    def get_flagant(self, tbeg, tend, datasetId):
        """
        Function to get a list of flaggeda antennas during each second 
        of the scan time
        """

        tint = 0.00001157 # 1 sec in mjd day
        #tbeg = scanob.startTime
        #tend = scanob.stopTime 
        print(tbeg,tend)
        trange  = np.arange(tbeg, tend, tint)
        antflag = {}
        for tm in trange:
            antflag[tm] = self.flag.flagged_ants(datasetId, tm, tm+tint)
        logging.info('Collected flagged antennas for the scan')
        self.metadata['ant_flag'] = antflag



    @staticmethod
    def collect_metadata(scan):

        """
        Extract the metadata parameters and add them to a dictionary 'meta'
        """
        meta = {}         #Initialize a dictionary for the metadata
        nant = scan.numAntenna   # No. of antennas
        bw = scan.get_receiver('BD') # bandwidth
        src = scan.source   #source name
        ifids = scan.IFids # IF Ids
        fcents=[]
        for i in ifids:
            fcents.append(scan.get_sslo(i)) # Center frequencies
        ra  = scan.ra_deg     # Right ascension in degrees
        dec = scan.dec_deg   # Declination in degrees
        tstart = scan.startTime # Starting time  of the scan
        tend = scan.stopTime   # Stopping time
        projid = scan.projid   # Project ID of the scan
        station = scan.listOfStations # List of stations
        mjd = str(tstart).split('.')[0] # MJD date
        baseband = scan.baseBandNames
        npol = scan.npol
        nchan = scan.nchan

        meta['nant'] = nant
        meta['bw'] = bw
        meta['src'] = src
        meta['ifids'] = ifids
        meta['fcents'] = fcents
        meta['ra_deg'] = ra
        meta['dec_deg'] = dec
        meta['tstart'] = tstart
        meta['tend'] = tend
        meta['projid'] = projid
        meta['station'] = station
        meta['mjd'] = mjd
        meta['baseband'] = baseband
        meta['npol'] = npol
        meta['nchan'] = nchan
        return meta



    def pub_msg_channel(self, red_server, chan_list, metadata):
         
        """Publish a message to redis channels' 
           (taken from the Meerkat coordinator code)
        Args:
            red_server: Redis server.
            chan_list (str): List of channel to be published to. 
            metadata: consisting of values and keys
        """
        for chan_name in chan_list:
            for key in metadata.keys():
                msg = '{}={}'.format(key, metadata[key])
                red_server.publish(chan_name, msg)
                logging.info('Published {} to channel {}'.format(msg, chan_name))

if __name__ == '__main__':

   c = Coordinator_VLA()
   c.run()
