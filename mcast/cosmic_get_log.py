import logging
import evla_mcast
from evla_mcast.flag_server import FlagServer
import time
import sched
import numpy as np

# Set log format, level
logging.basicConfig(format="%(asctime)-15s %(levelname)8s %(message)s",
        level=logging.INFO)

class SimpleController(evla_mcast.Controller):

    def __init__(self):

        # Call base class init
        super(SimpleController, self).__init__()

        # Require two Observation XML documents in order to get
        # the scan stop time as well as basic scan metadata:
        self.scans_require = ['obs', 'stop']

        #Calling to get flagged antennas at a time
        self.flag = FlagServer()

        #Calling to schedule functions at certain times
        self.sch = sched.scheduler(time.time, time.sleep)

    def handle_config(self, scan):
        
        # This function is called within another function in evla_mcast as it gets an observation with full metadata (It happens at the starting time of that scan).
        # This scan object has all the metadata in it.
        NANT = scan.numAntenna   #Num of antennas
        BW = scan.get_receiver('BD')  # Bandwidth
        SRC = scan.source       # Name of source
        IFids = scan.IFids     # IFs
        FCENTs=[]
        for i in IFids:      #Center Frequencies
            FCENTs.append(scan.get_sslo(i))
        RA  = scan.ra_deg     #RA, Dec, Start, End and Project Id
        DEC = scan.dec_deg
        TSTART = scan.startTime    # MJD units
        TEND = scan.stopTime
        PROJID = scan.projid
        #STATION = scan.listOfStations
        MJD = str(TSTART).split('.')[0]
        #with open("vla_output_MJD"+MJD+".dat", 'a') as f:
        print("PROJID= {0} SRC= {1} (ra,dec)=( {2} , {3} ) deg | MJD= {4} - {5} | NANT= {6} |BW = {7} | N_cent_freq = {8} | FCENT = {9} ".format(PROJID, SRC, RA, DEC, TSTART, TEND, NANT, BW, len(FCENTs), FCENTs))
        #f.close()

        # We can call the list of flagged antennas during the observation or after it is done. Here it is done after the observation. Now we calculate the delay units (in sec) from
        # the start time to do that.

        delay = (TEND - TSTART)*24.0*3600.00
        
        #Scheduling and running the get_flagant function at that delay units
        self.sch.enter(delay, 1, self.get_flagant, argument = (TSTART, TEND, scan.datasetId))
        self.sch.run()



    def get_flagant(self, tbeg, tend, datasetId):
        logging.info('Collecting flagged antennas for the scan')
        tint = 0.00001157 # 1 sec in mjd day
        trange  = np.arange(tbeg, tend, tint)
        antflag = {}
        print("Time in MJD : Flagged antenna list")
        for tm in trange:
            antflag['tm'] = self.flag.flagged_ants(datasetId, tm, tm+tint)
            print("%f : Bad ant = %s" %(tm, antflag['tm'])) 
         

c = SimpleController()
c.run()
