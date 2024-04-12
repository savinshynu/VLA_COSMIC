from astroquery.jplhorizons import Horizons
from scipy.constants import c
from astropy.time import Time
import matplotlib.pyplot as plt
spacecraft_freq = 8420.432097e+6

observatory_code = "Very Large Array"

query = Horizons(id = 'Voyager 1', location = str(observatory_code),
                 epochs = {'start' : '2023-04-11 09:51:19',
                           'stop'  : '2023-04-11 09:53:00',
                           'step' : '64'},
                 id_type = 'majorbody')

eph = query.ephemerides()
doppler = -eph['delta_rate']*1e3/c*spacecraft_freq

ts = Time(eph['datetime_jd'], format = 'jd').unix
fs = doppler.value.data + spacecraft_freq

plt.plot(ts-ts[0], fs/1e+6)
plt.xlabel("Time [unix seconds]")
plt.ylabel("Expected frequency [MHz]")
plt.show()