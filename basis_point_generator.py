import numpy as np

atto_width = np.linspace(0.15, 1., 10)
pulse_range = np.linspace(38, 44, 10)
zero_time = np.linspace(-5, 5, 100)  # bumped from the time boundaries
chirp_range = np.linspace(-5, 5, 10)

a, b, cc, d = np.meshgrid(atto_width, pulse_range, zero_time, chirp_range)  # c shadowed c so changed to cc
attoatto = a.flatten()
pulsepulse = b.flatten()
zerozero = cc.flatten()
chirpchirp = d.flatten()
indexer = np.arange(start=0, stop=chirpchirp.shape[0], step=1)
# Stack basis points and select only the number of basis_points from the list.
pulse_space = np.stack([indexer, attoatto, pulsepulse, zerozero, chirpchirp], axis=1)[7100]
print(pulse_space)
