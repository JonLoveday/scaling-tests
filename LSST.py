# Procedures for LSST data

from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np

def check_mask(datafile='edfs.fits', maskfile='edfs.ply'):
    '''Plot results of LSST query and GAIA bright star mask.'''
    t = Table.read(datafile)
    mask = np.loadtxt(maskfile, skiprows=2)

    ax = plt.subplot(111)
    plt.scatter(t['ra'], t['dec'], s=0.1)
    for i in range(mask.shape[1]):
        circle = plt.Circle((mask[i, 0], mask[i, 1]), mask[i, 2], color='r', fill=False)
        ax.add_patch(circle)
    plt.show()