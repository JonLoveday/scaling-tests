from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np
import os

def gaia_mask(infile='edfs.fits', outfile='edfs.ply'):
    '''Create mangle mask from GAIA bright stars.
    Exclusion radius is r [arcmin] = 10^(1.6-0.15 g); g < 16.'''

    stars = Table.read(infile)
    rad = 10**(1.6 - 0.15*stars['phot_g_mean_mag'])/60.0
    with open(outfile, 'w') as fout:
        print('circle 0 1', file=fout)
        print('unit d', file=fout)
        for i in range(len(stars)):
            print(stars['ra'][i], stars['dec'][i], rad[i], file=fout)


def plotran(infile='ran.dat'):
    '''Plot ransack generated randoms.'''
    ran = np.loadtxt(infile, skiprows=1)
    plt.scatter(ran[:, 0], ran[:, 1], s=0.1)
    plt.show()