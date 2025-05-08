# Procedures for LSST data

from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pymangle
import wcorr

DATA = os.environ['DATA']

def check_mask(datafile=DATA+'/LSST/ComCam/edfs.fits', maskfile=DATA+'/gaia/edfs.ply'):
    '''Plot results of LSST query and GAIA bright star mask.'''
    t = Table.read(datafile)
    mask = np.loadtxt(maskfile, skiprows=2)
    # pdb.pm()
    ax = plt.subplot(111)
    plt.scatter(t['ra'], t['dec'], s=0.1)
    for i in range(mask.shape[0]):
        circle = plt.Circle((mask[i, 0], mask[i, 1]), mask[i, 2], color='r', fill=False)
        ax.add_patch(circle)
    plt.show()


def wcorr_mag(galfile=DATA+'/LSST/ComCam/edfs.fits', ranfile=DATA+'/gaia/edfs/ran.dat',
              maskfile=DATA+'/gaia/edfs/mask.ply', magbins=np.linspace(18, 23, 6)):
    '''w(theta) in mag bins.'''

    sub_names = [f'm = [{magbins[i]:3.2f}, {magbins[i+1]:3.2f}]' for i in range(len(magbins)-1)]
    t = Table.read(galfile)
    mask = pymangle.Mangle(maskfile)
    sel = mask.contains(t['ra'], t['dec'])
    t = t[sel]
    galcat = wcorr.Cat(t['ra'], t['dec'], sub=np.digitize(t['imag'], magbins)-1,
                       sub_names=sub_names)
    randat = np.loadtxt(ranfile, skiprows=1)
    rancat = wcorr.Cat(randat[:, 0], randat[:, 1])

    wcorr.wcorr_sub(galcat, rancat, tmin=0.01, tmax=10, nbins=20, npatch=9, patch_plot_file='edfs.png')