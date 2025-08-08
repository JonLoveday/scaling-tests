# Procedures for LSST data

from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pymangle
import wcorr

DATA = os.environ['DATA']

def check_mask(datafile=DATA+'/LSST/DP1/ecdfs.fits', maskfile=DATA+'/gaia/ecdfs/holes.txt'):
    '''Plot results of LSST query and GAIA bright star mask.'''
    t = Table.read(datafile)
    mask = np.loadtxt(maskfile, skiprows=2)
    # pdb.pm()
    ax = plt.subplot(111)
    plt.scatter(t['coord_ra'], t['coord_dec'], s=0.1)
    for i in range(mask.shape[0]):
        circle = plt.Circle((mask[i, 0], mask[i, 1]), mask[i, 2], color='r', fill=False)
        ax.add_patch(circle)
    plt.show()


def wcorr_mag(galfile=DATA+'/LSST/DP1/ecdfs.fits', ranfile=DATA+'/gaia/ecdfs/ran.dat',
              maskfile=DATA+'/gaia/ecdfs/mask.ply', magbins=np.linspace(18, 25, 8)):
    '''w(theta) in mag bins.'''

    sub_names = [f'm = [{magbins[i]:3.2f}, {magbins[i+1]:3.2f}]' for i in range(len(magbins)-1)]
    t = Table.read(galfile)
    mask = pymangle.Mangle(maskfile)
    sel = mask.contains(t['coord_ra'], t['coord_dec'])
    t = t[sel]
    galcat = wcorr.Cat(t['coord_ra'], t['coord_dec'], sub=np.digitize(t['r_cModelMag'], magbins)-1,
                       sub_names=sub_names)
    randat = np.loadtxt(ranfile, skiprows=1)
    rancat = wcorr.Cat(randat[:, 0], randat[:, 1])

    wcorr.wcorr_sub(galcat, rancat, tmin=0.001, tmax=0.5, nbins=15, npatch=9, patch_plot_file='ecdfs.png')