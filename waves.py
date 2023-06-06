# Clustering measurements for WAVES target catalogues

import glob
import math
import multiprocessing as mp
import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import default_rng
import pickle
import pylab as plt
import scipy.optimize
import subprocess
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import Corrfunc
# import Corrfunc.mocks
# from Corrfunc.utils import convert_3d_counts_to_cf
import pymangle

import calc_kcor
import util
import wcorr

ln10 = math.log(10)
rng = default_rng()

# Flagship2 cosomology, converting to h=1 units
h = 1
Om0 = 0.319
cosmo = util.CosmoLookup(h, Om0)

north_limits = [157.25, 225.0, -3.95, 3.95]
south_limits = [330, 52.5, -35.6, -27]


def make_rect_mask_N():
    """Make simple rectangular mangle mask."""
    wcorr.make_rect_mask(limits=north_limits, rect_mask='mask_N.ply')


def make_rect_mask_S():
    """Make simple rectangular mangle mask."""
    wcorr.make_rect_mask(limits=south_limits, rect_mask='mask_S.ply')


def wcounts_N():
    """Angular pair counts in mag bins."""
    wcounts(infile='WAVES-N_0p2_Z22_GalsAmbig_CompletePhotoZ.fits',
            mask_file='mask_N.ply', out_pref='wmag_N/', limits=north_limits)

    
def wcounts_S():
    """Angular pair counts in mag bins."""
    wcounts(infile='WAVES-S_small.fits',
            mask_file='mask_S.ply', out_pref='wmag_S/', limits=south_limits)

    
def wcounts(infile, mask_file, out_pref, limits,
            nran=100000, nra=10, ndec=1,
            tmin=0.01, tmax=10, nbins=20,
            magbins=np.linspace(15, 22, 8)):
    """Angular pair counts in mag bins."""

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    ra_col = 'RAcen'
    dec_col = 'Deccen'
    mag_col = 'mag_Zt'
    t = Table.read(infile)
    ra, dec, mag = t[ra_col], t[dec_col], t[mag_col]
    print(len(t), 'galaxies read')

    # Check for RA wraparound
    ra_wrap = 0
    if limits[0] > limits[1]:
        ra_wrap = 1
        limits[:2] = wcorr.ra_shift(limits[:2])
        print('RA limits changed to', limits[:2])
        ra = wcorr.ra_shift(ra)
    
    sel = ((mag < magbins[-1]) *
               (limits[0] <= ra) * (ra < limits[1]) *
               (limits[2] <= dec) * (dec < limits[3]))
        
    t = t[sel]
    print(len(t), 'galaxies selected within limits')
    
    ra, dec, mag = t[ra_col], t[dec_col], t[mag_col]
    if ra_wrap:
         ra = wcorr.ra_shift(ra)
       
    sub = np.zeros(len(ra), dtype='int8')
    print('imag  ngal')
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        sub[sel] = imag
        print(imag, len(t[sel]))
    galcat = wcorr.Cat(ra, dec, sub=sub)
    galcat.assign_jk(limits, nra, ndec)

    mask = pymangle.Mangle(mask_file)
    ra, dec = mask.genrand_range(nran, *limits)
    rancat = wcorr.Cat(ra.astype('float64'), dec.astype('float64'))
    rancat.assign_jk(limits, nra, ndec)

    print(galcat.nobj, rancat.nobj, 'galaxies and randoms')

    njack = nra*ndec
    for ijack in range(njack+1):
        rcoords = rancat.sample(ijack)
        info = {'Jack': ijack, 'Nran': len(rcoords[0]), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_J{ijack}.pkl'
        pool.apply_async(wcorr.wcounts, args=(*rcoords, bins, info, outfile))
        for imag in range(len(magbins) - 1):
            print(ijack, imag)
            mlo, mhi = magbins[imag], magbins[imag+1]
            gcoords = galcat.sample(ijack, sub=imag)
            info = {'Jack': ijack, 'mlo': mlo, 'mhi': mhi,
                    'Ngal': len(gcoords[0]), 'Nran': len(rcoords[0]),
                                'bins': bins, 'tcen': tcen}
            outfile = f'{out_pref}GG_J{ijack}_m{imag}.pkl'
            pool.apply_async(wcorr.wcounts,
                             args=(*gcoords, bins, info, outfile))
            outfile = f'{out_pref}GR_J{ijack}_m{imag}.pkl'
            pool.apply_async(wcorr.wcounts,
                             args=(*gcoords, bins, info,  outfile, *rcoords))
    pool.close()
    pool.join()


def hists(infile='12244.fits', Mr_lims=[-24, -15],
          zbins=np.linspace(0, 1, 6)):
    """Abs mag histograms in redshift bins."""

    t = Table.read(infile)
    z = t['true_redshift_gal']
    plt.clf()
    fig, axes = plt.subplots(5, 1, sharex=True, num=1)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iz in range(len(zbins) - 1):
        sel = (zbins[iz] <= z) * (z < zbins[iz+1])
        M_r = t['abs_mag_r01'][sel]
        print(iz, np.percentile(M_r, (5, 50, 95)))
        ax = axes[iz]
        ax.semilogy()
        ax.hist(M_r, bins=np.linspace(*Mr_lims, 37))
        ax.text(0.05, 0.8, rf'z = {zbins[iz]:3.1f}-{zbins[iz+1]:3.1f}',
            transform=ax.transAxes)
    plt.xlabel(r'$M_r$')
    plt.show()


def w_plot(nmag=7, njack=10, fit_range=[0.01, 1], p0=[0.05, 1.7], prefix='wmag_N/',
           avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7,
           alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
           phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67]):
    """w(theta) from angular pair counts in mag bins."""

    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for imag in range(nmag):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{prefix}RR_J{ijack}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GG_J{ijack}_m{imag}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GR_J{ijack}_m{imag}.pkl'
            (info, DR_counts) = pickle.load(open(infile, 'rb'))
            corrs.append(
                wcorr.Corr1d(info['Ngal'], info['Nran'],
                             DD_counts, DR_counts, RR_counts,
                             mlo=info['mlo'], mhi=info['mhi']))
        corr = corrs[0]
        corr.err = np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        corr.ic_calc(fit_range, p0, 5)
        corr_slices.append(corr)
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax, color=color, label=f"m = [{info['mlo']}, {info['mhi']}]")
        popt, pcov = corr.fit_w(fit_range, p0, ax, color)
        print(popt, pcov)
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()

    wcorr.wplot_scale(cosmo, corr_slices, gamma1=gamma1, gamma2=gamma2,
                      r0=r0, eps=eps, alpha=alpha, Mstar=Mstar,
                      phistar=phistar, kcoeffs=kcoeffs)

