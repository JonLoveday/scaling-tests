# Euclid angular clustering measurements using treecorr

import glob
import math
import multiprocessing as mp
import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import default_rng
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import scipy.optimize
import subprocess
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy_healpix import HEALPix

import treecorr
import pdb
import pymangle

import calc_kcor
import limber
import st_util
import wcorr

ln10 = math.log(10)
rng = default_rng()

# Flagship2 cosomology, converting to h=1 units
h = 1
Om0 = 0.319
cosmo = st_util.CosmoLookup(h, Om0)

north_limits = [157.25, 225.0, -3.95, 3.95]
q1_south_limits = [55, 67.2, -51.6, -45.2]
q1_north_limits = [262, 277, 63, 69]

solid_angle_north = ((north_limits[1]-north_limits[0]) *
                     (north_limits[3]-north_limits[2]) * (math.pi/180)**2)

def mask_gal_randoms(gal_coords, maskfile, limits=None, ranfac=1):
    """Apply mask to galaxies and generate randoms."""

    with fits.open(maskfile) as hdulist:
        nside = hdulist[1].header['NSIDE']
        order = hdulist[1].header['ORDERING']
    mask = Table.read(maskfile)
    hp = HEALPix(nside=nside, order=order, frame='icrs')
    print('pixel resolution', hp.pixel_resolution, 'pixel area', hp.pixel_area)
    sel = mask['WEIGHT'] > 0
    mask = mask[sel]
    npix = len(mask)
    usepix = mask['PIXEL']

    # Trim galaxies to those lying within mask
    pixels = hp.lonlat_to_healpix(gal_coords.ra, gal_coords.dec)
    sel = np.isin(pixels, usepix)
    gal_coords = gal_coords[sel]
    ngal = len(gal_coords)
    print(ngal, 'galaxies inside mask')

    # Generate randoms.  limits from gal_coords min/max unless otherwise specified
    nran = int(ranfac*ngal)
    if limits is None:
        limits = [np.min(gal_coords.ra), np.max(gal_coords.ra),
                  np.min(gal_coords.dec), np.max(gal_coords.dec)]
    ra = limits[0] + (limits[1] - limits[0])*rng.random(nran)
    sin_dec_lo = np.sin(np.deg2rad(limits[2]))
    sin_dec_hi = np.sin(np.deg2rad(limits[3]))
    dec = np.rad2deg(np.arcsin(rng.random(nran)*(sin_dec_hi - sin_dec_lo) + sin_dec_lo))
    ran_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    pixels = hp.lonlat_to_healpix(ran_coords.ra, ran_coords.dec)
    sel = np.isin(pixels, usepix)
    return gal_coords, ran_coords[sel]


def w_mag_n(cat='/Users/loveday/Data/euclid/q1/1734103664898TIDR-result.fits',
            maskfile='/Users/loveday/Data/euclid/q1/merged_VMPZ-ID/VMPZ-ID_EDFN_HPCOVERAGE_NIR_J.fits',
            out_dir='/Users/loveday/Data/euclid/q1/wcorr_north',
            spurious_limit=0.1, point_prob_limit=0.1, ranfac=1, limits=q1_north_limits,
            magbins=[14, 16, 18, 20, 22, 24]):
    w_mag(maskfile=maskfile, out_dir=out_dir, limits=limits)

def w_mag(cat='/Users/loveday/Data/euclid/q1/1734103664898TIDR-result.fits',
          maskfile='/Users/loveday/Data/euclid/q1/merged_VMPZ-ID/VMPZ-ID_EDFS_HPCOVERAGE_NIR_J.fits',
          out_dir='/Users/loveday/Data/euclid/q1/wcorr',
          spurious_limit=0.1, point_prob_limit=0.1, ranfac=1, limits=q1_south_limits,
          magbins=[14, 16, 18, 20, 22, 24]):
    """Angular clustering in mag bins."""

    # Read mask
    with fits.open(maskfile) as hdulist:
        nside = hdulist[1].header['NSIDE']
        order = hdulist[1].header['ORDERING']
    mask = Table.read(maskfile)
    hp = HEALPix(nside=nside, order=order, frame='icrs')
    print('pixel resolution', hp.pixel_resolution, 'pixel area', hp.pixel_area)
    sel = mask['WEIGHT'] > 0
    mask = mask[sel]
    npix = len(mask)
    usepix = mask['PIXEL']

    # Select galaxies lying mask pixels with non-zero weight
    t = Table.read(cat)
    print(len(t), 'objects in', cat)
    sel = ((limits[0] <= t['RIGHT_ASCENSION']) * (limits[1] > t['RIGHT_ASCENSION']) *
           (limits[2] <= t['DECLINATION']) * (limits[3] > t['DECLINATION']) *
           (t['SPURIOUS_PROB'] < spurious_limit) * (t['POINT_LIKE_PROB'] < point_prob_limit))
    t = t[sel]
    print(len(t), 'galaxies selected')

    gal_coords = SkyCoord(t['RIGHT_ASCENSION'], t['DECLINATION'], unit='deg', frame='icrs')
    pixels = hp.lonlat_to_healpix(gal_coords.ra, gal_coords.dec)
    sel = np.isin(pixels, usepix)
    t = t[sel]
    gal_coords = SkyCoord(t['RIGHT_ASCENSION'], t['DECLINATION'], unit='deg', frame='icrs')
    ngal = len(gal_coords)
    print(ngal, 'galaxies inside mask')

    # Generate randoms.  limits from gal_coords min/max unless otherwise specified
    nran = int(ranfac*ngal)
    if limits is None:
        limits = [np.min(gal_coords.ra), np.max(gal_coords.ra),
                  np.min(gal_coords.dec), np.max(gal_coords.dec)]
    ra = limits[0] + (limits[1] - limits[0])*rng.random(nran)
    sin_dec_lo = np.sin(np.deg2rad(limits[2]))
    sin_dec_hi = np.sin(np.deg2rad(limits[3]))
    dec = np.rad2deg(np.arcsin(rng.random(nran)*(sin_dec_hi - sin_dec_lo) + sin_dec_lo))
    ran_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    pixels = hp.lonlat_to_healpix(ran_coords.ra, ran_coords.dec)
    sel = np.isin(pixels, usepix)
    ran_coords = ran_coords[sel]

    plt.scatter(gal_coords.ra, gal_coords.dec, s=0.1)
    plt.scatter(ran_coords.ra, ran_coords.dec, s=0.1, c='r')
    plt.show()

    subsets = []
    for imag in range(len(magbins)-1):
        mlo, mhi = magbins[imag], magbins[imag+1]
        sel = (mlo <= t['J_mag']) * (mhi > t['J_mag'])
        subsets.append({'label': f'm_{mlo}_{mhi}', 'sel': sel})
    wcounts(gal_coords, ran_coords, out_dir, subsets=subsets)


def wcounts(gal_coords, ran_coords, out_dir,
            npatch=9, tmin=0.01, tmax=1, nbins=20,
            subsets=None, plot=0):
    """Angular pair counts (in subsets if specified)."""

    def patch_plot(cat, ax, label, ra_shift=False, nmax=100000):
        # if nmax < cat.ntot:
        #     sel = rng.choice(len(cat), size=nmax, replace=False)
        #     cat = cat[sel]
        ras = cat.ra
        if ra_shift:
            ras = np.array(cat.ra) - math.pi
            neg = ras < 0
            ras[neg] += 2*math.pi
        ax.scatter(ras, cat.dec, c=cat.patch, s=0.1)
        ax.text(0.05, 0.05, label, transform=ax.transAxes)
       
    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # process the randoms
    rcat = treecorr.Catalog(ra=ran_coords.ra, dec=ran_coords.dec,
                            ra_units='deg', dec_units='deg',
                            npatch=npatch)
    rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees')
    rr.process(rcat)

    # Now the galaxies
    gcat = treecorr.Catalog(ra=gal_coords.ra, dec=gal_coords.dec,
                            ra_units='deg', dec_units='deg',
                            patch_centers=rcat.patch_centers)
    if plot:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=1)
        fig.set_size_inches(8, 4)
        fig.subplots_adjust(hspace=0, wspace=0)
        patch_plot(gcat, axes[0], 'gal', ra_shift=ra_shift)
        patch_plot(rcat, axes[1], 'ran', ra_shift=ra_shift)
        plt.show()

    dr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees')
    dr.process(gcat, rcat)
    dd = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees', var_method='jackknife')
    dd.process(gcat)
    dd.calculateXi(rr=rr, dr=dr)
    xi_jack, w = dd.build_cov_design_matrix('jackknife')
    outfile = f'{out_dir}/w.fits'
    dd.write(outfile, rr=rr, dr=dr)
    with fits.open(outfile, mode='update') as hdul:
        hdul.append(fits.PrimaryHDU(xi_jack))
        hdul.flush()

    if subsets:
        for subset in subsets:
            label = subset['label']
            sel = subset['sel']
            gcat = treecorr.Catalog(ra=gal_coords.ra[sel], dec=gal_coords.dec[sel],
                                    ra_units='deg', dec_units='deg',
                                    patch_centers=rcat.patch_centers)

            dr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                        sep_units='degrees')
            dr.process(gcat, rcat)
            dd = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                        sep_units='degrees', var_method='jackknife')
            dd.process(gcat)
            dd.calculateXi(rr=rr, dr=dr)
            xi_jack, w = dd.build_cov_design_matrix('jackknife')
            outfile = f'{out_dir}/w_{label}.fits'
            dd.write(outfile, rr=rr, dr=dr)
            with fits.open(outfile, mode='update') as hdul:
                hdr = hdul[1].header
                hdul.append(fits.PrimaryHDU(xi_jack))
                hdul.flush()


def wplot_mag(magbins=[14, 16, 18, 20, 22, 24], fit_range=[0.001, 1], p0=[0.05, 1.7],
    indir='/Users/loveday/Data/euclid/q1/wcorr/'):
    ax = plt.subplot(111)
    for imag in range(len(magbins)-1):
        mlo, mhi = magbins[imag], magbins[imag+1]
        infile = indir + f'w_m_{mlo}_{mhi}.fits'
        corr = wcorr.Corr1d(infile=infile)
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax, color=color, label=f"m = [{mlo}, {mhi}]")
        if fit_range:
            popt, pcov = corr.fit_w(fit_range, p0, ax, color)
            print(popt, pcov)

    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()
