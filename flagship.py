# Clustering/LF measurements for Euclid flagship

import glob
# from gplearn.genetic import SymbolicRegressor
import math
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import numpy.ma as ma
from numpy.polynomial import Polynomial
from numpy.random import default_rng
from pathlib import Path
import pdb
import pickle
import pylab as plt
from scipy import interpolate
from scipy.interpolate import interp1d, interpn, LinearNDInterpolator, NearestNDInterpolator, SmoothBivariateSpline, bisplev, bisplrep
import scipy.optimize
import subprocess
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import Corrfunc
# import Corrfunc.mocks
# from Corrfunc.utils import convert_3d_counts_to_cf
import pymangle
import treecorr

import cluster_z
import limber
import util
import wcorr

ln10 = math.log(10)
rng = default_rng()
np.seterr(all='warn')

# Flagship2 cosomology, converting to h=1 units
h = 1
Om0 = 0.319
cosmo = util.CosmoLookup(h, Om0, zbins=np.linspace(0.0001, 2, 200))
solid_angle = 400 * (math.pi/180)**2

band = 'z'
if band == 'h':
    # Global H-band k-correction polynomial fit over [0, 2] from function kcorr
    kcoeffs = [0, -0.642, 0.229]
    magbins = np.linspace(15, 20, 6)
    Mbins = np.linspace(-26, -16, 6)
    Mbins_lf = np.linspace(-26, -16, 41)
    mlim = 25
    infile = '14516.fits'
if band == 'r':
    # Global r-band k-correction polynomial fit over [0, 2] from function kcorr
    kcoeffs = [0, 0.4756, 0.734]
    magbins = np.linspace(17, 22, 6)
    Mbins = np.linspace(-24, -14, 6)
    Mbins_lf = np.linspace(-24, -14, 41)
    mlim = magbins[-1]
    infile = '15189.fits'
if band == 'z':
    # Global z-band k-correction polynomial fit over [0, 2] from function kcorr
    kcoeffs = [0, 0.236, 0.267]
    magbins = np.linspace(16, 22, 7)
    Mbins = np.linspace(-24, -14, 6)
    Mbins_lf = np.linspace(-24, -14, 41)
    mlim = magbins[-1]
    infile = 'fs2_1_z_22.fits'
kpoly = Polynomial(kcoeffs, domain=[0, 2], window=[0, 2])
w_prefix = f'w_mag_{band}/'
Nz_file = f'Nz_{band}.pkl'
lf_file = f'lf_{band}.pkl'
xi_file = f'xi_{band}.pkl'

def stats():
    """Absolute mag and redshift stats for apparent mag bins."""

    t = Table.read(infile)
    mag = t[band+'mag']
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        print(imag, np.percentile(t[band+'abs'].value[sel], (5, 50, 95)),
              np.percentile(t['true_redshift_gal'].value[sel], (5, 50, 95)))


def wcounts(mask_file='mask.ply',
            limits=(180, 200, 0, 20), nran=100000, nra=4, ndec=4,
            tmin=0.01, tmax=10, nbins=20, plots=1, mag_offset=0):
    """Angular pair counts in mag bins."""

    out_pref = f'w_mag_{band}_dm{mag_offset}/'
    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    t = Table.read(infile)
    sel = t[band+'mag'] + mag_offset < magbins[-1]
    t = t[sel]
    ra, dec, mag = t['ra_gal'].astype('float64'), t['dec_gal'].astype('float64'), t[band+'mag'] + mag_offset
    mmean = np.zeros(len(magbins)-1)
    sub = np.zeros(len(ra), dtype='int8')
    if plots:
        plt.clf()
        fig, axes = plt.subplots(5, 1, sharex=True, num=1)
        fig.set_size_inches(5, 6)
        fig.subplots_adjust(hspace=0, wspace=0)
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        sub[sel] = imag
        mmean[imag] = np.mean(mag[sel])
        print(imag, mmean[imag], np.percentile(t[band+'abs'][sel], (5, 50, 95)))
        if plots:
            ax = axes[imag]
            ax.hist(t[band+'abs'][sel], bins=np.linspace(-24, -15, 19))
            ax.text(0.7, 0.8, rf'm = {magbins[imag]:3.1f}-{magbins[imag+1]:3.1f}',
                    transform=ax.transAxes)

    if plots:
        plt.xlabel(rf'$M_{band}$')
        plt.show()
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
        plt.clf()
        plt.scatter(rcoords[0], rcoords[1], s=0.1)
        plt.show()
        info = {'Jack': ijack, 'Nran': len(rcoords[0]), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_J{ijack}.pkl'
        pool.apply_async(wcorr.wcounts, args=(*rcoords, bins, info, outfile))
        for imag in range(len(magbins) - 1):
            print(ijack, imag)
            mlo, mhi = magbins[imag], magbins[imag+1]
            gcoords = galcat.sample(ijack, sub=imag)
            info = {'Jack': ijack, 'mlo': mlo, 'mhi': mhi, 'mmean': mmean[imag],
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


def xir_counts_corrfunc(mask_file='mask.ply', 
               zbins=np.linspace(0, 2, 11), limits=(180, 200, 0, 20),
               ranfac=1, nra=3, ndec=3, rbins=np.logspace(-1, 2, 16),
               randist='shuffle', out_dir=f'xir_M_z_{band}', multi=True):
    """Real-space pair counts in magnitude and redshift bins using corrfunc."""

    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nM = len(Mbins) - 1
    nz = len(zbins) - 1
    rcen = 10**(0.5*np.diff(np.log10(rbins)) + np.log10(rbins[:-1]))
    if multi:
        ncpu = mp.cpu_count()
        pool = mp.Pool(ncpu)
        print('Using', ncpu, 'CPUs')
    
    t = Table.read(infile)
    sel = t[band+'mag'] < mlim
    t = t[sel]
    ra, dec, Mag = t['ra_gal'], t['dec_gal'], t[band+'abs']
    redshift = t['true_redshift_gal']
    r = cosmo.dc(redshift)

    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        for im in range(nM):
            Mlo, Mhi = Mbins[im], Mbins[im+1]
            sel = ((zlo <= redshift) * (redshift < zhi) *
                   (Mlo <= Mag) * (Mag < Mhi))
            Mmean = np.mean(Mag[sel])
            zmean = np.mean(redshift[sel])
            galcat = wcorr.Cat(ra[sel], dec[sel], r=r[sel])
            galcat.assign_jk(limits, nra, ndec)
            galcat.gen_cart()
            ngal = len(ra[sel])
            if ngal > 0:
                nran = int(ranfac*ngal)
                mask = pymangle.Mangle(mask_file)
                rar, decr = mask.genrand_range(nran, *limits)
                rr = rng.choice(r[sel], nran, replace=True)
                rancat = wcorr.Cat(rar.astype('float64'), decr.astype('float64'), r=rr)
                rancat.assign_jk(limits, nra, ndec)
                rancat.gen_cart()

                print(f'Redshift bin {iz}, Mag bin {im} ngal = {ngal}, nran = {nran}')
                for jack in range(galcat.njack+1):
                    x, y, z = galcat.sample(jack=jack, cart=True)
                    xr, yr, zr = rancat.sample(jack=jack, cart=True)
                    info = {'jack': jack, 'Ngal': ngal, 'Nran': nran,
                            'rbins': rbins, 'rcen': rcen,
                            'Mlo': Mlo, 'Mhi': Mhi, 'zlo': zlo, 'zhi': zhi,
                            'Mmean': Mmean, 'zmean': zmean}
                    outgg = f'{out_dir}/GG_j{jack}_z{iz}_M{im}.pkl'
                    outgr = f'{out_dir}/GR_j{jack}_z{iz}_M{im}.pkl'
                    outrr = f'{out_dir}/RR_j{jack}_z{iz}_M{im}.pkl'
                    if multi:
                        pool.apply_async(wcorr.xir_counts,
                                         args=(x, y, z, rbins, info, outgg))
                        pool.apply_async(wcorr.xir_counts,
                                         args=(xr, yr, zr, rbins, info, outrr))
                        pool.apply_async(wcorr.xir_counts,
                                         args=(x, y, z, rbins, info, outgr, xr, yr, zr))
                    else:
                        wcorr.xir_counts(x, y, z, rbins, info, outgg)
                        wcorr.xir_counts(xr, yr, zr, rbins, info, outrr)
                        wcorr.xir_counts(x, y, z, rbins, info, outgr, xr, yr, zr)
    if multi:
        pool.close()
        pool.join()


def xir_M_z(mask_file='mask.ply', 
            zbins=np.linspace(0, 2, 11), limits=(180, 200, 0, 20),
            ranfac=1, npatch=9, rmin=0.1, rmax=100, nbins=16,
            out_dir=f'xir_M_z_{band}'):
    """Real-space pair counts in magnitude and redshift bins using treecorr."""

    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nM = len(Mbins) - 1
    nz = len(zbins) - 1
   
    t = Table.read(infile)
    sel = t[band+'mag'] < mlim
    t = t[sel]
    ra, dec, Mag = t['ra_gal'], t['dec_gal'], t[band+'abs']
    redshift = t['true_redshift_gal']
    r = cosmo.dc(redshift)

    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        for im in range(nM):
            Mlo, Mhi = Mbins[im], Mbins[im+1]
            sel = ((zlo <= redshift) * (redshift < zhi) *
                   (Mlo <= Mag) * (Mag < Mhi))
            Mmean = np.mean(Mag[sel])
            zmean = np.mean(redshift[sel])
            ngal = len(ra[sel])
            if ngal > 0:
                nran = int(ranfac*ngal)
                mask = pymangle.Mangle(mask_file)
                rar, decr = mask.genrand_range(nran, *limits)
                rr = rng.choice(r[sel], nran, replace=True)
                rancat = treecorr.Catalog(
                    ra=rar.astype('float64'), dec=decr.astype('float64'), r=rr,
                    ra_units='deg', dec_units='deg', npatch=npatch)
                galcat = treecorr.Catalog(
                    ra=ra[sel], dec=dec[sel], r=r[sel],
                    ra_units='deg', dec_units='deg',
                    patch_centers=rancat.patch_centers)

                print(f'Redshift bin {iz}, Mag bin {im} ngal = {ngal}, nran = {nran}')
                dd = treecorr.NNCorrelation(
                    min_sep=rmin, max_sep=rmax, nbins=nbins,
                    var_method='jackknife')
                dd.process(galcat)
                dr = treecorr.NNCorrelation(
                    min_sep=rmin, max_sep=rmax, nbins=nbins)
                dr.process(galcat, rancat)
                rr = treecorr.NNCorrelation(
                    min_sep=rmin, max_sep=rmax, nbins=nbins)
                rr.process(rancat)
                dd.calculateXi(rr=rr, dr=dr)
                xi_jack, w = dd.build_cov_design_matrix('jackknife')
                outfile = f'{out_dir}/xir_iz{iz}_im{im}.fits'
                dd.write(outfile, rr=rr, dr=dr)
                with fits.open(outfile, mode='update') as hdul:
                    hdr = hdul[1].header
                    hdr['Mlo'] = Mlo
                    hdr['Mhi'] = Mhi
                    hdr['Mmean'] = Mmean
                    hdr['zlo'] = zlo
                    hdr['zhi'] = zhi
                    hdr['zmean'] = zmean
                    hdr['Ngal'] = galcat.nobj
                    hdr['Nran'] = rancat.nobj
                    hdul.append(fits.PrimaryHDU(xi_jack))
                    hdul.flush()


def xir_app_mag(mask_file='mask.ply', limits=(180, 200, 0, 20),
               ranfac=1, npatch=9, rmin=0.1, rmax=100, nbins=16,
               out_dir=f'xir_mag_{band}'):
    """Real-space pair counts in apparent magnitude bins using treecorr."""

    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nm = len(magbins) - 1
   
    t = Table.read(infile)
    sel = t[band+'mag'] < magbins[-1]
    t = t[sel]
    ra, dec, mag = t['ra_gal'], t['dec_gal'], t[band+'mag']
    redshift = t['true_redshift_gal']
    r = cosmo.dc(redshift)
    mask = pymangle.Mangle(mask_file)

    for im in range(nm):
        mlo, mhi = magbins[im], magbins[im+1]
        sel = (mlo <= mag) * (mag < mhi)
        mmean = np.mean(mag[sel])
        Mmean = np.mean(t[band+'abs'][sel])
        zmean = np.mean(redshift[sel])
        ngal = len(ra[sel])
        nran = int(ranfac*ngal)
        rar, decr = mask.genrand_range(nran, *limits)
        rr = rng.choice(r[sel], nran, replace=True)
        rancat = treecorr.Catalog(
            ra=rar.astype('float64'), dec=decr.astype('float64'), r=rr,
            ra_units='deg', dec_units='deg', npatch=npatch)
        galcat = treecorr.Catalog(
            ra=ra[sel], dec=dec[sel], r=r[sel],
            ra_units='deg', dec_units='deg',
            patch_centers=rancat.patch_centers)

        print(f'mag bin {im} ngal = {ngal}, nran = {nran}')
        dd = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins,
            var_method='jackknife')
        dd.process(galcat)
        dr = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins)
        dr.process(galcat, rancat)
        rr = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins)
        rr.process(rancat)
        dd.calculateXi(rr=rr, dr=dr)
        xi_jack, w = dd.build_cov_design_matrix('jackknife')
        outfile = f'{out_dir}/xir_im{im}.fits'
        dd.write(outfile, rr=rr, dr=dr)
        with fits.open(outfile, mode='update') as hdul:
            hdr = hdul[1].header
            hdr['Mlo'] = mlo
            hdr['Mhi'] = mhi
            hdr['M_app_mean'] = mmean
            hdr['M_abs_mean'] = Mmean
            hdr['zmean'] = zmean
            hdr['Ngal'] = galcat.nobj
            hdr['Nran'] = rancat.nobj
            hdul.append(fits.PrimaryHDU(xi_jack))
            hdul.flush()


def xir_app_mag_z(mask_file='mask.ply', zbins=np.linspace(0, 2, 11), 
                  limits=(180, 200, 0, 20), nmin=1000,
                  ranfac=1, npatch=9, rmin=0.1, rmax=100, nbins=16,
                  out_dir=f'xir_app_mag_z_{band}'):
    """Real-space pair counts in apparent magnitude and redshift bins using treecorr."""

    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nm = len(magbins) - 1
    nz = len(zbins) - 1
  
    t = Table.read(infile)
    sel = ((t[band+'mag'] < magbins[-1]) * (zbins[0] <= t['true_redshift_gal']) *
           (t['true_redshift_gal'] < zbins[-1]))
    t = t[sel]
    ra, dec, mag = t['ra_gal'], t['dec_gal'], t[band+'mag']
    redshift = t['true_redshift_gal']
    r = cosmo.dc(redshift)
    mask = pymangle.Mangle(mask_file)

    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        sel = (zlo <= redshift) * (redshift < zhi)
        mmean = np.mean(mag[sel])
        Mmean = np.mean(t[band+'abs'][sel])
        zmean = np.mean(redshift[sel])
        ngal = len(ra[sel])
        nran = int(ranfac*ngal)
        print(f'z bin {iz} ngal = {ngal}, nran = {nran}')
        rar, decr = mask.genrand_range(nran, *limits)
        rr = rng.choice(r[sel], nran, replace=True)
        rancat = treecorr.Catalog(
            ra=rar.astype('float64'), dec=decr.astype('float64'), r=rr,
            ra_units='deg', dec_units='deg', npatch=npatch)
        galcat = treecorr.Catalog(
            ra=ra[sel], dec=dec[sel], r=r[sel],
            ra_units='deg', dec_units='deg',
            patch_centers=rancat.patch_centers)

        dd = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins,
            var_method='jackknife')
        dd.process(galcat)
        dr = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins)
        dr.process(galcat, rancat)
        rr = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins)
        rr.process(rancat)
        dd.calculateXi(rr=rr, dr=dr)
        xi_jack, w = dd.build_cov_design_matrix('jackknife')
        outfile = f'{out_dir}/xir_z{iz}.fits'
        dd.write(outfile, rr=rr, dr=dr)
        with fits.open(outfile, mode='update') as hdul:
            hdr = hdul[1].header
            hdr['zlo'] = zlo
            hdr['zhi'] = zhi
            hdr['M_app_mean'] = mmean
            hdr['M_abs_mean'] = Mmean
            hdr['zmean'] = zmean
            hdr['Ngal'] = galcat.nobj
            hdr['Nran'] = rancat.nobj
            hdul.append(fits.PrimaryHDU(xi_jack))
            hdul.flush()

        for im in range(nm):
            mlo, mhi = magbins[im], magbins[im+1]
            sel = (zlo <= redshift) * (redshift < zhi) * (mlo <= mag) * (mag < mhi)
            mmean = np.mean(mag[sel])
            Mmean = np.mean(t[band+'abs'][sel])
            zmean = np.mean(redshift[sel])
            ngal = len(ra[sel])
            nran = int(ranfac*ngal)
            print(f'z bin {iz}, mag bin {im} ngal = {ngal}, nran = {nran}')
            if ngal > nmin:
                rar, decr = mask.genrand_range(nran, *limits)
                rr = rng.choice(r[sel], nran, replace=True)
                rancat = treecorr.Catalog(
                    ra=rar.astype('float64'), dec=decr.astype('float64'), r=rr,
                    ra_units='deg', dec_units='deg', npatch=npatch)
                galcat = treecorr.Catalog(
                    ra=ra[sel], dec=dec[sel], r=r[sel],
                    ra_units='deg', dec_units='deg',
                    patch_centers=rancat.patch_centers)

                dd = treecorr.NNCorrelation(
                    min_sep=rmin, max_sep=rmax, nbins=nbins,
                    var_method='jackknife')
                dd.process(galcat)
                dr = treecorr.NNCorrelation(
                    min_sep=rmin, max_sep=rmax, nbins=nbins)
                dr.process(galcat, rancat)
                rr = treecorr.NNCorrelation(
                    min_sep=rmin, max_sep=rmax, nbins=nbins)
                rr.process(rancat)
                dd.calculateXi(rr=rr, dr=dr)
                xi_jack, w = dd.build_cov_design_matrix('jackknife')
                outfile = f'{out_dir}/xir_iz{iz}_im{im}.fits'
                dd.write(outfile, rr=rr, dr=dr)
                with fits.open(outfile, mode='update') as hdul:
                    hdr = hdul[1].header
                    hdr['zlo'] = zlo
                    hdr['zhi'] = zhi
                    hdr['mlo'] = mlo
                    hdr['mhi'] = mhi
                    hdr['M_app_mean'] = mmean
                    hdr['M_abs_mean'] = Mmean
                    hdr['zmean'] = zmean
                    hdr['Ngal'] = galcat.nobj
                    hdr['Nran'] = rancat.nobj
                    hdul.append(fits.PrimaryHDU(xi_jack))
                    hdul.flush()


def cz_test(phot_gal=infile, spec_gal='spec_gal.fits',
            phot_ran='phot_ran.fits', spec_ran='spec_ran.fits',
            mask_file='mask.ply', ra_col='ra_gal', dec_col='dec_gal', 
            z_col='true_redshift_gal', zbins=np.linspace(0, 2, 11),
            magbins=np.linspace(16, 22, 7),
            limits=(180, 200, 0, 20),
            ranfac=1, npatch=9, rmin=0.1, rmax=100, nbins=16,
            out_dir=f'cz_test', spec_frac=0.01):
    """Test cluster-z using spec_frac of flagship as reference spectroscopic sample."""

    def mag_fn(t):
        """Flagship magnitudes."""
        return t['zmag']

    t = Table.read(infile)
    redshift = t[z_col]
    nphot = len(redshift)
    nspec = int(spec_frac*nphot)
    sel = rng.choice(nphot, nspec, replace=False)
    tspec = t[sel]
    tspec.write(spec_gal, overwrite=True)

    nran = int(ranfac*nphot)
    mask = pymangle.Mangle(mask_file)
    rar, decr = mask.genrand_range(nran, *limits)
    tran = Table((rar.astype(np.float64), decr.astype(np.float64)), 
                 names=(ra_col, dec_col))
    tran.write(phot_ran, overwrite=True)

    nran = int(ranfac*nspec)
    rar, decr = mask.genrand_range(nran, *limits)
    zr = rng.choice(redshift, nran, replace=True)
    tran = Table((rar.astype(np.float64), decr.astype(np.float64), zr),
                 names=(ra_col, dec_col, z_col))
    tran.write(spec_ran, overwrite=True)

    cluster_z.pair_counts(
            spec_gal, spec_ran, phot_gal, phot_ran, out_dir, mag_fn=mag_fn,
            ra_col=ra_col, dec_col=dec_col, z_col=z_col,
            zbins=zbins, magbins=magbins, npatch=9,
            exclude_psf=False, identify_overlap=False)


def Nz_comp(infile=infile, Nz_file='cz_test/Nz.pkl', z_col='true_redshift_gal',
            zbins=np.linspace(0, 2, 41)):
    """Compare flagship N(z) in mag bins with cluster_z prediction."""

    (zmean, pmz, pmz_err, mlo, mhi, be_pars, rmin, rmax, weight, fitbin) = pickle.load(open(Nz_file, 'rb'))
    nmag = len(mlo)
    t = Table.read(infile)
    redshift = t[z_col]
    mag = t['zmag']

    fig, axes = plt.subplots(nmag, 1, sharex=True, sharey=False)
    fig.set_size_inches(5, 7)
    fig.subplots_adjust(hspace=0, wspace=0)

    for imag in range(nmag):
        sel = (mlo[imag] <= mag) * (mag < mhi[imag])
        ax = axes[imag]
        ax.hist(redshift[sel], bins=zbins)
        be_int, err = scipy.integrate.quad(be_fit, zbins[0], zbins[-1], args=be_pars[:, imag])
        norm = (zbins[1]-zbins[0])*len(redshift[sel])/be_int
        # pdb.set_trace()
        ax.plot(zbins, norm*be_fit(zbins, be_pars[:, imag]))
        ax.text(0.7, 0.8, f"m=[{mlo[imag]}, {mhi[imag]}]",
                transform=ax.transAxes)
        print(be_pars[:, imag])
    axes[nmag-1].set_xlabel(r'Redshift')
    axes[nmag//2].set_ylabel(r'$N(z)$')
    axes[0].text(0.1, 1.1, f'rmin {rmin}, rmax {rmax}, weight {weight}, fitbin {fitbin}',
                transform=axes[0].transAxes)
    plt.show()


def hists(infile='14516.fits', Mbins=np.linspace(-26, -16, 41),
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
        M = t['habs'][sel]
        print(iz, np.percentile(M, (5, 50, 95)))
        ax = axes[iz]
        ax.semilogy()
        ax.hist(M, Mbins)
        ax.text(0.05, 0.8, rf'z = {zbins[iz]:3.1f}-{zbins[iz+1]:3.1f}',
            transform=ax.transAxes)
    plt.xlabel(r'$M_h$')
    plt.show()


def w_plot(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7],
           avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7,
           alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
           phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67], ic_corr=0):
    """w(theta) from angular pair counts in mag bins."""

    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for imag in range(nmag):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{w_prefix}RR_J{ijack}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{w_prefix}GG_J{ijack}_m{imag}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{w_prefix}GR_J{ijack}_m{imag}.pkl'
            (info, DR_counts) = pickle.load(open(infile, 'rb'))
            corrs.append(
                wcorr.Corr1d(info['Ngal'], info['Ngal'], info['Nran'], info['Nran'],
                             DD_counts, DR_counts, DR_counts, RR_counts,
                             mlo=info['mlo'], mhi=info['mhi']))
        corr = corrs[0]
        corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        if ic_corr:
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

    # wcorr.wplot_scale(cosmo, corr_slices, gamma1=gamma1, gamma2=gamma2,
    #                   r0=r0, eps=eps, alpha=alpha, Mstar=Mstar,
    #                   phistar=phistar, kcoeffs=kcoeffs)

# def w_plot_pred(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7], prefix='w_mag/',
#                 avgcounts=False, lf_pars='lf_pars.pkl', xi_pars='xi_pars.pkl',
#                 Mmin=-24, Mmax=-12, kcoeffs=[0.0, -0.39, 1.67], ic_corr=0,
#                 pltfile='integrand_plots.pdf'):
#     """Plot observed and predicted w(theta) in mag bins."""

#     if pltfile:
#         pdf = matplotlib.backends.backend_pdf.PdfPages(pltfile)
#     else:
#         pdf = None
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     corr_slices = []
#     for imag in range(nmag):
#         corrs = []
#         for ijack in range(njack+1):
#             infile = f'{prefix}RR_J{ijack}.pkl'
#             (info, RR_counts) = pickle.load(open(infile, 'rb'))
#             infile = f'{prefix}GG_J{ijack}_m{imag}.pkl'
#             (info, DD_counts) = pickle.load(open(infile, 'rb'))
#             infile = f'{prefix}GR_J{ijack}_m{imag}.pkl'
#             (info, DR_counts) = pickle.load(open(infile, 'rb'))
#             corrs.append(
#                 wcorr.Corr1d(info['Ngal'], info['Nran'],
#                              DD_counts, DR_counts, RR_counts,
#                              mlo=info['mlo'], mhi=info['mhi']))
#         corr = corrs[0]
#         corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
#         if ic_corr:
#             corr.ic_calc(fit_range, p0, 5)
#         corr_slices.append(corr)
#         color = next(ax._get_lines.prop_cycler)['color']
#         corr.plot(ax, color=color, label=f"m = [{info['mlo']}, {info['mhi']}]")

#         selfn = util.SelectionFunction(
#             cosmo, lf_pars=lf_pars, 
#             mlo=info['mlo'], mhi=info['mhi'], Mmin=Mmin, Mmax=Mmax,
#             nksamp=0, kcoeffs=kcoeffs, solid_angle=solid_angle)
#         w = []
#         plot_den = 1
#         for t in corr.sep:
#             wp = limber.w_lum(cosmo, selfn, t, info['mmean'],
#                               xi_pars, pdf=pdf, plot_den=plot_den)
#             plot_den = 0
#             w.append(wp)
#         ax.plot(corr.sep, w, color=color)
#         # popt, pcov = corr.fit_w(fit_range, p0, ax, color)
#         # print(popt, pcov)

#     if pdf:
#         pdf.close()
#     plt.loglog()
#     plt.legend()
#     plt.xlabel(r'$\theta$ / degrees')
#     plt.ylabel(r'$w(\theta)$')
#     plt.show()
#     plt.close(fig)


def be_fit(z, pars):
    """Generalised Baugh & Efstathiou (1993, eqn 7) model for N(z)."""
    (zc, alpha, beta, norm) = pars
    return norm * z**alpha * np.exp(-(z/zc)**beta)


def w_plot_pred(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7],
                avgcounts=False, lf_pars=lf_file,
                Nz_file=Nz_file, xi_pars=xi_file, w_prefix=w_prefix,
                ic_corr=0,  w_pred_file=f'w_pred_{band}.pkl',
                pltfile='integrand_plots.pdf'):
    """Plot observed and predicted w(theta) in mag bins.
    Use observed N(z) if Nz_file specified, otherwise use LF prediction."""

    kpoly, lf_dict, Mcen, zmean, lgphi = pickle.load(open(lf_pars, 'rb'))
    if Nz_file:
         # use observed N(z) rather than LF prediction
         counts_dict = pickle.load(open(Nz_file, 'rb'))

    plt.ioff()
    if pltfile:
        pdf = matplotlib.backends.backend_pdf.PdfPages(pltfile)
    else:
        pdf = None
    fig = plt.figure()
    ax = plt.subplot(111)
    corr_slices = []
    w_pred = {}
    chisq = 0
    nu = 0
    for imag in range(nmag):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{w_prefix}RR_J{ijack}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{w_prefix}GG_J{ijack}_m{imag}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{w_prefix}GR_J{ijack}_m{imag}.pkl'
            (info, DR_counts) = pickle.load(open(infile, 'rb'))
            corrs.append(
                wcorr.Corr1d(info['Ngal'], info['Ngal'], info['Nran'], info['Nran'],
                             DD_counts, DR_counts, DR_counts, RR_counts,
                             mlo=info['mlo'], mhi=info['mhi']))
        corr = corrs[0]
        corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        if ic_corr:
            corr.ic_calc(fit_range, p0, 5)
        corr_slices.append(corr)
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax, color=color, label=f"m = [{info['mlo']}, {info['mhi']}]")

        selfn = util.SelectionFunction(
            cosmo, lf_pars=lf_pars, 
            mlo=info['mlo'], mhi=info['mhi'], solid_angle=solid_angle, interp=1)
        if Nz_file:
            # Replace LF-calculated N(z) with smooth fit to observed
            (mlo, mhi, counts, popt) = counts_dict[imag]
            assert(info['mlo']==mlo and info['mhi']==mhi)
            selfn._Nz = be_fit(selfn._z, *popt)
        w = []
        plot_den = 1
        for t in corr.sep:
            wp = limber.w_lum_Nz(cosmo, selfn, t, info['mmean'],
                              xi_pars, kpoly, pdf=pdf, plot_den=plot_den)
            plot_den = 0
            w.append(wp)
        ax.plot(corr.sep, w, color=color)
        w_pred.update({imag: w})
        chisq += np.sum(((corr.est - w)/corr.err)**2)
        nu += len(w)

        # popt, pcov = corr.fit_w(fit_range, p0, ax, color)
        # print(popt, pcov)

    pickle.dump(w_pred, open(w_pred_file, 'wb'))
    if pdf:
        pdf.close()
    plt.loglog()
    plt.text(0.1, 0.1, rf'$\chi^2 = {chisq:4.1f}$', transform=ax.transAxes)
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()
    plt.close(fig)


def Nz(zbins=np.linspace(0.0, 2.0, 41), lf_pars=lf_file,
       interp=1, outfile=Nz_file):
    """Plot observed and predicted N(z) histograms in mag slices."""

    t = Table.read(infile)
    sel = t[band+'mag'] < magbins[-1]
    t = t[sel]
    mag, z = t[band+'mag'], t['true_redshift_gal']
    zcen = zbins[:-1] + 0.5*np.diff(zbins)
    zmin, zmax = zbins[0], zbins[-1]
    zp = np.linspace(zmin, zmax, 500)
    counts_dict = {'zbins': zbins, 'zcen': zcen}
    plt.clf()
    ax = plt.subplot(111)
    for imag in range(len(magbins) - 1):
        mlo, mhi = magbins[imag], magbins[imag+1]
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        color = next(ax._get_lines.prop_cycler)['color']
        counts, edges = np.histogram(z[sel], zbins)
        # spline = interpolate.UnivariateSpline(zcen, counts, bbox=(zmin, zmax),
        #                                       s=len(counts)*np.std(counts))
        # print(spline.roots(), spline.get_residual(), spline.get_knots(), spline.get_coeffs())
        #         pdb.set_trace()
        popt, pcov = scipy.optimize.curve_fit(be_fit, zcen, counts,
                                              p0=(0.5, 2.0, 1.5, 1e6))
        print(popt)

        counts_dict.update({imag: (mlo, mhi, counts, popt)})
        plt.stairs(counts, edges, color=color, label=f"m = {mlo}, {mhi}]")
        # plt.plot(zp, spline(zp), color=color, ls='-')
        plt.plot(zp, be_fit(zp, popt), color=color, ls='-')
        selfn = util.SelectionFunction(
            cosmo, lf_pars=lf_pars, 
            mlo=mlo, mhi=mhi, solid_angle=solid_angle,
            dz=zbins[1]-zbins[0], interp=interp)
        selfn.plot_Nz(ax, color=color, ls='--')

    pickle.dump(counts_dict, open(outfile, 'wb'))
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('N(z)')
    plt.show()


def mcmc(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7], prefix='w_mag/',
         avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7,
         alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
         phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67],
         nstep=[500, 10000]):
    """MCMC fit of LF and xi(r) params to optimise Limber scaling."""

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
        corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        corr.ic_calc(fit_range, p0, 5)
        corr_slices.append(corr)

    wcorr.mcmc(cosmo, corr_slices, gamma1=gamma1, gamma2=gamma2,
               r0=r0, eps=eps, alpha=alpha, Mstar=Mstar,
               phistar=phistar, kcoeffs=kcoeffs, nstep=nstep)

def xir_z_plot(nz=10, njack=10, fit_range=[0.1, 20], p0=[5, 1.7], prefix='xir_z/',
             avgcounts=False):
    """xi(r) from pair counts in redshift bins."""

    def xi_ev(z, A, eps):
        """Amplitude evolution of xi(r) = A * (1+z)**(-(3 + eps)."""
        return A * (1+z)**(-(3 + eps))
    
    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    zcen, r0, r0_err, gamma, gamma_err = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
    for iz in range(nz):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{prefix}GG_j{ijack}_z{iz}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GR_j{ijack}_z{iz}.pkl'
            (info, DR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}RR_j{ijack}_z{iz}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            corrs.append(
                wcorr.Corr1d(info['Ngal'], info['Nran'],
                             DD_counts, DR_counts, RR_counts))
        corr = corrs[0]
        corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        color = next(ax._get_lines.prop_cycler)['color']
        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
        print(popt, pcov)
        corr.plot(ax, color=color,
                  label=rf"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}], $r_0 = {popt[0]:3.2f} \pm {pcov[0][0]**0.5:3.2f}$, $\gamma = {popt[1]:3.2f} \pm {pcov[1][1]**0.5:3.2f}$")
        zcen[iz] = 0.5*(info['zlo'] + info['zhi'])
        r0[iz], gamma[iz] = popt[0], popt[1]
        r0_err[iz], gamma_err[iz] = pcov[0,0]**0.5, pcov[1,1]**0.5
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$r$ [Mpc/h]')
    plt.ylabel(r'$\xi(r)$')
    plt.show()

    gav = np.average(gamma, weights=gamma_err**-2)
    gerr = np.sum(gamma_err**-2)**-0.5
    popt, pcov = scipy.optimize.curve_fit(
        xi_ev, zcen, r0, p0=(4, 0), sigma=r0_err)
    print(popt)
    fig, axes = plt.subplots(2, 1, sharex=True, num=2)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].errorbar(zcen, r0, r0_err)
    axes[0].plot(zcen, xi_ev(zcen, *popt))
    axes[0].text(0.5, 0.8,
                 rf'$\epsilon = {popt[1]:5.3f} \pm {pcov[1,1]**0.5:5.3f}$',
                 transform=axes[0].transAxes)
    axes[0].set_ylabel('r0')
    axes[1].errorbar(zcen, gamma, gamma_err)
    axes[1].text(0.5, 0.8,
                 rf'$\gamma = {gav:5.3f} \pm {gerr:5.3f}$',
                 transform=axes[1].transAxes)
    axes[1].set_ylabel('gamma')
    axes[1].set_xlabel('z')
    plt.show()


# def r0_fun(p, M, z):
#     return (p[0] + p[1]*(M+20) + p[2]*(M+20)**2) * (1+z)**(-(3 + p[3]))

# def gam_fun(p, M, z):
#     return (p[0] + p[1]*(M+20) + p[2]*(M+20)**2) * (1+z)**(-(3 + p[3]))

def r0_fun(p, M, z):
#    return p[0] + p[1]*z + p[2]*(M+20) + p[3]*np.log10(1+z)*(M+20)**2
    return np.polynomial.polynomial.polyval2d(M+20, z, p)

def gam_fun(p, M, z):
#     return p[0] + p[1]*z + p[2]*(M+20) + p[3]*np.log10(1+z)*(M+20)**2
    return np.polynomial.polynomial.polyval2d(M+20, z, p)

def xir_M_z_plot_corrfunc(nm=5, nz=10, njack=9, fit_range=[0.1, 20], p0=[5, 1.7],
                 avgcounts=False, Ngal_min=500,
                 outfile=xi_file, xiscale=0):
    """xi(r) from pair counts in Magnitude and redshift bins using corrfunc."""

    prefix = f'xir_M_z_{band}/'
    
    # def xi_ev(z, A, eps):
    #     """Amplitude evolution of xi(r) = A * (1+z)**(-(3 + eps)."""
    #     return A * (1+z)**(-(3 + eps))
    
    # def r0_fun(Mz, r0_0, am, Mstar, eps):
    #     """r0 = r0_0 * (1+z)**(-(3 + eps) + am * L/L*."""
    #     M, z = Mz[0, :], Mz[1, :]
    #     return r0_0 * (1+z)**(-(3 + eps)) + am * 10**(0.4*(Mstar - M)) 
    
    # def gam_fun(Mz, gam_0, am, Mstar, eps):
    #     """gamma = gam_0 * (1+z)**(-(3 + eps) * am * L/L*."""
    #     M, z = Mz[0, :], Mz[1, :]
    #     return gam_0 * (1+z)**(-(3 + eps)) + am * 10**(0.4*(Mstar - M)) 

    def r0_resid(p, M, z, r0, r0_err):
        return ((r0_fun(p.reshape((nmp, nzp)), M, z) - r0)/r0_err)**2
    
    def gam_resid(p, M, z, gam, gam_err):
        return ((gam_fun(p.reshape((nmp, nzp)), M, z) - gam)/gam_err)**2
    
    def power_law(r, r0, gamma):
        """Power law xi(r) = (r0/r)**gamma."""
        return (r0/r)**gamma

    # Read model fit parameters from previous run to add to first plot
    xi_dict = pickle.load(open(outfile, 'rb'))

    plt.ion()
    plt.clf()
    # Fig 1: xi(r) colour coded by M in panels of z
    # fig, axes = plt.subplots(2, nz//2, sharex=True, sharey=True, num=1)
    fig, axes = plt.subplots(2, nz//2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    
    if xiscale:
        axes[0, 0].set_ylabel(r'$r^2 \xi(r)$')
        axes[1, 0].set_ylabel(r'$r^2 \xi(r)$')
    else:
        axes[0, 0].set_ylabel(r'$\xi(r)$')
        axes[1, 0].set_ylabel(r'$\xi(r)$')
    axes[1, nz//4].set_xlabel(r'$r$ [Mpc/h]')
    corr_slices = []
    Mmean, zmean, r0, r0_err, gamma, gamma_err = np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz))
    zlo, zhi =  np.zeros(nz), np.zeros(nz)
    for iz in range(nz):
        ax = axes.flatten()[iz]
        for im in range(nm):
            color = next(ax._get_lines.prop_cycler)['color']
            try:
                corrs = []
                for ijack in range(njack+1):
                    infile = f'{prefix}GG_j{ijack}_z{iz}_M{im}.pkl'
                    if ijack == 0:
                        (info, DD_counts) = pickle.load(open(infile, 'rb'))
                    else:
                        (_, DD_counts) = pickle.load(open(infile, 'rb'))
                    infile = f'{prefix}GR_j{ijack}_z{iz}_M{im}.pkl'
                    (_, DR_counts) = pickle.load(open(infile, 'rb'))
                    infile = f'{prefix}RR_j{ijack}_z{iz}_M{im}.pkl'
                    (_, RR_counts) = pickle.load(open(infile, 'rb'))
                    corr = wcorr.Corr1d()
                    corr.ngal = info['Ngal']
                    corr.nran = info['Nran']
                    corr.dd = DD_counts
                    corr.dr = DR_counts
                    corr.rr = RR_counts
                    corr.est = np.nan_to_num(
                        Corrfunc.utils.convert_3d_counts_to_cf(
                        corr.ngal, corr.ngal, corr.nran, corr.nran,
                        corr.dd, corr.dr, corr.dr, corr.rr))

                    corrs.append(corr)
                if info['Ngal'] > Ngal_min:
                    corr = corrs[0]
                    corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
                    if xiscale:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                             plot_scale=corr.sep**2)
                        ax.errorbar(corr.sep, corr.sep**2*corr.est,
                                    corr.sep**2*corr.err, color=color, fmt='o',
                                    label=rf"$M_{band} = [{info['Mlo']:3.1f}, {info['Mhi']:3.1f}]$")
                    else:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
                        ax.errorbar(corr.sep, corr.est,
                                    corr.err, color=color, fmt='o',
                                    label=rf"$M_{band} = [{info['Mlo']:3.1f}, {info['Mhi']:3.1f}]$")
                    # corr.plot(ax, color=color,
                    #           label=rf"M = [{info['Mlo']:3.1f}, {info['Mhi']:3.1f}], $r_0 = {popt[0]:3.2f} \pm {pcov[0][0]**0.5:3.2f}$, $\gamma = {popt[1]:3.2f} \pm {pcov[1][1]**0.5:3.2f}$")

                    # Show results from previous fit
                    try:
                        Mmean[im, iz] = info['Mmean']
                        zmean[im, iz] = info['zmean']
                        r0m = xi_dict['r0_fun'](xi_dict['r0_pars'],
                                                Mmean[im, iz], zmean[im, iz])
                        gammam = xi_dict['gam_fun'](xi_dict['gam_pars'],
                                                    Mmean[im, iz], zmean[im, iz])
                        if xiscale:
                            ax.plot(corr.sep, corr.sep**2*power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                        else:
                            ax.plot(corr.sep, power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                    except IndexError:
                        pass
                    
                    r0[im, iz], gamma[im, iz] = popt[0], popt[1]
                    r0_err[im, iz], gamma_err[im, iz] = pcov[0,0]**0.5, pcov[1,1]**0.5
                    corr.ic_calc_xi(fit_range)
                    print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'],
                          info['Ngal'], r0[im, iz], gamma[im, iz], corr.ic)
                    # if iz == 0 and im == 0:
                    #     pdb.set_trace()
                else:
                    print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'],
                          info['Ngal'])
            except FileNotFoundError:
                print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'], ' no data')

        zlo[iz], zhi[iz] = info['zlo'], info['zhi']
        ax.text(0.3, 0.9, rf"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}]",
                transform=ax.transAxes)
        # ax.legend()
    plt.xlim(0.1, 99)
    plt.ylim(0.005, 1000)
    plt.loglog()
    # ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels)
    lines_labels = [axes.flatten()[nz//2].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax = axes.flatten()[-1]
    ax.legend(lines, labels)
    plt.show()

    # Fig 4: xi(r) colour coded by z in panels of M
    # fig, axes = plt.subplots(2, nz//2, sharex=True, sharey=True, num=1)
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, num=4)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    
    if xiscale:
        axes[0, 0].set_ylabel(r'$r^2 \xi(r)$')
        axes[1, 0].set_ylabel(r'$r^2 \xi(r)$')
    else:
        axes[0, 0].set_ylabel(r'$\xi(r)$')
        axes[1, 0].set_ylabel(r'$\xi(r)$')
    axes[1, 1].set_xlabel(r'$r$ [Mpc/h]')
    corr_slices = []
    for im in range(nm):
        ax = axes.flatten()[im]
        for iz in range(nz):
            color = next(ax._get_lines.prop_cycler)['color']
            try:
                corrs = []
                for ijack in range(njack+1):
                    infile = f'{prefix}GG_j{ijack}_z{iz}_M{im}.pkl'
                    if ijack == 0:
                        (info, DD_counts) = pickle.load(open(infile, 'rb'))
                    else:
                        (_, DD_counts) = pickle.load(open(infile, 'rb'))
                    infile = f'{prefix}GR_j{ijack}_z{iz}_M{im}.pkl'
                    (_, DR_counts) = pickle.load(open(infile, 'rb'))
                    infile = f'{prefix}RR_j{ijack}_z{iz}_M{im}.pkl'
                    (_, RR_counts) = pickle.load(open(infile, 'rb'))
                    corrs.append(
                        wcorr.Corr1d(
                            info['Ngal'], info['Ngal'], info['Nran'], info['Nran'],
                            DD_counts, DR_counts, DR_counts, RR_counts))
                if info['Ngal'] > Ngal_min:
                    corr = corrs[0]
                    corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
                    if xiscale:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                             plot_scale=corr.sep**2)
                        ax.errorbar(corr.sep, corr.sep**2*corr.est,
                                    corr.sep**2*corr.err, color=color, fmt='o',
                                    label=rf"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}]")
                    else:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
                        ax.errorbar(corr.sep, corr.est,
                                    corr.err, color=color, fmt='o',
                                    label=rf"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}]")

                    # Show results from previous fit
                    try:
                        Mmean[im, iz] = info['Mmean']
                        zmean[im, iz] = info['zmean']
                        r0m = xi_dict['r0_fun'](xi_dict['r0_pars'],
                                                Mmean[im, iz], zmean[im, iz])
                        gammam = xi_dict['gam_fun'](xi_dict['gam_pars'],
                                                    Mmean[im, iz], zmean[im, iz])
                        if xiscale:
                            ax.plot(corr.sep, corr.sep**2*power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                        else:
                            ax.plot(corr.sep, power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                    except IndexError:
                        pass
                    
                    corr.ic_calc_xi(fit_range)
                    print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'],
                          info['Ngal'], r0[im, iz], gamma[im, iz], corr.ic)
                    # if iz == 0 and im == 0:
                    #     pdb.set_trace()
                else:
                    print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'],
                          info['Ngal'])
            except FileNotFoundError:
                print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'], ' no data')

        ax.text(0.3, 0.9, rf"M = [{info['Mlo']:3.1f}, {info['Mhi']:3.1f}]",
                transform=ax.transAxes)
        # ax.legend()
    plt.xlim(0.1, 99)
    plt.ylim(0.005, 1000)
    plt.loglog()
    # ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels)
    lines_labels = [axes.flatten()[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax = axes.flatten()[-1]
    ax.legend(lines, labels, loc='lower left')
    plt.show()

    # Mask out arrays for ill-defined (M,z) bins
    mask = Mmean > -5
    Mmean = ma.masked_array(Mmean, mask)
    zmean = ma.masked_array(zmean, mask)
    gamma = ma.masked_array(gamma, mask)
    gamma_err = ma.masked_array(gamma_err, mask)
    r0 = ma.masked_array(r0, mask)
    r0_err = ma.masked_array(r0_err, mask)
    Mc = Mmean.compressed()
    zc = zmean.compressed()
    gammac = gamma.compressed()
    r0c = r0.compressed()
    gammac_err = gamma_err.compressed()
    r0c_err = r0_err.compressed()

    # Plot gamma and r0 as function of (M, z) and interpolate

    # 2D grid for plotting results
    X = np.linspace(-26, -12, 57)
    Y = np.linspace(0, 2, 41)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    gamma_lin_interp = LinearNDInterpolator(list(zip(Mc, zc)), gammac, rescale=1)
    gamma_nn_interp = NearestNDInterpolator(list(zip(Mc, zc)), gammac, rescale=1)
    # ginterp = scipy.interpolate.interp2d(Mc, zc, gammac)
    gam_map = gamma_lin_interp(X, Y)
    bad = np.isnan(gam_map)
    gam_map[bad] = gamma_nn_interp(X, Y)[bad]
    
    # gam_spline = SmoothBivariateSpline(Mc, zc, gammac, bbox=bbox, s=0)
    # print('gamma knots', gam_spline.get_knots())
    # gam_map = gam_spline(X, Y)
    # tck = bisplrep(Mc, zc, gammac, gammac_err**-2, *bbox, task=-1,
    #                tx=tx, ty=ty, quiet=0)
    # tck = bisplrep(Mc, zc, gammac, gammac_err**-2, *bbox, s=0.1, nxest=50, nyest=50)
    # tck = bisplrep(Mc, zc, gammac)
    # print(tck)
    # gam_map = bisplev(X, Y, tck)

    r0_lin_interp = LinearNDInterpolator(list(zip(Mc, zc)), r0c, rescale=1)
    r0_nn_interp = NearestNDInterpolator(list(zip(Mc, zc)), r0c, rescale=1)
    # rinterp = scipy.interpolate.interp2d(Mc, zc, r0c)
    r0_map = r0_lin_interp(X, Y)
    bad = np.isnan(r0_map)
    r0_map[bad] = r0_nn_interp(X, Y)[bad]
    
    # r0_spline = SmoothBivariateSpline(Mc, zc, r0c, bbox=bbox, s=0)
    # print('r0 knots', r0_spline.get_knots())
    # r0_map = r0_spline(X, Y)
    # tck = bisplrep(Mc, zc, r0c, r0c_err**-2, *bbox, task=-1,
    #                tx=tx, ty=ty, quiet=0)
    # tck = bisplrep(Mc, zc, r0c, r0c_err**-2, *bbox, s=0.1, nxest=50, nyest=50)
    # tck = bisplrep(Mc, zc, r0c)
    # print(tck)
    # r0_map = bisplev(X, Y, tck)
    
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=2)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].pcolormesh(X, Y, gam_map, shading='auto')
    scatter = axes[0].scatter(Mc, zc, c=gammac)
    plt.colorbar(scatter, ax=axes[0], label='gammac', location='top')
    axes[0].set_xlabel('M')
    axes[0].set_ylabel('z')
    axes[1].pcolormesh(X, Y, r0_map, shading='auto')
    scatter = axes[1].scatter(Mc, zc, c=r0c)
    plt.colorbar(scatter, ax=axes[1], label='r0', location='top')
    axes[1].set_xlabel('M')
    # axes[1].set_yticklabels([])
    plt.show()

    # Fit and plot model fits to gamma(M, z) and r0(M, z)
    # ok = ~Mmean.mask
    # Mz = np.ma.vstack((Mmean[ok].flatten(), zmean[ok].flatten()))
    # r0_popt, r0_pcov = scipy.optimize.curve_fit(
    #     r0_fun, Mz, r0[ok].flatten(), p0=(4, 0.1, -21, 0),
    #     sigma=r0_err[ok].flatten(), ftol=0.001, xtol=0.001)
    # gam_popt, gam_pcov = scipy.optimize.curve_fit(
    #     gam_fun, Mz, gamma[ok].flatten(), p0=(4, 0.1, -21, 0),
    #     sigma=gamma_err[ok].flatten(), ftol=0.001, xtol=0.001)

    # x0 = (4, 0, 0, 0)
    nmp = 4
    nzp = 4
    x0 = np.ones((nmp, nzp))

    r0_pars, cov, info, mesg, ier = scipy.optimize.leastsq(
        r0_resid, x0=x0, args=(Mc, zc, r0c, r0c_err),
        full_output=True, ftol=0.001, xtol=0.001)
    chisq = np.sum(info['fvec']**2)
    nu = len(Mc) - len(r0_pars)
    r0_pars = r0_pars.reshape((nmp, nzp))
    print('r0 fit pars:', r0_pars, 'chi2:', chisq, 'nu:', nu)
    
    gam_pars, cov, info, mesg, ier = scipy.optimize.leastsq(
        gam_resid, x0=x0, args=(Mc, zc, gammac, gammac_err),
        full_output=True, ftol=0.001, xtol=0.001)
    chisq = np.sum(info['fvec']**2)
    nu = len(Mc) - len(gam_pars)
    gam_pars = gam_pars.reshape((nmp, nzp))
    print('gamma fit pars:', gam_pars, 'chi2:', chisq, 'nu:', nu)
    
    xi_dict = {'r0_fun': r0_fun, 'r0_pars': r0_pars,
               'gam_fun': gam_fun, 'gam_pars': gam_pars,
               'r0_lin_interp': r0_lin_interp, 'r0_nn_interp': r0_nn_interp,
               'gamma_lin_interp': gamma_lin_interp,
               'gamma_nn_interp': gamma_nn_interp}
    pickle.dump(xi_dict, open(outfile, 'wb'))

    fig, axes = plt.subplots(2, nz, sharex=True, sharey='row', num=3)
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0, 0].set_ylabel(r'$r_0$')
    axes[1, 0].set_ylabel(r'$\gamma$')
    axes[1, 4].set_xlabel(r'$M_r$')
    for iy in range(2):
        for iz in range(nz):
            # pdb.set_trace()
            ax = axes[iy, iz]
            if iy == 0:
                ax.text(0.02, 0.9, rf"z = [{zlo[iz]:3.1f}, {zhi[iz]:3.1f}]",
                transform=ax.transAxes)
                ax.errorbar(Mmean[:, iz], r0[:, iz], r0_err[:, iz])
                ax.plot(Mmean[:, iz],
                         r0_fun(r0_pars, Mmean[:, iz], zmean[:, iz]))
                ax.set_ylim(3, 10)
            else:
                ax.errorbar(Mmean[:, iz], gamma[:, iz], gamma_err[:, iz])
                ax.plot(Mmean[:, iz],
                         gam_fun(gam_pars, Mmean[:, iz], zmean[:, iz]))
                ax.set_ylim(1, 2.5)
    plt.show()
   
    # Mz = np.ma.vstack((Mmean[ok].flatten(), zmean[ok].flatten())).T
    # r0_gp = SymbolicRegressor(population_size=5000,
    #                            generations=50, stopping_criteria=0.01,
    #                            const_range=(-100, 100),
    #                            p_crossover=0.7, p_subtree_mutation=0.1,
    #                            p_hoist_mutation=0.05, p_point_mutation=0.1,
    #                            max_samples=0.9, verbose=1,
    #                            parsimony_coefficient=0.001, random_state=0)
    # r0_gp.fit(Mz, r0[ok], sample_weight=r0_err[ok]**-2)
    # print('r0:', r0_gp._program)
    
    # gam_gp = SymbolicRegressor(population_size=5000,
    #                            generations=50, stopping_criteria=0.01,
    #                            const_range=(-100, 100),
    #                            p_crossover=0.7, p_subtree_mutation=0.1,
    #                            p_hoist_mutation=0.05, p_point_mutation=0.1,
    #                            max_samples=0.9, verbose=1,
    #                            parsimony_coefficient=0.001, random_state=0)
    # gam_gp.fit(Mz, gamma[ok], sample_weight=gamma_err[ok]**-2)
    # print('gamma:', gam_gp._program)
    
    # fig, axes = plt.subplots(2, nz, sharex=True, sharey='row', num=4)
    # fig.set_size_inches(16, 8)
    # fig.subplots_adjust(hspace=0, wspace=0)
    # axes[0, 0].set_ylabel(r'$r_0$')
    # axes[1, 0].set_ylabel(r'$\gamma$')
    # axes[1, 2].set_xlabel(r'$M_r$')
    # for iy in range(2):
    #     for iz in range(nz):
    #         # pdb.set_trace()
    #         Mz = np.ma.vstack((Mmean[:, iz], zmean[:, iz])).T
    #         ax = axes[iy, iz]
    #         if iy == 0:
    #             ax.text(0.5, 0.9, rf"z = [{zlo[iz]:3.1f}, {zhi[iz]:3.1f}]",
    #             transform=ax.transAxes)
    #             ax.errorbar(Mmean[:, iz], r0[:, iz], r0_err[:, iz])
    #             ax.plot(Mmean[:, iz], r0_gp.predict(Mz))
    #         else:
    #             ax.errorbar(Mmean[:, iz], gamma[:, iz], gamma_err[:, iz])
    #             ax.plot(Mmean[:, iz], gam_gp.predict(Mz))
    # plt.show()


def xir_M_z_plot(nm=5, nz=10, njack=9, fit_range=[0.1, 20], p0=[5, 1.7],
                 avgcounts=False, Ngal_min=500,
                 outfile=xi_file, xiscale=0):
    """xi(r) from pair counts in Magnitude and redshift bins from treecorr."""

    prefix = f'xir_M_z_{band}/'
    
    # def xi_ev(z, A, eps):
    #     """Amplitude evolution of xi(r) = A * (1+z)**(-(3 + eps)."""
    #     return A * (1+z)**(-(3 + eps))
    
    # def r0_fun(Mz, r0_0, am, Mstar, eps):
    #     """r0 = r0_0 * (1+z)**(-(3 + eps) + am * L/L*."""
    #     M, z = Mz[0, :], Mz[1, :]
    #     return r0_0 * (1+z)**(-(3 + eps)) + am * 10**(0.4*(Mstar - M)) 
    
    # def gam_fun(Mz, gam_0, am, Mstar, eps):
    #     """gamma = gam_0 * (1+z)**(-(3 + eps) * am * L/L*."""
    #     M, z = Mz[0, :], Mz[1, :]
    #     return gam_0 * (1+z)**(-(3 + eps)) + am * 10**(0.4*(Mstar - M)) 

    def r0_resid(p, M, z, r0, r0_err):
        return ((r0_fun(p.reshape((nmp, nzp)), M, z) - r0)/r0_err)**2
    
    def gam_resid(p, M, z, gam, gam_err):
        return ((gam_fun(p.reshape((nmp, nzp)), M, z) - gam)/gam_err)**2
    
    def power_law(r, r0, gamma):
        """Power law xi(r) = (r0/r)**gamma."""
        return (r0/r)**gamma

    # Read model fit parameters from previous run to add to first plot
    xi_dict = pickle.load(open(outfile, 'rb'))

    plt.ion()
    plt.clf()
    # Fig 1: xi(r) colour coded by M in panels of z
    # fig, axes = plt.subplots(2, nz//2, sharex=True, sharey=True, num=1)
    fig, axes = plt.subplots(2, nz//2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    
    if xiscale:
        axes[0, 0].set_ylabel(r'$r^2 \xi(r)$')
        axes[1, 0].set_ylabel(r'$r^2 \xi(r)$')
    else:
        axes[0, 0].set_ylabel(r'$\xi(r)$')
        axes[1, 0].set_ylabel(r'$\xi(r)$')
    axes[1, nz//4].set_xlabel(r'$r$ [Mpc/h]')
    corr_slices = []
    Mmean, zmean, r0, r0_err, gamma, gamma_err = np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz))
    zlo, zhi =  np.zeros(nz), np.zeros(nz)
    for iz in range(nz):
        ax = axes.flatten()[iz]
        for im in range(nm):
            color = next(ax._get_lines.prop_cycler)['color']
            try:
                infile = f'{prefix}xir_iz{iz}_im{im}.fits'
                corr = wcorr.Corr1d(infile)
                if corr.meta['NGAL'] > Ngal_min:
                    if xiscale:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                                plot_scale=corr.sep**2)
                        ax.errorbar(corr.sep, corr.sep**2*corr.est_corr(),
                                    corr.sep**2*corr.err, color=color, fmt='o',
                                    label=rf"$M_{band} = [{corr.meta['MLO']:3.1f}, {corr.meta['MHI']:3.1f}]$")
                    else:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
                        ax.errorbar(corr.sep, corr.est_corr(),
                                    corr.err, color=color, fmt='o',
                                    label=rf"$M_{band} = [{corr.meta['MLO']:3.1f}, {corr.meta['MHI']:3.1f}]$")

                    # Show results from previous fit
                    try:
                        Mmean[im, iz] = corr.meta['MMEAN']
                        zmean[im, iz] = corr.meta['ZMEAN']
                        r0m = xi_dict['r0_fun'](xi_dict['r0_pars'],
                                                Mmean[im, iz], zmean[im, iz])
                        gammam = xi_dict['gam_fun'](xi_dict['gam_pars'],
                            Mmean[im, iz], zmean[im, iz])
                        if xiscale:
                            ax.plot(corr.sep, corr.sep**2*power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                        else:
                            ax.plot(corr.sep, power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                    except IndexError:
                        pass
                    
                    r0[im, iz], gamma[im, iz] = popt[0], popt[1]
                    r0_err[im, iz], gamma_err[im, iz] = pcov[0,0]**0.5, pcov[1,1]**0.5
                    corr.ic_calc_xi(fit_range)
                    print(corr.meta['ZLO'], corr.meta['ZHI'], corr.meta['MLO'], corr.meta['MHI'],
                            corr.meta['NGAL'], r0[im, iz], gamma[im, iz], corr.ic)
                    # if iz == 0 and im == 0:
                    #     pdb.set_trace()
                else:
                    print(corr.meta['ZLO'], corr.meta['ZHI'], corr.meta['MLO'], corr.meta['MHI'],
                            corr.meta['NGAL'])
            except FileNotFoundError:
                print(iz, im, ' no data')

        zlo[iz], zhi[iz] = corr.meta['ZLO'], corr.meta['ZHI']
        ax.text(0.3, 0.9, rf"z = [{corr.meta['ZLO']:3.1f}, {corr.meta['ZHI']:3.1f}]",
                transform=ax.transAxes)
        # ax.legend()
    plt.xlim(0.1, 99)
    plt.ylim(0.005, 1000)
    plt.loglog()
    # ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels)
    lines_labels = [axes.flatten()[nz//2].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax = axes.flatten()[-1]
    ax.legend(lines, labels)
    plt.show()

    # Fig 4: xi(r) colour coded by z in panels of M
    # fig, axes = plt.subplots(2, nz//2, sharex=True, sharey=True, num=1)
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, num=4)
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    
    if xiscale:
        axes[0, 0].set_ylabel(r'$r^2 \xi(r)$')
        axes[1, 0].set_ylabel(r'$r^2 \xi(r)$')
    else:
        axes[0, 0].set_ylabel(r'$\xi(r)$')
        axes[1, 0].set_ylabel(r'$\xi(r)$')
    axes[1, 1].set_xlabel(r'$r$ [Mpc/h]')
    corr_slices = []
    for im in range(nm):
        ax = axes.flatten()[im]
        for iz in range(nz):
            color = next(ax._get_lines.prop_cycler)['color']
            try:
                infile = f'{prefix}xir_iz{iz}_im{im}.fits'
                corr = wcorr.Corr1d(infile)
                if corr.meta['NGAL'] > Ngal_min:
                    if xiscale:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                             plot_scale=corr.sep**2)
                        ax.errorbar(corr.sep, corr.sep**2*corr.est_corr(),
                                    corr.sep**2*corr.err, color=color, fmt='o',
                                    label=rf"z = [{corr.meta['ZLO']:3.1f}, {corr.meta['ZHI']:3.1f}]")
                    else:
                        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
                        ax.errorbar(corr.sep, corr.est_corr(),
                                    corr.err, color=color, fmt='o',
                                    label=rf"z = [{corr.meta['ZLO']:3.1f}, {corr.meta['ZHI']:3.1f}]")

                    # Show results from previous fit
                    try:
                        Mmean[im, iz] = corr.meta['MMEAN']
                        zmean[im, iz] = corr.meta['ZMEAN']
                        r0m = xi_dict['r0_fun'](xi_dict['r0_pars'],
                                                Mmean[im, iz], zmean[im, iz])
                        gammam = xi_dict['gam_fun'](xi_dict['gam_pars'],
                                                    Mmean[im, iz], zmean[im, iz])
                        if xiscale:
                            ax.plot(corr.sep, corr.sep**2*power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                        else:
                            ax.plot(corr.sep, power_law(
                                corr.sep, r0m, gammam), '--', color=color)
                    except IndexError:
                        pass
                    
                    corr.ic_calc_xi(fit_range)
                    print(corr.meta['ZLO'], corr.meta['ZHI'], corr.meta['MLO'], corr.meta['MHI'],
                          corr.meta['NGAL'], r0[im, iz], gamma[im, iz], corr.ic)
                    # if iz == 0 and im == 0:
                    #     pdb.set_trace()
                else:
                    print(corr.meta['ZLO'], corr.meta['ZHI'], corr.meta['MLO'], corr.meta['MHI'],
                          corr.meta['NGAL'])
            except FileNotFoundError:
                print(iz, im, ' no data')

        ax.text(0.3, 0.9, rf"M = [{corr.meta['MLO']:3.1f}, {corr.meta['MHI']:3.1f}]",
                transform=ax.transAxes)
        # ax.legend()
    plt.xlim(0.1, 99)
    plt.ylim(0.005, 1000)
    plt.loglog()
    # ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels)
    lines_labels = [axes.flatten()[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax = axes.flatten()[-1]
    ax.legend(lines, labels, loc='lower left')
    plt.show()

    # Mask out arrays for ill-defined (M,z) bins
    mask = Mmean > -5
    Mmean = ma.masked_array(Mmean, mask)
    zmean = ma.masked_array(zmean, mask)
    gamma = ma.masked_array(gamma, mask)
    gamma_err = ma.masked_array(gamma_err, mask)
    r0 = ma.masked_array(r0, mask)
    r0_err = ma.masked_array(r0_err, mask)
    Mc = Mmean.compressed()
    zc = zmean.compressed()
    gammac = gamma.compressed()
    r0c = r0.compressed()
    gammac_err = gamma_err.compressed()
    r0c_err = r0_err.compressed()

    # Plot gamma and r0 as function of (M, z) and interpolate

    # 2D grid for plotting results
    X = np.linspace(-26, -12, 57)
    Y = np.linspace(0, 2, 41)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    gamma_lin_interp = LinearNDInterpolator(list(zip(Mc, zc)), gammac, rescale=1)
    gamma_nn_interp = NearestNDInterpolator(list(zip(Mc, zc)), gammac, rescale=1)
    # ginterp = scipy.interpolate.interp2d(Mc, zc, gammac)
    gam_map = gamma_lin_interp(X, Y)
    bad = np.isnan(gam_map)
    gam_map[bad] = gamma_nn_interp(X, Y)[bad]
    
    # gam_spline = SmoothBivariateSpline(Mc, zc, gammac, bbox=bbox, s=0)
    # print('gamma knots', gam_spline.get_knots())
    # gam_map = gam_spline(X, Y)
    # tck = bisplrep(Mc, zc, gammac, gammac_err**-2, *bbox, task=-1,
    #                tx=tx, ty=ty, quiet=0)
    # tck = bisplrep(Mc, zc, gammac, gammac_err**-2, *bbox, s=0.1, nxest=50, nyest=50)
    # tck = bisplrep(Mc, zc, gammac)
    # print(tck)
    # gam_map = bisplev(X, Y, tck)

    r0_lin_interp = LinearNDInterpolator(list(zip(Mc, zc)), r0c, rescale=1)
    r0_nn_interp = NearestNDInterpolator(list(zip(Mc, zc)), r0c, rescale=1)
    # rinterp = scipy.interpolate.interp2d(Mc, zc, r0c)
    r0_map = r0_lin_interp(X, Y)
    bad = np.isnan(r0_map)
    r0_map[bad] = r0_nn_interp(X, Y)[bad]
    
    # r0_spline = SmoothBivariateSpline(Mc, zc, r0c, bbox=bbox, s=0)
    # print('r0 knots', r0_spline.get_knots())
    # r0_map = r0_spline(X, Y)
    # tck = bisplrep(Mc, zc, r0c, r0c_err**-2, *bbox, task=-1,
    #                tx=tx, ty=ty, quiet=0)
    # tck = bisplrep(Mc, zc, r0c, r0c_err**-2, *bbox, s=0.1, nxest=50, nyest=50)
    # tck = bisplrep(Mc, zc, r0c)
    # print(tck)
    # r0_map = bisplev(X, Y, tck)
    
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=2)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].pcolormesh(X, Y, gam_map, shading='auto')
    scatter = axes[0].scatter(Mc, zc, c=gammac)
    plt.colorbar(scatter, ax=axes[0], label='gammac', location='top')
    axes[0].set_xlabel('M')
    axes[0].set_ylabel('z')
    axes[1].pcolormesh(X, Y, r0_map, shading='auto')
    scatter = axes[1].scatter(Mc, zc, c=r0c)
    plt.colorbar(scatter, ax=axes[1], label='r0', location='top')
    axes[1].set_xlabel('M')
    # axes[1].set_yticklabels([])
    plt.show()

    # Fit and plot model fits to gamma(M, z) and r0(M, z)
    # ok = ~Mmean.mask
    # Mz = np.ma.vstack((Mmean[ok].flatten(), zmean[ok].flatten()))
    # r0_popt, r0_pcov = scipy.optimize.curve_fit(
    #     r0_fun, Mz, r0[ok].flatten(), p0=(4, 0.1, -21, 0),
    #     sigma=r0_err[ok].flatten(), ftol=0.001, xtol=0.001)
    # gam_popt, gam_pcov = scipy.optimize.curve_fit(
    #     gam_fun, Mz, gamma[ok].flatten(), p0=(4, 0.1, -21, 0),
    #     sigma=gamma_err[ok].flatten(), ftol=0.001, xtol=0.001)

    # x0 = (4, 0, 0, 0)
    nmp = 4
    nzp = 4
    x0 = np.ones((nmp, nzp))

    r0_pars, cov, info, mesg, ier = scipy.optimize.leastsq(
        r0_resid, x0=x0, args=(Mc, zc, r0c, r0c_err),
        full_output=True, ftol=0.001, xtol=0.001)
    chisq = np.sum(info['fvec']**2)
    nu = len(Mc) - len(r0_pars)
    r0_pars = r0_pars.reshape((nmp, nzp))
    print('r0 fit pars:', r0_pars, 'chi2:', chisq, 'nu:', nu)
    
    gam_pars, cov, info, mesg, ier = scipy.optimize.leastsq(
        gam_resid, x0=x0, args=(Mc, zc, gammac, gammac_err),
        full_output=True, ftol=0.001, xtol=0.001)
    chisq = np.sum(info['fvec']**2)
    nu = len(Mc) - len(gam_pars)
    gam_pars = gam_pars.reshape((nmp, nzp))
    print('gamma fit pars:', gam_pars, 'chi2:', chisq, 'nu:', nu)
    
    xi_dict = {'r0_fun': r0_fun, 'r0_pars': r0_pars,
               'gam_fun': gam_fun, 'gam_pars': gam_pars,
               'r0_lin_interp': r0_lin_interp, 'r0_nn_interp': r0_nn_interp,
               'gamma_lin_interp': gamma_lin_interp,
               'gamma_nn_interp': gamma_nn_interp}
    pickle.dump(xi_dict, open(outfile, 'wb'))

    fig, axes = plt.subplots(2, nz, sharex=True, sharey='row', num=3)
    fig.set_size_inches(12, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0, 0].set_ylabel(r'$r_0$')
    axes[1, 0].set_ylabel(r'$\gamma$')
    axes[1, 4].set_xlabel(r'$M_r$')
    for iy in range(2):
        for iz in range(nz):
            # pdb.set_trace()
            ax = axes[iy, iz]
            if iy == 0:
                ax.text(0.02, 0.9, rf"z = [{zlo[iz]:3.1f}, {zhi[iz]:3.1f}]",
                transform=ax.transAxes)
                ax.errorbar(Mmean[:, iz], r0[:, iz], r0_err[:, iz])
                ax.plot(Mmean[:, iz],
                         r0_fun(r0_pars, Mmean[:, iz], zmean[:, iz]))
                ax.set_ylim(3, 10)
            else:
                ax.errorbar(Mmean[:, iz], gamma[:, iz], gamma_err[:, iz])
                ax.plot(Mmean[:, iz],
                         gam_fun(gam_pars, Mmean[:, iz], zmean[:, iz]))
                ax.set_ylim(1, 2.5)
    plt.show()
   
    # Mz = np.ma.vstack((Mmean[ok].flatten(), zmean[ok].flatten())).T
    # r0_gp = SymbolicRegressor(population_size=5000,
    #                            generations=50, stopping_criteria=0.01,
    #                            const_range=(-100, 100),
    #                            p_crossover=0.7, p_subtree_mutation=0.1,
    #                            p_hoist_mutation=0.05, p_point_mutation=0.1,
    #                            max_samples=0.9, verbose=1,
    #                            parsimony_coefficient=0.001, random_state=0)
    # r0_gp.fit(Mz, r0[ok], sample_weight=r0_err[ok]**-2)
    # print('r0:', r0_gp._program)
    
    # gam_gp = SymbolicRegressor(population_size=5000,
    #                            generations=50, stopping_criteria=0.01,
    #                            const_range=(-100, 100),
    #                            p_crossover=0.7, p_subtree_mutation=0.1,
    #                            p_hoist_mutation=0.05, p_point_mutation=0.1,
    #                            max_samples=0.9, verbose=1,
    #                            parsimony_coefficient=0.001, random_state=0)
    # gam_gp.fit(Mz, gamma[ok], sample_weight=gamma_err[ok]**-2)
    # print('gamma:', gam_gp._program)
    
    # fig, axes = plt.subplots(2, nz, sharex=True, sharey='row', num=4)
    # fig.set_size_inches(16, 8)
    # fig.subplots_adjust(hspace=0, wspace=0)
    # axes[0, 0].set_ylabel(r'$r_0$')
    # axes[1, 0].set_ylabel(r'$\gamma$')
    # axes[1, 2].set_xlabel(r'$M_r$')
    # for iy in range(2):
    #     for iz in range(nz):
    #         # pdb.set_trace()
    #         Mz = np.ma.vstack((Mmean[:, iz], zmean[:, iz])).T
    #         ax = axes[iy, iz]
    #         if iy == 0:
    #             ax.text(0.5, 0.9, rf"z = [{zlo[iz]:3.1f}, {zhi[iz]:3.1f}]",
    #             transform=ax.transAxes)
    #             ax.errorbar(Mmean[:, iz], r0[:, iz], r0_err[:, iz])
    #             ax.plot(Mmean[:, iz], r0_gp.predict(Mz))
    #         else:
    #             ax.errorbar(Mmean[:, iz], gamma[:, iz], gamma_err[:, iz])
    #             ax.plot(Mmean[:, iz], gam_gp.predict(Mz))
    # plt.show()


def xir_mag_plot(nm=6, fit_range=[0.1, 20], p0=[5, 1.7],
                 xiscale=0, outfile='xi_z_mag.pkl'):
    """xi(r) from pair counts in apparent magnitude bins from treecorr."""

    prefix = f'xir_mag_{band}/'
    mlo, mhi, mmean = np.zeros(nm), np.zeros(nm), np.zeros(nm)
    r0, gamma = np.zeros(nm), np.zeros(nm)
    r0_err, gamma_err = np.zeros(nm), np.zeros(nm)
    plt.ioff()
    plt.clf()
    ax = plt.subplot(111)
    
    if xiscale:
        ax.set_ylabel(r'$r^2 \xi(r)$')
        ax.set_ylabel(r'$r^2 \xi(r)$')
    else:
        ax.set_ylabel(r'$\xi(r)$')
        ax.set_ylabel(r'$\xi(r)$')
    ax.set_xlabel(r'$r$ [Mpc/h]')
    for im in range(nm):
        color = next(ax._get_lines.prop_cycler)['color']
        infile = f'{prefix}xir_im{im}.fits'
        corr = wcorr.Corr1d(infile)
        mlo[im], mhi[im] = corr.meta['MLO'], corr.meta['MHI']
        if xiscale:
            popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                    plot_scale=corr.sep**2)
            ax.errorbar(corr.sep, corr.sep**2*corr.est_corr(),
                        corr.sep**2*corr.err, color=color, fmt='o',
                        label=rf"$m_{band} = [{mlo[im]:3.1f}, {mhi[im]:3.1f}]$")
        else:
            popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
            ax.errorbar(corr.sep, corr.est_corr(),
                        corr.err, color=color, fmt='o',
                        label=rf"$m_{band} = [{mlo[im]:3.1f}, {mhi[im]:3.1f}]$")
        mmean[im] = corr.meta['M_app_mean']
        r0[im], gamma[im] = popt
        r0_err[im], gamma_err[im] = pcov[0, 0]**0.5, pcov[1, 1]**0.5
        print(popt)
    pickle.dump((mlo, mhi, r0, gamma), open(outfile, 'wb'))
    plt.loglog()
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True, sharey='row', num=2)
    fig.set_size_inches(4, 8)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].set_ylabel(r'$r_0$')
    axes[0].errorbar(mmean, r0, r0_err)
    axes[1].set_ylabel(r'$\gamma$')
    axes[1].errorbar(mmean, gamma, gamma_err)
    axes[1].set_xlabel(r'$m_z$')
    plt.show()


def xir_mag_z_plot(nz=10, nm=6, fit_range=[1, 50], p0=[5, 1.7], xiscale=0,
                   rmin=1, rmax=50, weight=0):
    """xi(r) from pair counts in redshift and redshift-mag bins from treecorr."""

    prefix = 'xir_app_mag_z_z/'
    zmean, xi_int = np.zeros(nz), np.zeros(nz)
    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(8, 6)
    ax = axes[0]
    if xiscale:
        ax.set_ylabel(r'$r^2 \xi(r)$')
    else:
        ax.set_ylabel(r'$\xi(r)$')
    ax.set_xlabel(r'$r$ [Mpc/h]')
    for iz in range(nz):
        color = next(ax._get_lines.prop_cycler)['color']
        infile = f'{prefix}xir_z{iz}.fits'
        corr = wcorr.Corr1d(infile)
        zlo, zhi, zmean[iz] = corr.meta['ZLO'], corr.meta['ZHI'], corr.meta['ZMEAN']
        if xiscale:
            popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                    plot_scale=corr.sep**2)
            ax.errorbar(corr.sep, corr.sep**2*corr.est_corr(),
                        corr.sep**2*corr.err, color=color, fmt='o',
                        label=rf"$z = [{zlo:3.1f}, {zhi:3.1f}]$")
        else:
            popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
            ax.errorbar(corr.sep, corr.est_corr(),
                        corr.err, color=color, fmt='o',
                        label=rf"$z = [{zlo:3.1f}, {zhi:3.1f}]$")
            xi_int[iz] = wcorr.xi_power_law_integral(*popt, weight, rmin, rmax)
        print(popt)
    ax.loglog()
    ax.legend()

    ax = axes[1]
    ax.plot(zmean, xi_int)
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$b^2$')
    plt.show()

    fig, axes = plt.subplots(nm, 2, sharex='col')
    fig.set_size_inches(6, 7)
    fig.subplots_adjust(hspace=0)

    if xiscale:
        axes[nm//2, 0].set_ylabel(r'$r^2 \xi(r)$')
    else:
        axes[nm//2, 0].set_ylabel(r'$\xi(r)$')
    axes[-1, 0].set_xlabel(r'$r$ [Mpc/h]')
    axes[-1, 1].set_xlabel(r'$z$')
    axes[nm//2, 1].set_ylabel(r'$b^2$')
    for im in range(nm):
        zmean = []
        xi_int = []
        for iz in range(nz):
            ax = axes[im, 0]
            color = next(ax._get_lines.prop_cycler)['color']
            infile = f'{prefix}xir_iz{iz}_im{im}.fits'
            try:
                corr = wcorr.Corr1d(infile)
                zlo, zhi = corr.meta['ZLO'], corr.meta['ZHI']
                zmean.append(corr.meta['ZMEAN'])
                if xiscale:
                    popt, pcov = corr.fit_xi(fit_range, p0, ax, color,
                                            plot_scale=corr.sep**2)
                    ax.errorbar(corr.sep, corr.sep**2*corr.est_corr(),
                                corr.sep**2*corr.err, color=color, fmt='o',
                                label=rf"$z = [{zlo:3.1f}, {zhi:3.1f}]$")
                else:
                    popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
                    ax.errorbar(corr.sep, corr.est_corr(),
                                corr.err, color=color, fmt='o',
                                label=rf"$z = [{zlo:3.1f}, {zhi:3.1f}]$")
                    xi_int.append(wcorr.xi_power_law_integral(*popt, weight, rmin, rmax))
                print(popt)
            except FileNotFoundError:
                pass
        ax.loglog()
        # ax.legend()

        ax = axes[im, 1]
        ax.plot(zmean, xi_int)
    plt.show()


def kcorr(nplot=100000):
    """Empirically determine flagship K-corrections using k = m - M - DM."""

    t = Table.read(infile)
    sel = t['true_redshift_gal'] < 2
    t = t[sel]

    m, M, z = t[band+'mag'], t[band+'abs'], t['true_redshift_gal']
    dm = cosmo.distmod(z)
    k = m - M - dm
    # kc = calc_kcor.calc_kcor('r', z, 'g - r', g-r)
    sel = np.isfinite(k)
    z, k, M = z[sel], k[sel], M[sel]
    
    # Polynomial fit to K(z), constrained to pass through origin
    zp = np.linspace(0, 2, 41)
    plt.clf()
    ip = rng.choice(len(z), nplot)

    plt.scatter(z[ip], k[ip], s=0.01, c=M[ip])
    # for deg in [[1,], [1,2], [1,2,3], [1,2,3,4]]:
    #     p = Polynomial.fit(z, k, deg=deg, domain=[0, 1], window=[0, 1])
    #     kfit = p(zp)
    #     plt.plot(zp, kfit, label=str(deg))
    #     print(p.coef)
    deg = [1,2]
    p = Polynomial.fit(z, k, deg=deg, domain=[0, 2], window=[0, 2])
    kfit = p(zp)
    plt.plot(zp, kfit, label='All')
    print(p.coef)
    for im in range(len(Mbins)-1):
        sel = (Mbins[im] <= M) * (M < Mbins[im+1])
        p = Polynomial.fit(z[sel], k[sel], deg=deg, domain=[0, 2], window=[0, 2])
        kfit = p(zp)
        plt.plot(zp, kfit, label=f'[{Mbins[im]},  {Mbins[im+1]}]')
        print(p.coef)
    plt.colorbar(label=rf'$M_{band}$')
    plt.ylim(-1, 1)
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('Kcorr')
    # plt.subplot(212)
    # plt.scatter(z, k-kc, c=g-r, s=0.01)
    # # plt.scatter(z, k, s=0.01)
    # plt.colorbar(label='g-r')
    # plt.xlabel('Redshift')
    # plt.ylabel('Delta Kcorr')
    plt.show()


def zfun_lin(z, p):
    return p[0] + p[1]*z
    
def zfun_log(z, p):
    return p[0] + p[1]*np.log10(1+z)
    
def lf(zbins=np.linspace(0.0, 2.0, 11), p0=[-1.45, -21, -3],
       bounds=([-1.451, -23, 1e-4], [-1.449, -19, 1e-2]), zfit=0, beta=1,
       outfile=lf_file, mag_offset=0):
    """Flagship LF in redshift slices."""

    def Schechter(M, alpha, Mstar, lgphistar):
        L = 10**(0.4*(Mstar-M))
        schec = 0.4*ln10*10**lgphistar*L**(alpha+1)*np.exp(-L**beta)
        return schec

    # Lookup table of DM + kcorr for Vmax calculation
    dmk = cosmo.distmod(cosmo._z) + kpoly(cosmo._z)
    
    t = Table.read(infile)
    t = t[t[band+'mag'] + mag_offset < mlim]
    Mcen = Mbins_lf[:-1] + np.diff(Mbins_lf)
    area_frac = solid_angle/(4*math.pi)
    nm = len(Mbins_lf)-1
    nz = len(zbins)-1
    zmean = np.zeros(nz)
    alpha, Mstar, lgphistar = np.zeros(nz), np.zeros(nz), np.zeros(nz)
    alpha_err, Mstar_err, lgphistar_err = 1e10*np.ones(nz), 1e10*np.ones(nz), 1e10*np.ones(nz)

    lgphi = np.zeros((nm, nz))
    plt.ioff()
    fig = plt.figure(1)
    ax = fig.subplots()
    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        Vzlo = cosmo.comoving_volume(zlo)
        sel = (zlo <= t['true_redshift_gal']) * (t['true_redshift_gal'] < zhi)
        m, z = t[band+'mag'][sel] + mag_offset, t['true_redshift_gal'][sel]
        dm = cosmo.distmod(z)
        k = kpoly(z)
        M = m - dm - k
        zlim = np.clip(np.interp(mlim - M, dmk, cosmo._z), zlo, zhi)
        zmean[iz] = np.mean(z)
        V = area_frac * (cosmo.comoving_volume(z) - Vzlo)
        Vmax = area_frac * (cosmo.comoving_volume(zlim) - Vzlo)
        Vsel = Vmax > 0
        VVm = (V/Vmax)[Vsel]
        print(zlo, zhi, np.mean(VVm), np.std(VVm))

        # Find abs mag completeness limit at zlo (Loveday+2012 eqn 18)
        Mfaint = mlim - cosmo.distmod(zlo) - kpoly(zlo)

        N, edges = np.histogram(M, Mbins_lf)
        phi, edges = np.histogram(M, Mbins_lf, weights=1/Vmax)
        phi /= np.diff(Mbins_lf)
        phi_err = phi/N**0.5
        use = (N > 0) * (Mbins_lf[1:] < Mfaint)
        # Marr = np.append(Marr, Mcen[use])
        # zarr = np.append(zarr, zmean[iz]*np.ones(len(Mcen[use])))
        # lgphiarr = np.append(lgphiarr, np.log10(phi[use]))
        if len(Mcen[use]) > 2:
        
            # Interpolate lg phi(M) with extrapolation
            lgphi_int = interp1d(Mcen[use], np.log10(phi[use]),
                             fill_value='extrapolate')
            lgphi[:, iz] = lgphi_int(Mcen)
                        
            # lf_interp.update({iz: (zmean[iz],
            #                        interp1d(Mcen[use], np.log10(phi[use]),
            #                                 fill_value='extrapolate'))})
            color = next(ax._get_lines.prop_cycler)['color']
            popt, pcov = scipy.optimize.curve_fit(
                Schechter, Mcen[use], phi[use], p0=p0, sigma=phi_err[use],
                xtol=1e-3)
            p0 = popt
            alpha[iz], Mstar[iz], lgphistar[iz] = popt[0], popt[1], popt[2]
            alpha_err[iz], Mstar_err[iz], lgphistar_err[iz] = pcov[0][0]**0.5, pcov[1][1]**0.5, pcov[2][2]**0.5, 
            lbl = f'z = {zlo:3.1f} - {zhi:3.1f}; {popt[0]:3.2f} {popt[1]:3.2f} {popt[2]:3.2e}'
            ax.plot(Mcen, Schechter(Mcen, *popt), color=color)
        else:
            lbl = f'z = {zlo:3.1f} - {zhi:3.1f}'
        ax.errorbar(Mcen[use], phi[use], phi_err[use], fmt='o', color=color,
                    label=lbl)
        ax.plot(Mcen, 10**lgphi[:, iz], '--', color=color)

    # Plot LF extrapolated to z=0 as sanity check
    lf0 = 10**interpn((Mcen, zmean), lgphi,
                      np.array((Mbins_lf, np.zeros(len(Mbins_lf)))).T,
                      bounds_error=False, fill_value=None)
    ax.plot(Mbins_lf, lf0, '--', color='k', label='z = 0')

    ax.semilogy(base=10)
    ax.set_ylim(1e-7, 1e1)
    ax.set_xlabel(r'$M_h$')
    ax.set_ylabel(r'$\Phi(M_h)$')
    ax.legend()
    plt.show()

    
    # Schechter parameters as function of z (zfit=0) or lg(1+z) if zfit=1
    zp = np.array([zbins[0], zbins[-1]])
    zlbl = 'z'
    zfun = zfun_lin
    if zfit:
        zmean = np.log10(1+zmean)
        zp = np.log10(1+zp)
        zlbl = 'lg(1+z)'
        zfun = zfun_log
    fig, axes = plt.subplots(3, 1, sharex=True, num=3)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    lf_dict = {'zfun': zfun}
    
    ax = axes[0]
    use = alpha_err < 1e9
    ax.errorbar(zmean[use], alpha[use], alpha_err[use])
    p = Polynomial.fit(zmean[use], alpha[use], deg=1, w=alpha_err[use]**-2)
    lf_dict.update({'alpha': [p.coef[0], p.coef[1]]})
    yp = p(zp)
    ax.plot(zp, yp)
    ax.text(0.4, 0.85, rf'$\alpha = {p.coef[0]:5.3f} + {p.coef[1]:5.3f} {zlbl}$',
            transform=ax.transAxes)
    ax.set_ylabel(r'$\alpha$')

    ax = axes[1]
    use = Mstar_err < 1e9
    ax.errorbar(zmean[use], Mstar[use], Mstar_err[use])
    p = Polynomial.fit(zmean[use], Mstar[use], deg=1, w=Mstar_err[use]**-2)
    lf_dict.update({'Mstar': [p.coef[0], p.coef[1]]})
    yp = p(zp)
    ax.plot(zp, yp)
    ax.text(0.4, 0.85, rf'$M^* = {p.coef[0]:5.3f} + {p.coef[1]:5.3f} {zlbl}$',
            transform=ax.transAxes)
    ax.set_ylabel(r'$M^*$')
    
    ax = axes[2]
    use = lgphistar_err < 1e9
    ax.errorbar(zmean[use], lgphistar[use], lgphistar_err[use])
    p = Polynomial.fit(zmean[use], lgphistar[use], deg=1,
                       w=lgphistar_err[use]**-2)
    lf_dict.update({'lgphistar': [p.coef[0], p.coef[1]]})
    yp = p(zp)
    ax.plot(zp, yp)
    ax.text(0.4, 0.85, rf'$\lg \phi^* = {p.coef[0]:3.2e} + {p.coef[1]:3.2e} {zlbl}$',
            transform=ax.transAxes)
    ax.set_ylabel(r'$lg \Phi^*$')
    ax.set_xlabel(zlbl)
    plt.show()

    pickle.dump((kpoly, lf_dict, Mcen, zmean, lgphi), open(outfile, 'wb'))

def gen_selfn(infile='14516.fits', outfile='sel_14516.pkl',
              M_bins=np.linspace(-24, -16, 5),
              zbins=np.linspace(0, 1, 6), nksamp=100):
    """Generate selection functions in luminosity bins."""

    # K-correction coefficients for mag bins defined by M_bins
    kcoeffs = [[0, 1.70, 0.64], [0, 0.77, 0.94], [0, 0.65, 0.08], [0, -0.26, 0.53]]

    t = Table.read(infile)
    sel = t['true_redshift_gal'] < 1
    t = t[sel]

    m, M, z = t['hmag'], t['habs'], t['true_redshift_gal']
    dm = cosmo.distmod(z)
    k = m - M - dm
    ktable = Table([M, z, k], names=('M', 'z', 'k'))
    # kc = calc_kcor.calc_kcor('r', z, 'g - r', g-r)
    sel = np.isfinite(k)
    ktable = ktable[sel]

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    ax = axes[0]
    ax.set_ylabel('S(z)')
    ax = axes[1]
    ax.set_xlabel('z')
    ax.set_ylabel('N(z)')
    sel_dict = {}
    for im in range(len(M_bins) - 1):
        Mmin, Mmax = M_bins[im], M_bins[im+1]
        sel = (Mmin <= ktable['M']) * (ktable['M'] < Mmax)
        selfn = util.SelectionFunction(
            cosmo, alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
            phistar=[9.81e-4, -3.24e-4], mmin=0, mmax=25,
            Mmin=Mmin, Mmax=Mmax, ktable=ktable[sel], nksamp=nksamp,
            kcoeffs=kcoeffs[im], solid_angle=solid_angle,
            sax=axes[0], nax=axes[1])
        key = f'{Mmin:4.1f}-{Mmax:4.1f}'
        sel_dict[key] = selfn
    pickle.dump(sel_dict, open(outfile, 'wb'))
    axes[0].legend()
    axes[1].legend()
    plt.show()


def selfn_plot(infile='sel_14516.pkl'):
    """plot selection functions in luminosity bins."""

    sel_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    fig, axes = plt.subplots(5, 1, sharex=True, num=1)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].set_ylabel(r'$z - \langle \Delta z \rangle$')
    axes[1].set_ylabel(r'$\langle M \rangle$')
    axes[2].set_ylabel(r'$\langle k \rangle$')
    axes[3].set_ylabel('S(z)')
    axes[4].set_ylabel('N(z)')
    axes[4].set_xlabel('z')
    for key in sel_dict.keys():
        selfn = sel_dict[key]
        axes[0].plot(selfn._z, selfn._z - selfn._zmean, label=key)
        axes[1].plot(selfn._z, selfn._Mmean, label=key)
        axes[2].plot(selfn._z, selfn._kmean, label=key)
        axes[3].plot(selfn._z, selfn._sel, label=key)
        axes[4].plot(selfn._z, selfn._Nz, label=key)
    plt.legend()
    plt.show()


def dist_plot():
    """Proper and comoving distance vs redshift."""
    plt.clf()
    plt.plot(cosmo._z, cosmo._x, label='Comoving')
    plt.plot(cosmo._z, cosmo._x/(1 + cosmo._z), label='Proper')
    r = np.linspace(0.0, 2000, 200)
    z = cosmo.z_at_pdist(r)
    plt.plot(z, r, label='z(r)')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('Distance [Mpc]')
    plt.show()
