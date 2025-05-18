# Clustering measurements for WAVES target catalogues using treecorr

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
import treecorr
import pdb
from pycorr import TwoPointCorrelationFunction
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
south_limits = [330, 51.6, -35.6, -27]
solid_angle_north = ((north_limits[1]-north_limits[0]) *
                     (north_limits[3]-north_limits[2]) * (math.pi/180)**2)

def make_rect_mask_N():
    """Make simple rectangular mangle mask."""
    wcorr.make_rect_mask(limits=north_limits, rect_mask='mask_N.ply')


def make_rect_mask_S():
    """Make simple rectangular mangle mask."""
    wcorr.make_rect_mask(limits=south_limits, rect_mask='mask_S.ply')


def plot_pix_mask(mask='WAVES-N_pixelMask.fits'):
    """Plot pixel mask."""

    with fits.open(mask) as hdus:
        hdu = hdus[1]
        wcs = WCS(hdu.header)
        data = hdu.data
    print(data.shape)

    values, counts = np.unique(data, return_counts=True)
    print(values, counts)

    plt.subplot(projection=wcs)
    plt.imshow(data, origin='lower')
    plt.grid(color='white', ls='solid')
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.show()


def plot_cat_mask(mask='WAVES-N_pixelMask.fits',
                  galcat='WAVES-N_1p2_Z21.2_unmasked_ultralight.parquet',
                  rancat='randoms_N.fits', mlim=20):
    """Plot catalogues on top of pixel mask."""

    with fits.open(mask) as hdus:
        hdu = hdus[1]
        wcs = WCS(hdu.header)
        data = hdu.data
    print(data.shape)

    ax = plt.subplot(projection=wcs)
    plt.imshow(data, origin='lower', cmap='Greys', vmin=0, vmax=1)
    plt.grid(color='white', ls='solid')
    plt.xlabel('RA')
    plt.ylabel('Dec')
 
    t = Table.read(galcat)
    ra, dec, mag = t['RAmax'], t['Decmax'], t['mag_Zt']
    sel = mag < mlim
    print(len(ra[sel]), 'out of', len(ra), 'galaxies')
    ra, dec = ra[sel], dec[sel]
    plt.scatter(ra, dec, s=0.1, color='red', transform=ax.get_transform('world'))
 
    t = Table.read(rancat)
    ra, dec = t['RA'], t['DEC']
    print(len(ra), 'randoms')
    plt.scatter(ra, dec, s=0.1, color='yellow', transform=ax.get_transform('world'))
    plt.show()


def check_mask(mask='WAVES-S_pixelMask.fits'):
    """Return mask value for given coordinate."""

    hdu = fits.open(mask)[1]
    wcs = WCS(hdu.header)
    print(hdu.header, hdu.data.shape)
    ans = 'xxx'
    while ans != '':
        ans = input('RA, Dec [deg]: ')
        if ans != '':
            coords = SkyCoord(*[float(c) for c in ans.split(',')], frame='icrs', unit='deg')
            x, y = wcs.world_to_pixel(coords)
            try:
                print(coords, x, y, hdu.data[int(y), int(x)])
            except IndexError:
                print(coords, x, y, 'Outside mask')


def mangle_test(mask='mask_S.ply', nran=1000, limits=south_limits):
    pymask = pymangle.Mangle(mask)
    ra, dec = pymask.genrand_range(nran, *limits)
    # ra, dec = pymask.genrand(nran)
    plt.scatter(ra, dec, s=0.1)
    plt.show()


def wcounts_class():
    """Angular pair counts for various source classifications."""
    root = '/research/astro/gama/bb345/1-4MOST/1-Data/1-TC_star_gal/Nov23/waves_wide_TC_Nov23f'
    magbins=np.linspace(16, 21.2, 2)
    for field in (1, 2):
        if field == 1:
            NS = 'N'
            limits=north_limits
        if field == 2:
            NS = 'S'
            limits=south_limits
        for TC_class in ('gal', 'star'):
            for BC_class in ('amb', 'gal', 'star'):
                infile = root+f'{field}_TC_{TC_class}_B_{BC_class}.fits'
                out_dir = f'wmag_{NS}_TC_{TC_class}_B_{BC_class}/'
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                wcounts(infile=infile, 
                        mask_file=f'mask_{NS}.ply',
                        pixel_mask=f'WAVES-{NS}_pixelMask.fits',
                        out_pref=out_dir,
                        limits=limits, magbins=magbins)

    
def trim_cat_N():
    trim_cat(22, north_limits, 'WAVES-N_pixelMask.fits',
             'WAVES-N_0p2_Z22_withMasking.parquet', 'WAVES_N.fits')

def trim_cat_S():
    trim_cat(22, south_limits, 'WAVES-S_pixelMask.fits',
             'WAVES-S_0p2_Z22_withMasking.parquet', 'WAVES_S.fits')

def trim_cat(mlim, limits, pixel_mask, infile, outfile):
    """Trim catalogue to limits, mlim, apply mask, and output only galaxy class.
    Not needed with v1p2."""

    def mask(ra, dec):
        """Returns mask value for each coordinate or -1 if outside mask."""
        
        coords = SkyCoord(ra.astype(np.float64), dec.astype(np.float64),
                          frame='icrs', unit='deg')
        x, y = wcs.world_to_pixel(coords)
        ix, iy = x.astype(int), y.astype(int)
        inside = (ix >= 0) * (ix < pixmask.shape[1]) * (iy >= 0) * (iy < pixmask.shape[0])
        mask = np.zeros(len(ix), dtype=int)
        mask[inside] = pixmask[iy[inside], ix[inside]]
        mask[~inside] = -1
        return mask
    
    with fits.open(pixel_mask) as hdus:
        hdu = hdus[1]
        wcs = WCS(hdu.header)
        pixmask = hdu.data
    print(pixmask.shape)

    t = Table.read(infile)
    ra, dec, mag, sg = t['RAmax'], t['Decmax'], t['mag_Zt'], t['class']
    print(len(t), 'objects read')

    # See https://stackoverflow.com/questions/66799475/how-to-elegantly-find-if-an-angle-is-between-a-range
    sel = ((mag < mlim) * (sg == 'galaxy') *
           (((ra - limits[0]) % 360 <= (limits[1] - limits[0]) % 360) *
            (limits[2] <= dec) * (dec < limits[3]) * (mask(ra, dec) == 0)))
        
    print(len(ra[sel]), 'trimmed and masked galaxies')
    t = Table((ra[sel].astype(np.float64), dec[sel].astype(np.float64), mag[sel]), names=('RA', 'DEC', 'MAG_Z'))
    t.write(outfile, overwrite=True)

def gen_rand_N():
    """Randoms for WAVES Wide-N."""
    gen_rand(limits=north_limits, mask_file='mask_N.ply',
            pixel_mask='WAVES-N_pixelMask.fits', outfile='randoms_N.fits')

    
def gen_rand_S():
    """Randoms for WAVES Wide-S."""
    gen_rand(limits=south_limits, mask_file='mask_S.ply',
            pixel_mask='WAVES-S_pixelMask.fits', outfile='randoms_S.fits')

    
def gen_rand(limits, mask_file, pixel_mask, outfile, nran=1000000):
    """Generate random points within limits and mask."""

    def mask(ra, dec):
        """Returns mask value for each coordinate or -1 if outside mask."""
        
        coords = SkyCoord(ra.astype(np.float64), dec.astype(np.float64),
                          frame='icrs', unit='deg')
        x, y = wcs.world_to_pixel(coords)
        ix, iy = x.astype(int), y.astype(int)
        inside = (ix >= 0) * (ix < pixmask.shape[1]) * (iy >= 0) * (iy < pixmask.shape[0])
        mask = np.zeros(len(ix), dtype=int)
        mask[inside] = pixmask[iy[inside], ix[inside]]
        mask[~inside] = -1
        return mask
    
    with fits.open(pixel_mask) as hdus:
        hdu = hdus[1]
        wcs = WCS(hdu.header)
        pixmask = hdu.data
    print(pixmask.shape)

    pymask = pymangle.Mangle(mask_file)
    # genrand_range does not work if limits wrap zero
    # ra, dec = pymask.genrand_range(nran, *limits)
    ra, dec = pymask.genrand(nran)
    sel = ((((ra - limits[0]) % 360 <= (limits[1] - limits[0]) % 360) *
            (limits[2] <= dec) * (dec < limits[3]) * (mask(ra, dec) == 0)))
    print(len(ra[sel]), 'out of', nran, 'unmasked randoms')

    t = Table((ra[sel].astype(np.float64), dec[sel].astype(np.float64)), names=('RA', 'DEC'))
    t.write(outfile, overwrite=True)

def wcounts_N():
    """Angular pair counts in mag bins."""
    wcounts(galfile='WAVES-N_1p2_Z22_unmasked_ToddClass.fits',
            ranfile='randoms_N.fits', out_dir='wmag_N')

    
def wcounts_S():
    """Angular pair counts in mag bins."""
    wcounts(galfile='WAVES-S_1p2_Z22_unmasked_ToddClass.fits',
            ranfile='randoms_S.fits', out_dir='wmag_S')

    
def wcounts(galfile, ranfile, out_dir,
            npatch=9, tmin=0.01, tmax=10, nbins=20,
            magbins=[16, 17, 18, 19, 20, 21, 22], plot=0,
            ra_col='RAmax', dec_col='Decmax', mag_col='mag_Zt'):
    """Angular pair counts in mag bins."""

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
    t = Table.read(ranfile)
    print(len(t), 'randoms read')
    rcat = treecorr.Catalog(ra=t['RA'], dec=t['DEC'],
                            ra_units='deg', dec_units='deg',
                            npatch=npatch)
    rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees')
    rr.process(rcat)

    # Now the galaxies
    t = Table.read(galfile)
    sel = t['cluster_label'] == 'galaxy'
    t = t[sel]
    ra, dec, mag = t[ra_col], t[dec_col], t[mag_col]
    print(len(t), 'galaxies read')
    ra_shift = 'S' in out_dir

    print('imag  ngal')
    for imag in range(len(magbins) - 1):
        mlo, mhi = magbins[imag], magbins[imag+1]
        sel = (mlo <= mag) * (mag < mhi)
        print(imag, len(ra[sel]))
        gcat = treecorr.Catalog(ra=ra[sel], dec=dec[sel],
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
        outfile = f'{out_dir}/w_m{imag}.fits'
        dd.write(outfile, rr=rr, dr=dr)
        with fits.open(outfile, mode='update') as hdul:
            hdr = hdul[1].header
            hdr['mlo'] = mlo
            hdr['mhi'] = mhi
            hdr['Ngal'] = gcat.nobj
            hdr['Nran'] = rcat.nobj
            hdul.append(fits.PrimaryHDU(xi_jack))
            hdul.flush()


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


def w_plot_class(nmag=1, njack=10, fit_range=[0.01, 5], p0=[0.05, 1.7],
           avgcounts=False):
    """w(theta) for different source classes"""

    # for (num, NS) in zip((1, 2), 'NS'):
    for (num, NS) in zip((1, ), 'N'):
        plt.clf()
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, num=num)
        fig.set_size_inches(8, 6)
        fig.subplots_adjust(hspace=0, wspace=0)
        axes[0, 0].set_ylabel(r'$w(\theta)$')
        axes[1, 0].set_ylabel(r'$w(\theta)$')
        axes[1, 1].set_xlabel(r'$\theta$ [deg]')
        for (row, TC_class) in zip((0, 1), ('gal', 'star')):
            for (col, BC_class) in zip((0, 1, 2), ('gal', 'star', 'amb')):
                ax = axes[row, col]
                prefix = f'wmag_{NS}_TC_{TC_class}_B_{BC_class}/'
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
                            wcorr.Corr1d(info['Ngal'], info['Ngal'],
                                         info['Nran'], info['Nran'],
                                         DD_counts, DR_counts, DR_counts, RR_counts,
                                         mlo=info['mlo'], mhi=info['mhi']))
                    corr = corrs[0]
                    corr.err = (njack-1)*np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
                    corr.ic_calc(fit_range, p0, 5)
                    color = next(ax._get_lines.prop_cycler)['color']
                    corr.plot(ax, color=color, label=f"m = [{info['mlo']}, {info['mhi']}]")
                    popt, pcov = corr.fit_w(fit_range, p0, ax, color)
                    print(popt, pcov)
                    ax.text(0.9, 0.9, prefix[:-1], transform=ax.transAxes, ha='right')
                    ax.text(0.9, 0.8,
                            fr'$A = {popt[0]:3.2e}, \gamma = {popt[1]:3.2f}$',
                            transform=ax.transAxes, ha='right')
        # plt.semilogx()
        plt.loglog()
        plt.ylim(1e-3, 2)
        # plt.legend()
    plt.show()


def w_plot(nmag=6, fit_range=[0.01, 5], p0=[0.05, 1.7],
           prefix='wmag_N/',
        #    Nz_file='/Users/loveday/Data/Legacy/corr/cmass_ngc_Legacy_9/Nz.pkl',
           Nz_file='/Users/loveday/Data/flagship/Nz_z.pkl',
           xi_pars='/Users/loveday/Data/flagship/xi_z_mag.pkl'):
    """w(theta) from angular pair counts in mag bins.
    Use observed N(z) if Nz_file specified, otherwise use LF prediction."""

    if Nz_file:
        # (zmean, pmz, pmz_err, Nz_mlo, Nz_mhi, be_pars) = pickle.load(open(Nz_file, 'rb'))
        Nz_dict = pickle.load(open(Nz_file, 'rb'))
        (xi_mlo, xi_mhi, r0, gamma) = pickle.load(open(xi_pars, 'rb'))

    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for imag in range(nmag):
        infile = f'{prefix}w_m{imag}.fits'
        corr = wcorr.Corr1d(infile)
        mlo, mhi = corr.meta['MLO'], corr.meta['MHI']
        m = 0.5*(mlo + mhi)
        corr.plot(ax, label=f"m = [{mlo}, {mhi}]")
        clr = plt.gca().lines[-1].get_color()  # save colour for fit and prediction
        if fit_range:
            popt, pcov = corr.fit_w(fit_range, p0, ax, color=clr)
            print(popt, pcov)
        if Nz_file:
            if ((mlo == Nz_dict['mbins'][imag] == xi_mlo[imag]) and 
                (mhi == Nz_dict['mbins'][imag+1] == xi_mhi[imag])):
                wlim = limber.w_lum_Nz_fit(cosmo, corr.sep, m, r0[imag],
                                           gamma[imag], Nz_dict['be_pars'][imag, :], # was imag-1
                                           plotint=0, pdf=None, plot_den=0)
                plt.plot(corr.sep, wlim, '--', color=clr)
                print(imag, r0[imag], gamma[imag], Nz_dict['be_pars'][imag, :])
            else:
                print('mismatched mag limits', mlo, Nz_dict['mbins'][imag],
                      mhi, Nz_dict['mbins'][imag+1])
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()

    # wcorr.wplot_scale(cosmo, corr_slices, gamma1=gamma1, gamma2=gamma2,
    #                   r0=r0, eps=eps, lf_pars='lf_pars.pkl', Nz_file=Nz_file)


def zfun_lin(z, p):
    return p[0] + p[1]*z
    
def lf(alpha=[-1.26, -0.2], Mstar=[-20.71, -1],
       lgphistar=[-2.02, -0.4], kcoeffs=[0.0, -0.39, 1.67],
       outfile='lf_pars.pkl'):
    """Save assumed LF parameters (taken from GAMA fits)."""

    kpoly = Polynomial(kcoeffs, domain=[0, 2], window=[0, 2])
    lf_dict = {'zfun': zfun_lin, 'alpha': alpha, 'Mstar': Mstar,
               'lgphistar': lgphistar}
    pickle.dump((kpoly, lf_dict, None, None, None), open(outfile, 'wb'))


def Nz(infile='WAVES-N_0p2_Z22_GalsAmbig_CompletePhotoZ.fits',
       solid_angle=solid_angle_north, magbins=np.linspace(15, 22, 8),
       zbins=np.linspace(0.0, 2.0, 41), lf_pars='lf_pars.pkl',
       interp=0, outfile='NzN.pkl'):
    """Plot observed and predicted N(z) histograms in mag slices."""

    def be_fit(z, zc, alpha, beta, norm):
        """Generalised Baugh & Efstathiou (1993, eqn 7) model for N(z)."""
        return norm * z**alpha * np.exp(-(z/zc)**beta)
    
    t = Table.read(infile)
    sel = t['mag_Zt'] < magbins[-1]
    t = t[sel]
    mag, z = t['mag_Zt'], t['z_best']
    zcen = zbins[:-1] + 0.5*np.diff(zbins)
    zmin, zmax = zbins[0], zbins[-1]
    zp = np.linspace(zmin, zmax, 500)
    Nz_dict = {'zbins': zbins, 'zcen': zcen}
    plt.clf()
    ax = plt.subplot(111)
    for imag in range(len(magbins) - 1):
        mlo, mhi = magbins[imag], magbins[imag+1]
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        color = next(ax._get_lines.prop_cycler)['color']
        counts, edges = np.histogram(z[sel], zbins)
        popt, pcov = scipy.optimize.curve_fit(
            be_fit, zcen, counts, p0=(0.5, 2.0, 1.5, 1e6), ftol=1e-3, xtol=1e-3)
        print(popt)

        Nz_dict.update({imag: (mlo, mhi, counts, popt)})
        plt.stairs(counts, edges, color=color, label=f"m = {mlo}, {mhi}]")
        # plt.plot(zp, spline(zp), color=color, ls='-')
        plt.plot(zp, be_fit(zp, *popt), color=color, ls='-')
        selfn = st_util.SelectionFunction(
            cosmo, lf_pars=lf_pars, 
            mlo=mlo, mhi=mhi, solid_angle=solid_angle,
            dz=zbins[1]-zbins[0], interp=interp)
        selfn.plot_Nz(ax, color=color, ls='--')

    pickle.dump(Nz_dict, open(outfile, 'wb'))
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('N(z)')
    plt.show()


def zmag_comp():
    """Compare SDSS (GAMAII) and VISTA (GAMAIII) z-band magnitudes."""

    tmatch = Table.read('/Users/loveday/Data/gama/DR4/gkvGamaIIMatchesv01.fits')
    tgkv = Table.read('/Users/loveday/Data/gama/DR4/gkvScienceCatv02.fits')
    teq = Table.read('/Users/loveday/Data/gama/TilingCatv46.fits')

    teqmatch = join(tmatch, teq, keys='CATAID')
    t  = join(teqmatch, tgkv, keys='uberID')
    z_vista = 8.9 - 2.5*np.log10(t['flux_Zt'])
    plt.clf()
    plt.scatter(t['Z_MODEL'], z_vista, s=0.1)
    plt.xlabel('SDSS z model mag')
    plt.ylabel('VISTA Z mag')
    plt.show()

# Routines for Shark mocks
def shark_xir(infile='waves_wide_gals.parquet', mask_file='../../v1p2/mask_N.ply',
              region='N', limits=north_limits, ranfac=1, npatch=9, rmin=0.1, rmax=100, nbins=16,
              magbins=[16, 17, 18, 19, 20, 21, 21.2]):
    """Real-space pair counts in apparent magnitude bins."""

    # Create out_dir if it doesn't already exist
    out_dir = f'xir_mag_{region}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nm = len(magbins) - 1
    magcol = 'total_ap_dust_Z_VISTA'
    mabscol = 'total_ab_dust_Z_VISTA'

    t = Table.read(infile)
    if region == 'N':
        sel = t['dec'] > -10
    if region == 'S':
        sel = t['dec'] < -10
    print(len(t[sel]), 'out of', len(t), 'galaxies selected')
    t = t[sel]
    ra, dec, mag, redshift = t['ra'], t['dec'], t[magcol], t['zcos']
    r = cosmo.dc(redshift)
    mask = pymangle.Mangle(mask_file)

    for im in range(nm):
        mlo, mhi = magbins[im], magbins[im+1]
        sel = (mlo <= mag) * (mag < mhi)
        mmean = np.mean(mag[sel])
        Mmean = np.mean(t[mabscol][sel])
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
            var_method='jackknife', cross_patch_weight='match')
        dd.process(galcat)
        dr = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins, cross_patch_weight='match')
        dr.process(galcat, rancat)
        rr = treecorr.NNCorrelation(
            min_sep=rmin, max_sep=rmax, nbins=nbins, cross_patch_weight='match')
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

def xir_mag_plot(nm=6, fit_range=[0.1, 20], p0=[5, 1.7],
                 xiscale=0, outfile='xi_z_mag.pkl'):
    """xi(r) from pair counts in apparent magnitude bins from treecorr."""

    prefix = 'xir_mag_N/'
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
        infile = f'{prefix}xir_im{im}.fits'
        corr = wcorr.Corr1d(infile)
        mlo[im], mhi[im] = corr.meta['MLO'], corr.meta['MHI']
        if xiscale:
            popt, pcov = corr.fit_xi(fit_range, p0, ax,
                                    plot_scale=corr.sep**2)
            clr = plt.gca().lines[-1].get_color()
            ax.errorbar(corr.sep, corr.sep**2*corr.est_corr(),
                        corr.sep**2*corr.err, color=clr, fmt='o',
                        label=rf"$m_{band} = [{mlo[im]:3.1f}, {mhi[im]:3.1f}]$")
        else:
            popt, pcov = corr.fit_xi(fit_range, p0, ax)
            clr = plt.gca().lines[-1].get_color()
            ax.errorbar(corr.sep, corr.est_corr(),
                        corr.err, color=clr, fmt='o',
                        label=rf"$m_z = [{mlo[im]:3.1f}, {mhi[im]:3.1f}], r_0={popt[0]:3.2f}, \gamma={popt[1]:3.2f}$")
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


def shark_xi_rp_pi(infile='waves_wide_gals.parquet', mask_file='../../v1p2/mask_N.ply',
              region='N', limits=north_limits, ranfac=1, npatch=9,
              edges = (np.linspace(0, 50, 51), np.linspace(-50, 50, 101))):
    """xi(rp, pi) calculated with pycorr."""

    outfile = f'xi_rp_pi_{region}.pkl'

    t = Table.read(infile)
    if region == 'N':
        sel = t['dec'] > -10
    if region == 'S':
        sel = t['dec'] < -10
    sel *= t['total_ap_dust_Z_VISTA'] < 20
    print(len(t[sel]), 'out of', len(t), 'galaxies selected')
    t = t[sel]
    ra, dec, redshift = t['ra'], t['dec'], t['zobs']
    r = cosmo.dc(redshift)
    galpos = np.vstack((ra, dec, r))

    mask = pymangle.Mangle(mask_file)
    ngal = len(ra)
    nran = int(ranfac*ngal)
    rar, decr = mask.genrand_range(nran, *limits)
    rr = rng.choice(r, nran, replace=True)
    ranpos = np.vstack((rar, decr, rr))
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=galpos,
                                     randoms_positions1=ranpos, position_type='rdd',
                                     engine='corrfunc', nthreads=4)
    pickle.dump(result, open(outfile, 'wb'))


def xi_rp_pi_plot(infile='xi_rp_pi_N.pkl', cmap=None, aspect='auto', prange=[-2, 2]):
    result = pickle.load(open(infile, 'rb'))
    extent = (-result.edges[0][-1], result.edges[0][-1], -result.edges[1][-1], result.edges[1][-1])
    logxi = np.log10(result.corr).T
    npi, nrp = logxi.shape[0], logxi.shape[1]
    map = np.zeros((npi, 2*nrp))
    map[:, nrp:] = logxi
    map[:, :nrp] = np.fliplr(logxi)

    plt.clf()
    ax = plt.subplot(111)
    im = ax.imshow(map, cmap, aspect=aspect, interpolation='none',
                   vmin=prange[0], vmax=prange[1],
                   extent=extent)
    ax.set_xlabel(r'$r_\perp\ [h^{-1} {{\rm Mpc}}]$')
    ax.set_ylabel(r'$r_\parallel\ [h^{-1} {{\rm Mpc}}]$')
    plt.show()
