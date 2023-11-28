# Clustering measurements for WAVES target catalogues

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
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
import Corrfunc
# import Corrfunc.mocks
# from Corrfunc.utils import convert_3d_counts_to_cf
import pdb
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
solid_angle_north = ((north_limits[1]-north_limits[0]) *
                     (north_limits[3]-north_limits[2]) * (math.pi/180)**2)

def make_rect_mask_N():
    """Make simple rectangular mangle mask."""
    wcorr.make_rect_mask(limits=north_limits, rect_mask='mask_N.ply')


def make_rect_mask_S():
    """Make simple rectangular mangle mask."""
    wcorr.make_rect_mask(limits=south_limits, rect_mask='mask_S.ply')


def plot_pix_mask(mask='WAVES-S_pixelMask.fits'):
    """Plot pixel mask."""

    hdu = fits.open(mask)[1]
    wcs = WCS(hdu.header)

    plt.subplot(projection=wcs)
    plt.imshow(hdu.data, origin='lower')
    plt.grid(color='white', ls='solid')
    plt.xlabel('RA')
    plt.ylabel('Dec')
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

    
def wcounts_N():
    """Angular pair counts in mag bins."""
    wcounts(infile='WAVES-N_0p2_Z22_GalsAmbig_CompletePhotoZ.fits',
            mask_file='mask_N.ply', pixel_mask='WAVES-N_pixelMask.fits',
            out_pref='wmag_N/', limits=north_limits)

    
def wcounts_S():
    """Angular pair counts in mag bins."""
    wcounts(infile='WAVES-S_small.fits',
            mask_file='mask_S.ply', pixel_mask='WAVES-S_pixelMask.fits',
            out_pref='wmag_S/', limits=south_limits)

    
def wcounts(infile, mask_file, pixel_mask, out_pref, limits,
            nran=100000, nra=10, ndec=1,
            tmin=0.01, tmax=10, nbins=20,
            magbins=np.linspace(15, 22, 8)):
    """Angular pair counts in mag bins."""

    def mask(ra, dec):
        """Returns mask value for each coordinate or -1 if outside mask."""
        
        coords = SkyCoord(ra.astype(np.float64), dec.astype(np.float64),
                          frame='icrs', unit='deg')
        x, y = wcs.world_to_pixel(coords)
        ix, iy = x.astype(int), y.astype(int)
        inside = (ix >= 0) * (ix < hdu.data.shape[1]) * (iy >= 0) * (iy < hdu.data.shape[0])
        mask = np.zeros(len(ix), dtype=int)
        mask[inside] = hdu.data[iy[inside], ix[inside]]
        mask[~inside] = -1
        return mask

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

    hdu = fits.open(pixel_mask)[1]
    wcs = WCS(hdu.header)
    print(hdu.data.shape)
    
    # Check for RA wraparound
    # ra_wrap = 0
    # if limits[0] > limits[1]:
    #     ra_wrap = 1
    #     limits[:2] = wcorr.ra_shift(limits[:2])
    #     print('RA limits changed to', limits[:2])
    #     ra = wcorr.ra_shift(ra)
    
    # See https://stackoverflow.com/questions/66799475/how-to-elegantly-find-if-an-angle-is-between-a-range
    sel = ((mag < magbins[-1]) *
           (((ra - limits[0]) % 360 <= (limits[1] - limits[0]) % 360) *
            (limits[2] <= dec) * (dec < limits[3]) * (mask(ra, dec) == 0)))
        
    t = t[sel]
    print(len(t), 'unmasked galaxies')
    
    ra, dec, mag = t[ra_col], t[dec_col], t[mag_col]
    # if ra_wrap:
    #      ra = wcorr.ra_shift(ra)
       
    sub = np.zeros(len(ra), dtype='int8')
    print('imag  ngal')
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        sub[sel] = imag
        print(imag, len(t[sel]))
    galcat = wcorr.Cat(ra, dec, sub=sub)
    galcat.assign_jk(limits, nra, ndec, verbose=1)

    pymask = pymangle.Mangle(mask_file)
    # genrand_range does not work if limits wrap zero
    # ra, dec = pymask.genrand_range(nran, *limits)
    ra, dec = pymask.genrand(nran)
    sel = ((((ra - limits[0]) % 360 <= (limits[1] - limits[0]) % 360) *
            (limits[2] <= dec) * (dec < limits[3]) * (mask(ra, dec) == 0)))
    rancat = wcorr.Cat(ra[sel].astype('float64'), dec[sel].astype('float64'))
    rancat.assign_jk(limits, nra, ndec, verbose=1)

    print(len(ra[sel]), 'out of', nran, 'unmasked randoms')

    njack = nra*ndec
    for ijack in range(njack+1):
        rcoords = rancat.sample(ijack)
        info = {'Jack': ijack, 'Nran': len(rcoords[0]), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_J{ijack}.pkl'
        pool.apply_async(wcorr.wcounts, args=(*rcoords, bins, info, outfile))
        for imag in range(len(magbins) - 1):
            mlo, mhi = magbins[imag], magbins[imag+1]
            gcoords = galcat.sample(ijack, sub=imag)
            info = {'Jack': ijack, 'mlo': mlo, 'mhi': mhi,
                    'Ngal': len(gcoords[0]), 'Nran': len(rcoords[0]),
                                'bins': bins, 'tcen': tcen}
            outfile = f'{out_pref}GG_J{ijack}_m{imag}.pkl'
            print(ijack, imag, len(gcoords[0]), len(rcoords[0]), outfile)
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


def w_plot_class(nmag=1, njack=10, fit_range=[0.01, 5], p0=[0.05, 1.7],
           avgcounts=False):
    """w(theta) for different source classes"""

    for (num, NS) in zip((1, 2), 'NS'):
        plt.clf()
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, num=num)
        for (row, TC_class) in zip((0, 1), ('gal', 'star')):
            for (col, BC_class) in zip((0, 1, 2), ('amb', 'gal', 'star')):
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
                            wcorr.Corr1d(info['Ngal'], info['Nran'],
                                         DD_counts, DR_counts, RR_counts,
                                         mlo=info['mlo'], mhi=info['mhi']))
                    corr = corrs[0]
                    corr.err = np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
                    corr.ic_calc(fit_range, p0, 5)
                    color = next(ax._get_lines.prop_cycler)['color']
                    corr.plot(ax, color=color, label=f"m = [{info['mlo']}, {info['mhi']}]")
                    popt, pcov = corr.fit_w(fit_range, p0, ax, color)
                    print(popt, pcov)
        plt.loglog()
        plt.legend()
        plt.xlabel(r'$\theta$ / degrees')
        plt.ylabel(r'$w(\theta)$')
        plt.show()


def w_plot(nmag=7, njack=10, fit_range=[0.01, 5], p0=[0.05, 1.7],
           prefix='wmag_N/', avgcounts=False, Nz_file='Nz.pkl',
           gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7):
    """w(theta) from angular pair counts in mag bins.
    Use observed N(z) if Nz_file specified, otherwise use LF prediction."""

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
                      r0=r0, eps=eps, lf_pars='lf_pars.pkl', Nz_file=Nz_file)


def w_plot_pred(nmag=7, njack=10, fit_range=[0.01, 1], p0=[0.05, 1.7],
                prefix='wmag_N/',
                avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7):
    """Plot observed and predicted w(theta) in mag bins.
    Use observed N(z) if Nz_file specified, otherwise use LF prediction."""

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
                      r0=r0, eps=eps, lf_pars='lf_pars.pkl')


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
    counts_dict = {'zbins': zbins, 'zcen': zcen}
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

        counts_dict.update({imag: (mlo, mhi, counts, popt)})
        plt.stairs(counts, edges, color=color, label=f"m = {mlo}, {mhi}]")
        # plt.plot(zp, spline(zp), color=color, ls='-')
        plt.plot(zp, be_fit(zp, *popt), color=color, ls='-')
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
