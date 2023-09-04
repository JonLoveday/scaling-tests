# Clustering measurements for DESI EDR

import glob
import math
import multiprocessing as mp
import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import default_rng
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

solid_angle_north = 1
def wcounts_N():
    """Angular pair counts for DESI north."""
    wcounts(galfile='BGS_ANY_N_clustering.dat.fits',
            ranfile='BGS_ANY_N_0_clustering.ran.fits',
            out_path='/pscratch/sd/l/loveday/DESI/w_N/')

    
def wcounts_S():
    """Angular pair counts in mag bins."""
    wcounts(galfile='BGS_ANY_S_clustering.dat.fits',
            ranfile='BGS_ANY_S_0_clustering.ran.fits',
            out_path='/pscratch/sd/l/loveday/DESI/w_S/')

    
def wcounts(galfile, ranfile, out_path,
            tmin=0.01, tmax=10, nbins=20,
            zbins=np.linspace(0.0, 0.5, 6)):
    """Angular pair counts in redshift bins."""

    def create_cat(infile):
        """Create catalogue from specified input file."""
        path = '/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering/'
        t = Table.read(path + infile)
        ra, dec, z, rosette = t['RA'], t['DEC'], t['Z'], t['ROSETTE_NUMBER']

        # Assign jackknife regions by rosette number - note these aren't contiguous
        rosettes = np.unique(rosette)
        njack = len(rosettes)
        jack = np.zeros(len(z))
        for ir in range(njack):
            sel = rosette == rosettes[ir]
            jack[sel] = ir + 1

        # Divide into redshift bins
        sub = np.zeros(len(ra), dtype='int8')
        print('iz  nobj')
        for iz in range(len(zbins) - 1):
            sel = (zbins[iz] <= z) * (z < zbins[iz+1])
            sub[sel] = iz
            print(iz, len(z[sel]))
        cat = wcorr.Cat(ra, dec, sub=sub, jack=jack)
        return cat, njack

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    galcat, njack = create_cat(galfile)
    rancat, njack_ran = create_cat(ranfile)
    assert (njack == njack_ran)

    for iz in range(len(zbins) - 1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        for ijack in range(njack+1):
            gcoords = rancat.sample(ijack, sub=iz)
            rcoords = rancat.sample(ijack, sub=iz)
            info = {'Jack': ijack, 'zlo': zlo, 'zhi': zhi,
                    'Ngal': len(gcoords[0]), 'Ngal': len(rcoords[0]),
                    'bins': bins, 'tcen': tcen}
            outfile = f'{out_path}/RR_J{ijack}_z{iz}.pkl'
            result = pool.apply_async(
                wcorr.wcounts, args=(*rcoords, bins, info, outfile))
            print(result.get())
            outfile = f'{out_path}/GG_J{ijack}_z{iz}.pkl'
            result = pool.apply_async(
                wcorr.wcounts, args=(*gcoords, bins, info, outfile))
            print(result.get())
            outfile = f'{out_path}/GR_J{ijack}_z{iz}.pkl'
            result = pool.apply_async(wcorr.wcounts,
                             args=(*gcoords, bins, info,  outfile, *rcoords))
            print(result.get())
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


def w_plot(nmag=7, njack=10, fit_range=[0.01, 5], p0=[0.05, 1.7],
           prefix='wmag_N/', avgcounts=False, Nz_file='Nz.pkl',
           gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7):
    """w(theta) from angular pair counts in mag bins.
    Use observed N(z) if Nz_file specified, otherwise use LF prediction."""

    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for iz in range(nmag):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{prefix}RR_J{ijack}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GG_J{ijack}_m{iz}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GR_J{ijack}_m{iz}.pkl'
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
    for iz in range(nmag):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{prefix}RR_J{ijack}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GG_J{ijack}_m{iz}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GR_J{ijack}_m{iz}.pkl'
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
    for iz in range(len(magbins) - 1):
        mlo, mhi = magbins[iz], magbins[iz+1]
        sel = (magbins[iz] <= mag) * (mag < magbins[iz+1])
        color = next(ax._get_lines.prop_cycler)['color']
        counts, edges = np.histogram(z[sel], zbins)
        popt, pcov = scipy.optimize.curve_fit(
            be_fit, zcen, counts, p0=(0.5, 2.0, 1.5, 1e6), ftol=1e-3, xtol=1e-3)
        print(popt)

        counts_dict.update({iz: (mlo, mhi, counts, popt)})
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
