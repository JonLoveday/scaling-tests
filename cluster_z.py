# Clustering redshifts

from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable, Table
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import scipy
import treecorr
import wcorr

class healpixMask:
    """healpix mask for tracking overlap between two catalogues."""
    
    def __init__(self, cat1, cat2, nside=64, f_occ=0.5):
        """Find ids of healpix pixels that contain objects from both catalogues.
        This done via the following algorithm:
        For each catalogue:
        1. Calculate mean number of points in each occupied pixel
        2. Select pixels with at least f_occ*mean points
        Final pixel list is AND of the pixel lists for each catalogue."""

        self.hp = HEALPix(nside)
        usepix1 = self.occupied_pixels(cat1, f_occ)
        usepix2 = self.occupied_pixels(cat2, f_occ)
        self.usepix = np.intersect1d(usepix1, usepix2)
        print(len(self.usepix), 'pixels common to both catalogues')

    def occupied_pixels(self, cat, f_occ):
        """Returns list of occupied pixels."""
        coords = SkyCoord(cat['RA'].value, cat['DEC'].value, unit='deg', frame='icrs')
        pixels, counts = np.unique(
            self.hp.lonlat_to_healpix(coords.ra, coords.dec),
            return_counts=True)
        mean = np.mean(counts)
        usepix = pixels[counts > f_occ*mean]
        print(len(pixels), 'with 1 or more points, mean count= =', mean)
        print(len(usepix), 'with more than', f_occ*mean, 'counts')
        return usepix

    def select(self, cat, plot=False):
        """Return indices to cat objects within healpix mask."""
        coords = SkyCoord(cat['RA'].value, cat['DEC'].value, unit='deg', frame='icrs')
        pixels = self.hp.lonlat_to_healpix(coords.ra, coords.dec)
        sel = np.isin(pixels, self.usepix)
        if plot:
            plt.clf()
            plt.scatter(coords.ra, coords.dec, s=0.1)
            print(len(cat[sel]), 'out of', len(cat), 'points selected')
            plt.scatter(coords[sel].ra, coords[sel].dec, s=0.1, c='r')
            plt.xlabel('RA')
            plt.ylabel('Dec')
            plt.show()
            plt.savefig(plot)
        return sel
    

def pair_counts(spec_gal_file, spec_ran_file, phot_gal_file, phot_ran_file,
                out_dir, mag_fn, ra_col='RA', dec_col='DEC', z_col='Z', 
                magbins=np.linspace(18, 23, 6), zbins=np.linspace(0.0, 1.0, 11),
                tmin=0.001, tmax=10, nbins=20, nran=1, npatch=9,
                nside=64, f_occ=0.5, exclude_psf=True, identify_overlap=True):
    """Perform paircounts using treecorr.
    Auto-counts for spec sample in redshift bins.
    Cross-corr between spec redshift bins and phot mag bins."""

    def patch_plot(cat, ax, label, ra_shift=False):
        ras = cat.ra
        if ra_shift:
            ras = np.array(ras) - math.pi
            neg = ras < 0
            ras[neg] += 2*math.pi
        ax.scatter(ras, cat.dec, c=cat.patch, s=0.1)
        ax.text(0.05, 0.05, label, transform=ax.transAxes)

    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Read in galaxy and random catalogues
    spec_gal = QTable.read(spec_gal_file)
    spec_ran = QTable.read(spec_ran_file)
    phot_gal = QTable.read(phot_gal_file)
    phot_ran = QTable.read(phot_ran_file)

    # Remove Legacy PSF sources
    if exclude_psf:
        phot_gal = phot_gal[phot_gal['LTYPE'] != 'PSF']
        
    # Select points in overlap between catalogues
    if identify_overlap:
        hpmask = healpixMask(spec_ran, phot_ran)
        spec_gal = spec_gal[hpmask.select(spec_gal, plot=out_dir+'/heal_sgal.png')]
        spec_ran = spec_ran[hpmask.select(spec_ran, plot=out_dir+'/heal_sran.png')]
        phot_gal = phot_gal[hpmask.select(phot_gal, plot=out_dir+'/heal_pgal.png')]
        phot_ran = phot_ran[hpmask.select(phot_ran, plot=out_dir+'/heal_pran.png')]

    # Read catalogues, assign patches, and plot.
    # Define patches using phot_ran_cat.
    phot_ran_cat = treecorr.Catalog(
        ra=phot_ran[ra_col], dec=phot_ran[dec_col], ra_units='deg', dec_units='deg',
        npatch=npatch)
    patch_centers = phot_ran_cat.patch_centers
    phot_gal_cat = treecorr.Catalog(
        ra=phot_gal[ra_col], dec=phot_gal[dec_col], ra_units='deg', dec_units='deg',
        patch_centers=patch_centers)
    spec_ran_cat = treecorr.Catalog(
        ra=spec_ran[ra_col], dec=spec_ran[dec_col], ra_units='deg', dec_units='deg',
        patch_centers=patch_centers)
    spec_gal_cat = treecorr.Catalog(
        ra=spec_gal[ra_col], dec=spec_gal[dec_col], ra_units='deg', dec_units='deg',
        patch_centers=patch_centers)
    ra_shift = 'sgc' in out_dir
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(hspace=0, wspace=0)
    patch_plot(spec_gal_cat, axes[0, 0], 'spec_gal', ra_shift=ra_shift)
    patch_plot(spec_ran_cat, axes[1, 0], 'spec_ran', ra_shift=ra_shift)
    patch_plot(phot_gal_cat, axes[0, 1], 'phot_gal', ra_shift=ra_shift)
    patch_plot(phot_ran_cat, axes[1, 1], 'phot_ran', ra_shift=ra_shift)
    plt.show()
    plt.savefig(out_dir + '/patch.png')

    # spec-spec and spec-phot random-random counts

    spec_rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                     sep_units='degrees')
    spec_rr.process(spec_ran_cat)

    spec_phot_rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                          sep_units='degrees')
    spec_phot_rr.process(spec_ran_cat, phot_ran_cat)

    mag = mag_fn(phot_gal)
    z = spec_gal[z_col]
    rd_list = []

    # spec auto-pair counts in redshift bins
    for iz in range(len(zbins) - 1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        sel = (zlo <= z) * (z < zhi)
        zmean = np.mean(z[sel])
        spec_gal_cat = treecorr.Catalog(
            ra=spec_gal[ra_col][sel], dec=spec_gal[dec_col][sel],
            ra_units='deg', dec_units='deg',
            patch_centers=patch_centers)

        spec_dr = treecorr.NNCorrelation(
            min_sep=tmin, max_sep=tmax, nbins=nbins, sep_units='degrees')
        spec_dr.process(spec_gal_cat, spec_ran_cat)
        spec_dd = treecorr.NNCorrelation(
            min_sep=tmin, max_sep=tmax, nbins=nbins,
            sep_units='degrees', var_method='jackknife')
        spec_dd.process(spec_gal_cat)
        spec_dd.calculateXi(rr=spec_rr, dr=spec_dr)
        xi_jack, w = spec_dd.build_cov_design_matrix('jackknife')
        outfile = f'{out_dir}/az{iz}.fits'
        spec_dd.write(outfile, rr=spec_rr, dr=spec_dr)
        with fits.open(outfile, mode='update') as hdul:
            hdr = hdul[1].header
            hdr['zlo'] = zlo
            hdr['zhi'] = zhi
            hdr['zmean'] = zmean
            hdr['Ngal'] = spec_gal_cat.nobj
            hdr['Nran'] = spec_ran_cat.nobj
            hdul.append(fits.PrimaryHDU(xi_jack))
            hdul.flush()

        spec_phot_dr = treecorr.NNCorrelation(
            min_sep=tmin, max_sep=tmax, nbins=nbins, sep_units='degrees')
        spec_phot_dr.process(spec_gal_cat, phot_ran_cat)

        # X-corr with phot mag bins
        for im in range(len(magbins) - 1):
            mlo, mhi = magbins[im], magbins[im+1]
            sel = (mlo <= mag) * (mag < mhi)
            phot_gal_cat = treecorr.Catalog(
                ra=phot_gal[ra_col][sel], dec=phot_gal[dec_col][sel],
                ra_units='deg', dec_units='deg',
                patch_centers=patch_centers)

            if iz == 0:
                spec_phot_rd = treecorr.NNCorrelation(
                    min_sep=tmin, max_sep=tmax, nbins=nbins,
                    sep_units='degrees')
                spec_phot_rd.process(spec_ran_cat, phot_gal_cat)
                rd_list.append(spec_phot_rd)
            else:
                spec_phot_rd = rd_list[im]

            spec_phot_dd = treecorr.NNCorrelation(
                min_sep=tmin, max_sep=tmax, nbins=nbins,
                sep_units='degrees', var_method='jackknife')
            spec_phot_dd.process(spec_gal_cat, phot_gal_cat)
            spec_phot_dd.calculateXi(rr=spec_phot_rr, dr=spec_phot_dr, rd=spec_phot_rd)
            xi_jack, w = spec_phot_dd.build_cov_design_matrix('jackknife')
            outfile = f'{out_dir}/xz{iz}m{im}.fits'
            spec_phot_dd.write(outfile, rr=spec_phot_rr, dr=spec_phot_dr, rd=spec_phot_rd)
            with fits.open(outfile, mode='update') as hdul:
                hdr = hdul[1].header
                hdr['zlo'] = zlo
                hdr['zhi'] = zhi
                hdr['zmean'] = zmean
                hdr['mlo'] = mlo
                hdr['mhi'] = mhi
                hdr['Ngal1'] = spec_gal_cat.nobj
                hdr['Ngal2'] = phot_gal_cat.nobj
                hdr['Nran1'] = spec_ran_cat.nobj
                hdr['Nran2'] = phot_ran_cat.nobj
                hdul.append(fits.PrimaryHDU(xi_jack))
                hdul.flush()

def be_fit(z, zc, alpha, beta, norm):
    """Generalised Baugh & Efstathiou (1993, eqn 7) model for N(z)."""
    return norm * z**alpha * np.exp(-(z/zc)**beta)

def Nz(fit_range=[0.001, 1], p0=[0.05, 1.7], rmin=0.01, rmax=10, ylim=(-0.5, 1),
       weight=-1, fitbin=0, zsamp=np.linspace(0, 2, 41)):
    """Cluster redshifts from angular clustering of galaxies in mag bins about
    reference sample in redshift bins.
    w(theta) integral is weighted by theta^weight.
    Set fitbin=1 to integrate binned w(teta) directly, else use power-law fit."""
     
    nz = len(glob.glob('az*.fits'))
    nm = len(glob.glob('xz0m*.fits'))
    with fits.open('az0.fits') as hdul:
        njack = hdul[2].shape[0]

    # w_rr in redshift bins
    w_rr_av = np.zeros(nz)
    w_rr_av_jack = np.zeros((njack, nz))
    d, zmean = np.zeros(nz), np.zeros(nz)
    fig, axes = plt.subplots(1, nz, sharex=True, sharey=True)
    fig.set_size_inches(16, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iz in range(nz):
        infile = f'az{iz}.fits'
        corr = wcorr.Corr1d(infile)
        ax = axes[iz]
        corr.plot(ax=ax)
        popt, pcov = corr.fit_w(fit_range, p0, ax)
        ax.text(0.0, 1.05, f"z=[{corr.meta['ZLO']:3.2f}, {corr.meta['ZHI']:3.2f}]",
                transform=ax.transAxes)
        zmean[iz] = corr.meta['ZMEAN']
        d[iz] = cosmo.comoving_distance(zmean[iz]).value
        tmin, tmax = 180/math.pi*rmin/d[iz], 180/math.pi*rmax/d[iz]
        if fitbin:
            w_rr_av[iz] = corr.integral(tmin, tmax, weight)
        else:
            w_rr_av[iz] = wcorr.power_law_integral(*popt, weight, tmin, tmax)
        print(f'{zmean[iz]:4.3f}, {d[iz]:4.3f}, {popt[0]:4.3f}, {popt[1]:4.3f}, {tmin:4.3e}, {tmax:4.3e}, {w_rr_av[iz]:4.3f}')
        ax.axvline(tmin, c='g')
        ax.axvline(tmax, c='g')

        # Fits to jackknife
        for ijack in range(njack):
            if fitbin:
                w_rr_av_jack[ijack, iz] = corr.integral(tmin, tmax, weight, ijack+1)
            else:
                popt, pcov = corr.fit_w(fit_range, p0, ax, ijack=ijack+1)
                w_rr_av_jack[ijack, iz] = wcorr.power_law_integral(*popt, weight, tmin, tmax)
            
    plt.loglog()
    axes[nz//2].set_xlabel(r'$\theta$ / degrees')
    axes[0].set_ylabel(r'$w(\theta)$')
    plt.show()

    # w_rt in redshift-magnitude bins
    w_rt_av = np.zeros((nz, nm))
    w_rt_av_jack = np.zeros((njack, nz, nm))
    mlo, mhi = np.zeros(nm), np.zeros(nm)
    fig, axes = plt.subplots(nm, nz, sharex=True, sharey=True)
    fig.set_size_inches(16, 8)
    fig.subplots_adjust(hspace=0, wspace=0)
    print('Zmean Distance im  A   gamma   tmin  tmax  w_av')
    for iz in range(nz):
        for im in range(nm):
            infile = f'xz{iz}m{im}.fits'
            corr = wcorr.Corr1d(infile)
            ax = axes[im, iz]
            corr.plot(ax=ax)
            popt, pcov = corr.fit_w(fit_range, p0, ax)
            mlo[im], mhi[im] = corr.meta['MLO'], corr.meta['MHI']
            if im == 0:
                ax.text(0.1, 1.05, f"z=[{corr.meta['ZLO']:3.2f}, {corr.meta['ZHI']:3.2f}]",
                        transform=ax.transAxes)
            if iz == nz-1:
                ax.text(1.05, 0.5, f"m=[{mlo[im]:3.1f}, {mhi[im]:3.1f}]",
                        transform=ax.transAxes)
            tmin, tmax = 180/math.pi*rmin/d[iz], 180/math.pi*rmax/d[iz]
            if fitbin:
                w_rt_av[iz, im] = corr.integral(tmin, tmax, weight)
            else:
                w_rt_av[iz, im] = wcorr.power_law_integral(*popt, weight, tmin, tmax)
            print(f'{zmean[iz]:4.3f}, {d[iz]:4.3f}, {im}, {popt[0]:4.3f}, {popt[1]:4.3f}, {tmin:4.3e}, {tmax:4.3e}, {w_rt_av[iz, im]:4.3f}')
            ax.axvline(tmin, c='g')
            ax.axvline(tmax, c='g')

            # Fits to jackknife
            for ijack in range(njack):
                if fitbin:
                    w_rt_av_jack[ijack, iz, im] = corr.integral(tmin, tmax, weight, ijack+1)
                else:   
                    popt, pcov = corr.fit_w(fit_range, p0, ax, ijack=ijack+1)
                    w_rt_av_jack[ijack, iz, im] = wcorr.power_law_integral(*popt, weight, tmin, tmax)
            print(np.array_str(w_rt_av_jack[:, iz, im], precision=3))
    plt.loglog()
    axes[nm-1, nz//2].set_xlabel(r'$\theta$ / degrees')
    axes[nm//2, 0].set_ylabel(r'$w(\theta)$')
    plt.show()

    # N(z) in mag bins
    be_pars = np.zeros((4, nm))
    pmz = np.zeros((nz, nm))
    pmz_err = np.zeros((nz, nm))
    pmz_jack = np.zeros((njack, nz, nm))
    fig, axes = plt.subplots(1, nm, sharex=True, sharey=True)
    fig.set_size_inches(16, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for im in range(nm):
        ax = axes[im]
        pmz[:, im] = w_rt_av[:, im]/w_rr_av**0.5
        for ijack in range(njack):
            pmz_jack[ijack, :, im] = w_rt_av_jack[ijack, :, im]/w_rr_av_jack[ijack, :]**0.5
        pmz_err[:, im] = (njack-1)**0.5 * np.std(pmz_jack[:, :, im], axis=0)
        bad = np.isinf(pmz[:, im]) + np.isnan(pmz[:, im])
        pmz[bad, im] = 0
        pmz_err[bad, im] = 1
        ax.errorbar(zmean, pmz[:, im], pmz_err[:, im])
        sel = pmz_err[:, im] > 0
        try:
            popt, pcov = scipy.optimize.curve_fit(
                be_fit, zmean[sel],  pmz[sel, im], sigma=pmz_err[sel, im],
                p0=(0.5, 2.0, 1.5, 1), bounds=(0, np.inf), ftol=1e-3, xtol=1e-3)
            ax.plot(zsamp, be_fit(zsamp, *popt), ls='-')
            be_pars[:, im] = popt
            print(popt)
        except RuntimeError:
            print('Error in fit')
        ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                transform=ax.transAxes)
        # print(zmean, pmz[:, im], pmz_err[:, im])
    axes[nm//2].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$N(z)$')
    plt.ylim(ylim)
    plt.show()

    pickle.dump((zmean, pmz, pmz_err, mlo, mhi, be_pars, rmin, rmax, weight, fitbin),
                open('Nz.pkl', 'wb'))


def Nz_average(indirs, magbins=np.linspace(16, 22, 7),
               zbins=np.linspace(0.0, 1.1, 45), ylim=(-0.5, 1)):
    """Inverse-variance weighted average N(z) from pairs of tracers in indirs."""
    nest, nz, nm = len(indirs), len(zbins) - 1, len(magbins) - 1
    Pz, invvar = np.zeros((nest, nz, nm)), np.zeros((nest, nz, nm))
    zstep = zbins[1] - zbins[0]
    iest = 0
    for indir in indirs:
        (zmean, pmz, pmz_err, mlo, mhi, be_pars) = pickle.load(open(indir+'/Nz.pkl', 'rb'))
        assert nm == len(mlo)
        fig, axes = plt.subplots(1, nm, sharex=True, sharey=True)
        fig.set_size_inches(8, 4)
        fig.subplots_adjust(hspace=0, wspace=0)
        for im in range(nm):
            ax = axes[im]
            ax.errorbar(zmean, pmz[:, im], pmz_err[:, im])
            ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                    transform=ax.transAxes)
            iz = 0
            for z in zmean:
                oz = int((z - zbins[0])/zstep)
                Pz[iest, oz, im] = pmz[iz, im]
                invvar[iest, oz, im] = pmz_err[iz, im]**-2
                iz += 1
        axes[nm//2].set_xlabel(r'Redshift')
        axes[0].set_ylabel(r'$N(z)$')
        plt.ylim(ylim)
        plt.show()
        iest += 1
    Pz_mean, invvar_sum = np.ma.average(Pz, axis=0, weights=invvar, returned=True)
     
    fig, axes = plt.subplots(1, nm, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for im in range(nm):
        ax = axes[im]
        ax.errorbar(zbins[1:] - 0.5*zstep, Pz_mean[:, im], invvar_sum[:, im]**-0.5)
        ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                transform=ax.transAxes)
    axes[nm//2].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$N(z)$')
    plt.show()


def Nz_plot(ylim=(-0.5, 1)):
    """Plot N(z)."""
    (zmean, pmz, pmz_err, mlo, mhi, be_pars) = pickle.load(open('Nz.pkl', 'rb'))
    nm = len(mlo)
    fig, axes = plt.subplots(1, nm, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for im in range(nm):
        ax = axes[im]
        ax.errorbar(zmean, pmz[:, im], pmz_err[:, im])
        ax.plot(zmean, be_fit(zmean, *be_pars[:, im]), ls='-')
        ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                transform=ax.transAxes)
    axes[nm//2].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$N(z)$')
    plt.ylim(ylim)
    plt.show()
