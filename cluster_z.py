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
                out_dir, mag_fn, z_col='Z', 
                magbins=np.linspace(18, 23, 6), zbins=np.linspace(0.0, 1.0, 11),
                tmin=0.001, tmax=10, nbins=20, nran=1, npatch=9,
                nside=64, f_occ=0.5):
    """Perform paircounts using treecorr.
    Auto-counts for spec sample in redshift bins.
    Cross-corr between spec redshift bins and phot mag bins."""

    # Create out_dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Read in galaxy and random catalogues
    spec_gal = QTable.read(spec_gal_file)
    spec_ran = QTable.read(spec_ran_file)
    phot_gal = QTable.read(phot_gal_file)
    phot_ran = QTable.read(phot_ran_file)
    
    # Select points in overlap between catalogues
    hpmask = healpixMask(spec_ran, phot_ran)
    spec_gal = spec_gal[hpmask.select(spec_gal, plot=out_dir+'/heal_sgal.png')]
    spec_ran = spec_ran[hpmask.select(spec_ran, plot=out_dir+'/heal_sran.png')]
    phot_gal = phot_gal[hpmask.select(phot_gal, plot=out_dir+'/heal_pgal.png')]
    phot_ran = phot_ran[hpmask.select(phot_ran, plot=out_dir+'/heal_pran.png')]

    # spec-spec and spec-phot random-random counts
    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))

    spec_ran_cat = treecorr.Catalog(
        ra=spec_ran['RA'], dec=spec_ran['DEC'], ra_units='deg', dec_units='deg',
        npatch=npatch)
    spec_rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                     sep_units='degrees')
    spec_rr.process(spec_ran_cat)

    phot_ran_cat = treecorr.Catalog(
        ra=phot_ran['RA'], dec=phot_ran['DEC'], ra_units='deg', dec_units='deg',
        patch_centers=spec_ran_cat.patch_centers)
    spec_phot_rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                          sep_units='degrees')
    spec_phot_rr.process(spec_ran_cat, phot_ran_cat)

    mag = mag_fn(phot_gal)
    z = spec_gal[z_col]
    rd_list = []

    patch_plot = out_dir + '/patch.png'
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].scatter(spec_ran_cat.ra, spec_ran_cat.dec, c=spec_ran_cat.patch, s=0.1)
    axes[1].scatter(phot_ran_cat.ra, phot_ran_cat.dec, c=phot_ran_cat.patch, s=0.1)
    plt.show()
    plt.savefig(patch_plot)

    # spec auto-pair counts in redshift bins
    for iz in range(len(zbins) - 1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        sel = (zlo <= z) * (z < zhi)
        zmean = np.mean(z[sel])
        spec_gal_cat = treecorr.Catalog(
            ra=spec_gal['RA'][sel], dec=spec_gal['DEC'][sel],
            ra_units='deg', dec_units='deg',
            patch_centers=spec_ran_cat.patch_centers)

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
                ra=phot_gal['RA'][sel], dec=phot_gal['DEC'][sel],
                ra_units='deg', dec_units='deg',
                patch_centers=spec_ran_cat.patch_centers)

            if iz == 0:
                spec_phot_rd = treecorr.NNCorrelation(
                    min_sep=tmin, max_sep=tmax, nbins=nbins,
                    sep_units='degrees', var_method='jackknife')
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

def Nz(fit_range=[0.001, 1], p0=[0.05, 1.7], rmin=0.01, rmax=10):
    """Cluster redshifts from angular clustering of galaxies in mag bins about
    reference sample in redshift bins."""

    nz = len(glob.glob('az*.fits'))
    nm = len(glob.glob('xz0m*.fits'))
    with fits.open('az0.fits') as hdul:
        njack = hdul[2].shape[0]

    # w_rr in redshift bins
    w_rr_av = np.zeros(nz)
    w_rr_av_jack = np.zeros((njack, nz))
    d, zmean = np.zeros(nz), np.zeros(nz)
    fig, axes = plt.subplots(1, nz, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iz in range(nz):
        infile = f'az{iz}.fits'
        t = Table.read(infile)
        with fits.open(infile) as hdul:
            xi_jack = hdul[2].data
        corr = wcorr.Corr1d()
        corr.sep = t['meanr']
        corr.est = np.vstack((t['xi'], xi_jack))
        corr.err = t['sigma_xi']
        corr.r1r2 = t['RR']
        ax = axes[iz]
        corr.plot(ax=ax)
        popt, pcov = corr.fit_w(fit_range, p0, ax)
        A = popt[0]
        omg = 1 - popt[1]
        ax.text(0.0, 1.05, f"z=[{t.meta['ZLO']:3.2f}, {t.meta['ZHI']:3.2f}]",
                transform=ax.transAxes)
        zmean[iz] = t.meta['ZMEAN']
        d[iz] = cosmo.comoving_distance(zmean[iz]).value
        tmin, tmax = 180/math.pi*rmin/d[iz], 180/math.pi*rmax/d[iz]
        w_rr_av[iz] = A/omg*(tmax**omg - tmin**omg)
        print(f'{zmean[iz]:4.3f}, {d[iz]:4.3f}, {popt[0]:4.3f}, {popt[1]:4.3f}, {tmin:4.3e}, {tmax:4.3e}, {w_rr_av[iz]:4.3f}')
        ax.axvline(tmin, c='g')
        ax.axvline(tmax, c='g')

        # Fits to jackknife
        for ijack in range(njack):
            popt, pcov = corr.fit_w(fit_range, p0, ax, ijack=ijack+1)
            A = popt[0]
            omg = 1 - popt[1]
            w_rr_av_jack[ijack, iz] = A/omg*(tmax**omg - tmin**omg)
            
    plt.loglog()
    axes[nz//2].set_xlabel(r'$\theta$ / degrees')
    axes[0].set_ylabel(r'$w(\theta)$')
    plt.show()

    # w_rt in redshift-magnitude bins
    w_rt_av = np.zeros((nz, nm))
    w_rt_av_jack = np.zeros((njack, nz, nm))
    mlo, mhi = np.zeros(nm), np.zeros(nm)
    fig, axes = plt.subplots(nm, nz, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    print('Zmean Distance im  A   gamma   tmin  tmax  w_av')
    for iz in range(nz):
        for im in range(nm):
            infile = f'xz{iz}m{im}.fits'
            t = Table.read(infile)
            with fits.open(infile) as hdul:
                xi_jack = hdul[2].data
            corr = wcorr.Corr1d()
            corr.sep = t['meanr']
            corr.est = np.vstack((t['xi'], xi_jack))
            corr.err = t['sigma_xi']
            corr.r1r2 = t['RR']
            ax = axes[im, iz]
            corr.plot(ax=ax)
            popt, pcov = corr.fit_w(fit_range, p0, ax)
            A = popt[0]
            omg = 1 - popt[1]
            mlo[im], mhi[im] = t.meta['MLO'], t.meta['MHI']
            if im == 0:
                ax.text(0.1, 1.05, f"z=[{t.meta['ZLO']:3.2f}, {t.meta['ZHI']:3.2f}]",
                        transform=ax.transAxes)
            if iz == nz-1:
                ax.text(1.05, 0.5, f"m=[{mlo[im]:3.1f}, {mhi[im]:3.1f}]",
                        transform=ax.transAxes)
            tmin, tmax = 180/math.pi*rmin/d[iz], 180/math.pi*rmax/d[iz]
            w_rt_av[iz, im] = A/omg*(tmax**omg - tmin**omg)
            print(f'{zmean[iz]:4.3f}, {d[iz]:4.3f}, {im}, {popt[0]:4.3f}, {popt[1]:4.3f}, {tmin:4.3e}, {tmax:4.3e}, {w_rt_av[iz, im]:4.3f}')
            ax.axvline(tmin, c='g')
            ax.axvline(tmax, c='g')

            # Fits to jackknife
            for ijack in range(njack):
                popt, pcov = corr.fit_w(fit_range, p0, ax, ijack=ijack+1)
                A = popt[0]
                omg = 1 - popt[1]
                w_rt_av_jack[ijack, iz, im] = A/omg*(tmax**omg - tmin**omg)
            print(np.array_str(w_rt_av_jack[:, iz, im], precision=3))
    plt.loglog()
    axes[nm-1, nz//2].set_xlabel(r'$\theta$ / degrees')
    axes[nm//2, 0].set_ylabel(r'$w(\theta)$')
    plt.show()

    # N(z) in mag bins
    pmz = np.zeros((nz, nm))
    pmz_err = np.zeros((nz, nm))
    pmz_jack = np.zeros((njack, nz, nm))
    fig, axes = plt.subplots(1, nm, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for im in range(nm):
        ax = axes[im]
        pmz[:, im] = w_rt_av[:, im]/w_rr_av**0.5
        for ijack in range(njack):
            pmz_jack[ijack, :, im] = w_rt_av_jack[ijack, :, im]/w_rr_av_jack[ijack, :]**0.5
        pmz_err[:, im] = (njack-1)**0.5 * np.std(pmz_jack[:, :, im], axis=0)
        ax.errorbar(zmean, pmz[:, im], pmz_err[:, im])
        ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                transform=ax.transAxes)
    axes[nm//2].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$N(z)$')
    plt.show()

    pickle.dump((zmean, pmz, pmz_err, mlo, mhi), open('Nz.pkl', 'wb'))


def Nz_average(indirs, magbins=np.linspace(16, 22, 7),
               zbins=np.linspace(0.0, 1.1, 45)):
    """Inverse-variance weighted average N(z) from pairs of tracers in indirs."""
    nest, nz, nm = len(indirs), len(zbins) - 1, len(magbins) - 1
    Pz, invvar = np.zeros((nest, nz, nm)), np.zeros((nest, nz, nm))
    zstep = zbins[1] - zbins[0]
    iest = 0
    for indir in indirs:
        (zmean, pmz, pmz_err, mlo, mhi) = pickle.load(open(indir+'/Nz.pkl', 'rb'))
        assert nm == len(mlo)
        for im in range(nm):
            iz = 0
            for z in zmean:
                oz = int((z - zbins[0])/zstep)
                Pz[iest, oz, im] = pmz[iz, im]
                invvar[iest, oz, im] = pmz_err[iz, im]**-2
        iest += 1
    Pz_mean, invvar_sum = np.average(Pz, axis=0, weights=invvar, returned=True)
     
    fig, axes = plt.subplots(1, nm, sharex=True, sharey=True)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for im in range(nm):
        ax = axes[im]
        ax.errorbar(zbins[1:] - 0.5*zstep, Pz_mean, invvar_sum**-0.5)
        ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                transform=ax.transAxes)
    axes[nm//2].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$N(z)$')
    plt.show()
