# Clustering support utilities, using corrfunc for pair counts

import corner
import emcee
import glob
import math
# from multiprocessing import Pool
import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import default_rng
# import os
# os.environ["OMP_NUM_THREADS"] = "1"  # avoid clash with multiprocessing
import pandas as pd
import pdb
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
import multiprocessing as mp
import pymangle

import calc_kcor
import limber
import util

ln10 = math.log(10)
rng = default_rng()

class Cat(object):
    """Set of galaxies or random points."""

    def __init__(self, ra, dec, r=None, sub=None, jack=None, nthreads=2):
        self.nobj = len(ra)
        self.ra = ra
        self.dec = dec
        if (r is not None):
            self.r = r
        if (sub is not None):
            self.sub = sub
            self.nsub = np.max(sub) + 1
        else:
            self.nsub = 0
        if (jack is not None):
            self.jack = jack
        self.nthreads = nthreads

    def gen_cart(self):
        """Generate Cartesian coordinates from ra, dec, r."""
        rar, decr = np.radians(self.ra), np.radians(self.dec)
        self.x = self.r * np.cos(rar) * np.cos(decr)
        self.y = self.r * np.sin(rar) * np.cos(decr)
        self.z = self.r * np.sin(decr)

    def assign_jk(self, limits, nra, ndec, verbose=0):
        """Assign objects to a jackknife region."""
        ramin, ramax, decmin, decmax = *limits,
        rastep, decstep = (ramax-ramin)/nra, (decmax-decmin)/ndec
        jack = 1  # jack=0 refers to full sample
        self.njack = nra*ndec
        self.jack = np.zeros(self.nobj, dtype='int8')
        if verbose:
            print('jack   nobj')
        for idec in range(ndec):
            declo = decmin + idec*decstep
            dechi = decmin + (idec+1)*decstep
            for ira in range(nra):
                ralo = ramin + ira*rastep
                rahi = ramin + (ira+1)*rastep
                sel = ((ralo <= self.ra) * (self.ra < rahi) *
                       (declo <= self.dec) * (self.dec < dechi))
                self.jack[sel] = jack
                if verbose:
                    print(jack, len(self.jack[sel]))
                jack += 1

    def sample(self, jack=0, sub=-1, cart=False):
        """Return coords, excluding specified jk region if jack > 0,
        and including only specified subsample if sub > -1.
        Returns angular coords, unless cart=True."""
        sel = np.ones(self.nobj, dtype=bool)
        if jack > 0:
            sel *= self.jack != jack
        if sub > -1:
            sel *= self.sub == sub
        if cart:
            return self.x[sel], self.y[sel], self.z[sel]
        else:
            return self.ra[sel], self.dec[sel]

    def write(self, outfile):
        """Write catalogue to csv file."""
        if hasattr(self, 'r'):
            np.savetxt(outfile, (self.ra, self.dec, self.r))
        else:
            np.savetxt(outfile, (self.ra, self.dec))

    def radec_plot(self, plotfile):
        """Ra, dec plot."""
        
class Corr1d(object):
    """1d clustering estimate."""

    def __init__(self, ngal=0, nran=0, dd=0, dr=0, rr=0, mlo=0, mhi=0):

        self.mlo = mlo
        self.mhi = mhi
        self.ic = 0
        if ngal > 0:
            self.ngal = ngal
            self.nran = nran
            try:
                self.lgsep = 0.5*(np.log10(dd['thetamin']) + np.log10(dd['thetamax']))
                self.sep = 10**self.lgsep
                self.sep_av = dd['thetaavg']
            except ValueError:
                self.sep = 10**(0.5*(np.log10(dd['rmin']) + np.log10(dd['rmax'])))
                self.sep_av = dd['ravg']
            self.dd = dd['npairs']
            self.dr = dr['npairs']
            self.rr = rr['npairs']
            self.est = np.nan_to_num(Corrfunc.utils.convert_3d_counts_to_cf(
                ngal, ngal, nran, nran, dd, dr, dr, rr))
            
    def average(self, corrs, avgcounts=False):
        """Average over realisations or subsamples.  Set avgcounts=True
        to average counts rather than averaging corr fn estimate."""

        nest = len(corrs)
        self.mlo = corrs[0].mlo
        self.mhi = corrs[0].mhi
        self.ngal = np.sum(np.array([corrs[i].ngal for i in range(nest)]))
        self.nran = np.sum(np.array([corrs[i].nran for i in range(nest)]))
        self.sep = corrs[0].sep
        self.sep_av = corrs[0].sep
        self.dd = np.sum(np.array([corrs[i].dd for i in range(nest)]), axis=0)
        self.dr = np.sum(np.array([corrs[i].dr for i in range(nest)]), axis=0)
        self.rr = np.sum(np.array([corrs[i].rr for i in range(nest)]), axis=0)
        self.err = np.nan_to_num(np.std(np.array([corrs[i].est for i in range(nest)]), axis=0))
        if avgcounts:
            self.est = np.nan_to_num(Corrfunc.utils.convert_3d_counts_to_cf(
                self.ngal, self.ngal, self.nran, self.nran,
                self.dd, self.dr, self.dr, self.rr))
        else:
            self.est = np.mean(np.array([corrs[i].est for i in range(nest)]), axis=0)

    def ic_calc_old(self, gamma, r0, ic_rmax):
        """Returns estimated integral constraint for power law xi(r)
        truncated at ic_rmax."""
        xi_mod = np.zeros(len(self.sep))
        pos = (self.sep > 0) * (self.sep < ic_rmax)
        xi_mod[pos] = (self.sep[pos]/r0)**-gamma
        self.ic = (self.ranpairs * xi_mod).sum() / (self.ranpairs).sum()

    def ic_calc(self, fit_range, p0=[0.05, 1.7], ic_rmax=10, niter=3):
        """Returns estimated integral constraint for power law xi(r)
        truncated at ic_rmax."""

        def power_law(theta, A, gamma):
            """Power law w(theta) = A theta**(1-gamma)."""
            return A * theta**(1-gamma)
    
        xi_mod = np.zeros(len(self.sep))
        pos = (self.sep > 0) * (self.sep < ic_rmax)
        for i in range(niter):
            popt, pcov = self.fit_w(fit_range, p0)
            xi_mod[pos] = power_law(self.sep[pos], *popt)
            self.ic = (self.rr * xi_mod).sum() / (self.rr).sum()
            print(i, *popt)

    def est_corr(self):
        """Returns integral-constraint corrected estimate."""
        return self.est + self.ic
    
    def plot(self, ax, tscale=1, wscale=1, color=None, fout=None, label=None,
             pl_div=None):
        if pl_div:
            pl_fit = (self.sep/pl_div[0])**(- pl_div[1])
        else:
            pl_fit = 1
        ax.errorbar(tscale*self.sep, wscale*self.est_corr()/pl_fit,
                    wscale*self.err/pl_fit,
                    fmt='o', color=color, label=label, capthick=1)
        if fout:
            print(label, file=fout)
            for i in range(self.nbin):
                print(self.sep[i], self.est_corr()[i],
                      self.cov.sig[i], file=fout)

    def fit_w(self, fit_range, p0=[0.05, 1.7], ax=None, color=None,
              ftol=1e-3, xtol=1e-3):
        """Fit a power law to w(theta)."""

        def power_law(theta, A, gamma):
            """Power law w(theta) = A theta**(1-gamma)."""
            return A * theta**(1-gamma)

        sel = ((self.sep >= fit_range[0]) * (self.sep < fit_range[1]) *
               np.isfinite(self.est) * (self.err > 0) * (self.rr > 10))
        popt, pcov = scipy.optimize.curve_fit(
            power_law, self.sep[sel], self.est_corr()[sel], p0=p0,
            sigma=self.err[sel], ftol=ftol, xtol=xtol)
        if ax:
            ax.plot(self.sep[sel], power_law(self.sep[sel], *popt), color=color)

        return popt, pcov

    def fit_xi(self, fit_range, p0=[5, 1.7], ax=None, color=None,
               ftol=1e-3, xtol=1e-3):
        """Fit a power law to xi(r)."""

        def power_law(r, r0, gamma):
            """Power law xi(r) = (r0/r)**gamma."""
            return (r0/r)**gamma

        sel = ((self.sep >= fit_range[0]) * (self.sep < fit_range[1]) *
               np.isfinite(self.est) * (self.err > 0) * (self.rr > 10))
        popt, pcov = scipy.optimize.curve_fit(
            power_law, self.sep[sel], self.est[sel], p0=p0,
            sigma=self.err[sel], ftol=ftol, xtol=xtol)
        if ax:
            ax.plot(self.sep[sel], power_law(self.sep[sel], *popt), color=color)

        return popt, pcov

    def interp(self, r, jack=0, log=False):
        """Returns interpolated value and error (zero for r > r_max).
        Interpolates in log-log space if log=True."""
        if log:
            return np.expm1(np.interp(np.log(r), np.log(self.sep),
                                      np.log1p(self.est[:, jack]), right=0)), \
                   np.expm1(np.interp(np.log(r), np.log(self.sep),
                                      np.log1p(self.cov.sig)))
        else:
            return np.interp(r, self.sep, self.est[:, jack], right=0), \
                   np.interp(r, self.sep, self.cov.sig)


class Cov(object):
    """Covariance matrix and eigenvalue decomposition."""

    def __init__(self, ests, err_type):
        """Generate covariance matrix from jackknife or mock estimates."""

        dims = ests.shape[:-1]
        ndat = np.prod(dims)
        nest = ests.shape[-1]
        self.cov = np.ma.cov(ests.reshape((ndat, nest), order='F'))
        if err_type == 'jack':
            self.cov *= (nest-1)
        try:
            self.icov = np.linalg.inv(self.cov)
        except:
            print('Unable to invert covariance matrix')
#            pdb.set_trace()
        try:
            self.sig = np.sqrt(np.diag(self.cov)).reshape(dims, order='F')
            self.siginv = np.diag(1.0/np.sqrt(np.diag(self.cov)))
#            pdb.set_trace()
            cnorm = np.nan_to_num(self.siginv.dot(self.cov).dot(self.siginv))
            self.cnorm = np.clip(cnorm, -1, 1)
            eig_val, eig_vec = np.linalg.eigh(self.cnorm)
            idx = eig_val.argsort()[::-1]
            self.eig_val = eig_val[idx]
            self.eig_vec = eig_vec[:, idx]
        except:
            self.sig = np.sqrt(self.cov)
            self.siginv = 1.0/self.sig

    def add(self, cov):
        """Add second covariance matrix to self."""
        self.cov += cov.cov
        dims = self.cov.shape[:-1]
        self.sig = np.sqrt(np.diag(self.cov)).reshape(dims, order='F')
        self.siginv = np.diag(1.0/np.sqrt(np.diag(self.cov)))
        cnorm = np.nan_to_num(self.siginv.dot(self.cov).dot(self.siginv))
        self.cnorm = np.clip(cnorm, -1, 1)
        eig_val, eig_vec = np.linalg.eig(self.cnorm)
        idx = eig_val.argsort()[::-1]
        self.eig_val = eig_val[idx].real
        self.eig_vec = eig_vec[:, idx].real

    def chi2(self, obs, model, neig=0):
        """
        Chi^2 residual between obs and model, using first neig eigenvectors
        (Norberg+2009, eqn 12).  By default (neig=0), use diagonal elements
        only.  Set neig='full' for full covariance matrix,
        'all' for all e-vectors.  For chi2 calcs using mean of mock catalogues,
        multiply returned chi2 by nest to convert from standard deviation
        to standard error."""

        if neig == 0:
            if len(obs) > 1:
                diag = np.diag(self.cov)
                nonz = diag > 0
                return np.sum((obs[nonz] - model[nonz])**2 / diag[nonz])
            else:
                return (obs - model)**2 / self.cov
        if neig == 'full':
            return (obs-model).T.dot(self.icov).dot(obs-model)
        yobs = self.eig_vec.T.dot(self.siginv).dot(obs)
        ymod = self.eig_vec.T.dot(self.siginv).dot(model)
        if neig == 'all':
            return np.sum((yobs - ymod)**2 / self.eig_val)
        else:
            return np.sum((yobs[:neig] - ymod[:neig])**2 / self.eig_val[:neig])

    def plot(self, norm=False, ax=None, label=None):
        """Plot (normalised) covariance matrix."""
        try:
            ndat = self.cov.shape[0]
            extent = (0, ndat, 0, ndat)
            aspect = 1

            if ax is None:
                plt.clf()
                ax = plt.subplot(111)
            if norm:
                val = self.cnorm
                xlabel = 'Normalized Covariance'
            else:
                val = self.cov
                xlabel = 'Covariance'

            im = ax.imshow(val, aspect=aspect, interpolation='none',
                           extent=extent, origin='lower')
            ax.set_xlabel(xlabel)
            if label:
                ax.set_title(label)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
        except:
            print('Error plottong covariance matrix')

    def plot_eig(self):
        """Plot eigenvalues & eigenvectors."""
        if hasattr(self, 'eig_val'):
            plt.clf()
            ax = plt.subplot(121)
            ax.plot(self.eig_val/self.eig_val.sum())
            ax.plot(np.cumsum(self.eig_val/self.eig_val.sum()))
            # ax.semilogy(basey=10, nonposy='clip')
            ax.set_xlabel('eigen number')
            ax.set_ylabel(r'$\lambda_i / \sum \lambda$')

            ax = plt.subplot(122)
            for i in range(len(self.eig_val)):
                ax.plot(self.eig_vec[i, :]/(self.eig_vec**2).sum(axis=0)**0.5)
            # ax.semilogy(basey=10, nonposy='clip')
            ax.set_xlabel('separation bin')
            ax.set_ylabel(r'$E_i / (\sum E_i^2)^{0.5}$')
            plt.show()


def ra_shift(ra):
    """Shift RAs by 180 deg to avoid wraparound."""
    ras = np.array(ra) - 180
    neg = ras < 0
    ras[neg] += 360
    return ras

def make_rect_mask(limits=[180, 200, 0, 20], rect_mask='mask.ply'):
    """Make simple rectangular mangle mask."""
    if limits[0] > limits[1]:
        limits[:2] = ra_shift(limits[:2])
        print('RA limits changed to', limits[:2])

    with open(rect_mask, 'w') as f:
        print(*limits, file=f)

    # Snap to polygon format (default weight=1)
    cmd = f'$MANGLE_DIR/bin/snap -ir1 {rect_mask} {rect_mask}'
    subprocess.call(cmd, shell=True)


def wcalc(galcat, rancat, bins, jack=-1):
    """w(theta) calc."""

    ra, dec = galcat.sample(jack=jack)
    rar, decr = rancat.sample(jack=jack)

    # Number of threads to use
    nthreads = galcat.nthreads

    # Auto pairs counts in RR
    RR_counts = Corrfunc.mocks.DDtheta_mocks(1, nthreads, bins, rar, decr)

    # Auto pair counts in DD
    DD_counts = Corrfunc.mocks.DDtheta_mocks(1, nthreads, bins, ra, dec)

    # Cross pair counts in DR
    DR_counts = Corrfunc.mocks.DDtheta_mocks(0, nthreads, bins, ra, dec, RA2=rar, DEC2=decr)

    # All the pair counts are done, get the angular correlation function
    wtheta = Corrfunc.utils.convert_3d_counts_to_cf(
        len(ra), len(ra), len(rar),  len(rar),
        DD_counts, DR_counts, DR_counts, RR_counts)

    return wtheta, DD_counts, DR_counts, RR_counts


def wcounts(ra, dec, bins, info, outfile, ra2=None, dec2=None, nthreads=1,
            output_thetaavg=False):
    """w(theta) counts."""

    if ra2 is None:
        autocorr = 1
    else:
        autocorr = 0

    counts = Corrfunc.mocks.DDtheta_mocks(
        autocorr, nthreads, bins, ra, dec,
        RA2=ra2, DEC2=dec2, output_thetaavg=output_thetaavg)

    pickle.dump((info, counts), open(outfile, 'wb'))


def xir_counts(x, y, z, bins, info, outfile, x2=None, y2=None, z2=None,
               nthreads=1, output_ravg=False):
    """xi(r) counts."""

    if x2 is None:
        autocorr = 1
    else:
        autocorr = 0

    counts = Corrfunc.theory.DD(
        autocorr, nthreads, bins, x, y, z, periodic=False,
        X2=x2, Y2=y2, Z2=z2, output_ravg=output_ravg)

    pickle.dump((info, counts), open(outfile, 'wb'))


def xir_counts_bf(x, y, z, bins, info, outfile, x2=None, y2=None, z2=None,
               nthreads=1, output_ravg=False):
    """xi(r) counts brute force."""

    lgbins = np.log10(bins)
    logmin = lgbins[0]
    logstep = lgbins[1] - lgbins[0]
    nbin = len(bins) - 1
    npair = np.zeros(nbin)
    
    # Find which coord has largest range
    coords = pd.DataFrame([x, y, z]).transpose()
    dr = np.array([np.max(coords[i]) - np.min(coords[i]) for i in range(3)])
    imax = np.argmax(dr)
    coords = coords.sort_values(imax)
    N = len(coords)
    
    if x2 is None:
        # auto correlation, forward pair counts only
        for i in range(N-1):
            j = i
            dd = 0
            while j < N-1 and dd < bins[-1]:
                j += 1
                dd = coords[imax][i] - coords[imax][j]
                if dd < bins[-1]:
                    sepsq = 0
                    for ic in range(3):
                        sepsq += (coords[ic][j] - coords[ic][i])**2
                    lgs = 0.5*math.log10(sepsq)
                    ibin = int((lgs - logmin)/logstep)
                    if (ibin >= 0) and (ibin < nbin):
                        npair[ibin] += 1

    else:
        # cross correlation, count pairs in both directions
        coords2 = pd.DataFrame([x2, y2, z2]).transpose()
        coords2 = coords2.sort_values(imax)
        N2 = len(coords2)
        jstart = -1
        for i in range(N-1):
            j = jstart
            dd = 2*bins[-1]
            while j < N2-1 and dd > bins[-1]:
                j += 1
                dd = coords[imax][i] - coords2[imax][j]
            j -= 1
            jstart = j
            dd = 0
            while j < N2-1 and dd < bins[-1]:
                j += 1
                dd = coords[imax][i] - coords2[imax][j]
                if dd < bins[-1]:
                    sepsq = 0
                    for ic in range(3):
                        sepsq += (coords[ic][i] - coords2[ic][j])**2
                    lgs = 0.5*math.log10(sepsq)
                    ibin = int((lgs - logmin)/logstep)
                    if (ibin >= 0) and (ibin < nbin):
                        npair[ibin] += 1

        
    counts = np.array([bins[:-1], bins[1:], npair],
                      dtype=[('rmin', '<f4'), ('rmax', '<f4'), ('npairs', '<i4')])
    pickle.dump((info, counts), open(outfile, 'wb'))


def xir_calc(galcat, rancat, bins, jack=-1):
    """xi(r) calc."""

    x, y, z = galcat.sample(jack=jack, cart=True)
    xr, yr, zr = rancat.sample(jack=jack, cart=True)

    # Number of threads to use
    nthreads = galcat.nthreads

    # Auto pairs counts in RR
    RR_counts = Corrfunc.theory.DD(1, nthreads, bins, xr, yr, zr, periodic=False)

    # Auto pair counts in DD
    DD_counts = Corrfunc.theory.DD(1, nthreads, bins, x, y, z, periodic=False)

    # Cross pair counts in DR
    DR_counts = Corrfunc.theory.DD(0, nthreads, bins, x, y, z, periodic=False,
                              X2=xr, Y2=yr, Z2=zr)

    # All the pair counts are done, get the angular correlation function
    xi = Corrfunc.utils.convert_3d_counts_to_cf(
        len(x), len(x), len(xr),  len(xr),
        DD_counts, DR_counts, DR_counts, RR_counts)

    return xi, DD_counts, DR_counts, RR_counts


def wsamp(galcat, rancat, bins, jack=-1):
    """w(theta) calc for multiple sub-samples."""

    wsub = np.zeros((galcat.nsub, len(bins)-1))
    rar, decr = rancat.sample(jack)

    # Number of threads to use
    nthreads = galcat.nthreads
    print('nthreads = ', nthreads)

    # Auto pairs counts in RR: same for every sub-sample
    RR_counts = Corrfunc.mocks.DDtheta_mocks(1, nthreads, bins, rar, decr)

    for isub in range(galcat.nsub):
        ra, dec = galcat.sample(jack, isub)
        
        # Auto pair counts in DD
        DD_counts = Corrfunc.mocks.DDtheta_mocks(1, nthreads, bins, ra, dec)

        # Cross pair counts in DR
        DR_counts = Corrfunc.mocks.DDtheta_mocks(0, nthreads, bins, ra, dec, RA2=rar, DEC2=decr)

        # All the pair counts are done, get the angular correlation function
        wsub[isub, :] = Corrfunc.utils.convert_3d_counts_to_cf(
            len(ra), len(ra), len(rar),  len(rar),
            DD_counts, DR_counts, DR_counts, RR_counts)

    return wsub, DD_counts, DR_counts, RR_counts


def w_jack(galcat, rancat, bins):
    """w(theta) with JK errors."""

    w, DD_counts, DR_counts, RR_counts = wcalc(galcat, rancat, bins)
    wj = np.zeros((galcat.njack, len(bins)-1))
    for jack in range(galcat.njack):
        wj[jack, :], _, _, _ = wcalc(galcat, rancat, bins, jack=jack)
    w_err = (galcat.njack-1)**0.5 * np.std(wj, axis=0)
    return w, w_err, DD_counts, DR_counts, RR_counts


def xir_jack(galcat, rancat, bins):
    """xi(r) with JK errors."""

    xi, DD_counts, DR_counts, RR_counts = xir_calc(galcat, rancat, bins)
    xij = np.zeros((galcat.njack, len(bins)-1))
    for jack in range(galcat.njack):
        xij[jack, :], _, _, _ = xir_calc(galcat, rancat, bins, jack=jack)
    xi_err = (galcat.njack-1)**0.5 * np.std(xij, axis=0)
    return {'bins': bins, 'xi': xi, 'xi_err': xi_err, 'xij': xij,
            'DD': DD_counts, 'DR': DR_counts, 'RR': RR_counts}


def w_jack_sub(galcat, rancat, bins):
    """w(theta) with JK errors for sub-samples."""

    w, DD_counts, DR_counts, RR_counts = wsamp(galcat, rancat, bins)
    wj = np.zeros((galcat.njack, galcat.nsub, len(bins)-1))
    for jack in range(galcat.njack):
        wj[jack, :, :], _, _, _ = wsamp(galcat, rancat, bins, jack=jack)
    w_err = (galcat.njack-1)**0.5 * np.std(wj, axis=0)
    return w, w_err, wj, DD_counts, DR_counts, RR_counts


def Nofz(z, N0, z0, alpha, beta):
    """Blake+2013 fit to N(z)."""
    return N0 * (z/z0)**alpha * np.exp(-(z/z0)**beta)


def Nofz_plot(N0=1, z0=0.5, alpha=1, beta=1, z=np.linspace(0.0, 1.0, 51)):
    h = Nofz(z, N0, z0, alpha, beta)
    plt.clf()
    plt.plot(z, h)
    plt.show()


def w_hsc_ud():
    """w(theta) for HSC_UD in mag bins."""
    w_hsc(gal_file='gal_mask_ud.fits', ran_file='ran_mask_ud.fits',
          out_file='w_hsc_ud.pkl', magbins = np.linspace(20, 26, 7),
          nra=4, ndec=4, tmin=0.01, tmax=1, nbins=20)


def w_hsc(gal_file='gal_mask.fits', ran_file='ran_mask.fits',
          out_file='w_hsc_wide.pkl', magbins = np.linspace(20, 25, 6),
          nra=4, ndec=4, tmin=0.01, tmax=1, nbins=20, nthreads=2, ranfrac=0.1):
    """w(theta) for HSC in mag bins."""
    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    lbl = [f'{magbins[i]} <= mag < {magbins[i+1]}' for i in range(len(magbins)-1)]
    t = Table.read(gal_file)
    ra, dec, mag = t['ra'], t['dec'], t['imag']
    sub = np.zeros(len(ra), dtype='int8')
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        sub[sel] = imag
    galcat = Cat(ra, dec, sub=sub, nthreads=nthreads)

    t = Table.read(ran_file)
    if ranfrac < 1:
        nran = int(ranfrac*len(t))
        sel = rng.choice(len(t), nran, replace=False)
        ra, dec = t['ra'][sel], t['dec'][sel]
    else:
        ra, dec = t['ra'], t['dec']
    rancat = Cat(ra, dec)

    limits = (np.min(ra), np.max(ra), np.min(dec), np.max(dec))
    galcat.assign_jk(limits, nra, ndec)
    rancat.assign_jk(limits, nra, ndec)

    print(galcat.nobj, rancat.nobj, 'galaxies and randoms')

    w, w_err, wj, DD_counts, DR_counts, RR_counts = w_jack_sub(galcat, rancat, bins)
    pickle.dump((tcen, w, w_err, wj, DD_counts, DR_counts, RR_counts, lbl), open(out_file, 'wb'))




def ic_pl(A, gamma, tmax, tcen, RR_counts):
        """Returns estimated integral constraint for power law w(theta)
        truncated at tmax."""
        w_mod = np.zeros(len(tcen))
        sel = tcen < tmax
        w_mod[sel] = A * tcen[sel]**(1-gamma)
        ic = (RR_counts['npairs'] * w_mod).sum() / RR_counts['npairs'].sum()
        return ic

def ic_binned(tcen, w, RR_counts):
        """Returns estimated integral constraint for binned w(theta)."""
        ic = (RR_counts['npairs'] * w).sum() / RR_counts['npairs'].sum()
        return ic

def rplot(infiles):
    """Plot N(r) histograms."""
    plt.clf()
    for infile in infiles:
        (ra, dec, r) = np.loadtxt(infile)       
        plt.hist(r, histtype='step', label=infile)
    plt.legend()
    plt.xlabel(r'$r$ [Mpc/h]')
    plt.ylabel(r'$N(r)$')
    plt.show()


def radec_plot(infiles, s=1):
    """Plot ra, dec distributions."""
    plt.clf()
    for infile in infiles:
        (ra, dec, r) = np.loadtxt(infile)       
        plt.scatter(ra, dec, s=s, label=infile)
    plt.legend()
    plt.xlabel(r'RA')
    plt.ylabel(r'Dec')
    plt.show()


def wplot(infiles):
    """Plot w(theta) results."""
    plt.clf()
    for infile in infiles:
        (tcen, w, w_err) = pickle.load(open(infile, 'rb'))       
        plt.errorbar(tcen, w, w_err, label=infile)
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ [degrees]')
    plt.ylabel(r'w($\theta$)')
    plt.show()


def wfit(tcen, w, w_err, tfitlo=0.01, tfithi=0.05, p0=[1e-2, 1.7]):
    """Power-law fit to w(theta)."""

    def power_law(theta, A, gamma):
        """Power law w(theta) = A theta**(1-gamma)."""
        return A * theta**(1-gamma)
    
    sel = (tcen >= tfitlo) * (tcen < tfithi) * np.isfinite(w)
    popt, pcov = scipy.optimize.curve_fit(
                power_law, tcen[sel], w[sel], p0=p0, sigma=w_err[sel])
    return popt, pcov


def wplot_samp(infile='w_hsc.pkl', tfitlo=0.01, tfithi=0.05, p0=[1e-2, 1.7],
               t2lo=0.2, t2hi=0.5, p02=[1e-3, 2.7], ic_corr='None',
               tmax_ic=0.1, niter=3):
    """Plot w(theta) results for multiple samples."""

    def power_law(theta, A, gamma):
        """Power law w(theta) = A theta**(1-gamma)."""
        return A * theta**(1-gamma)
    
    (tcen, w, w_err, wj, DD_counts, DR_counts, RR_counts, lbl) = pickle.load(open(infile, 'rb'))       
    ic = np.zeros(len(lbl))
    for iter in range(niter):
        wcorr = (w.T + ic).T
        plt.clf()
        ax = plt.gca()
        for isub in range(len(lbl)):
            sel = (tcen >= tfitlo) * (tcen < tfithi) * np.isfinite(wcorr[isub, :])
            popt, pcov = scipy.optimize.curve_fit(
                power_law, tcen[sel], wcorr[isub, sel], p0=p0,
                sigma=w_err[isub, sel])
            if ic_corr == 'pl':
                ic[isub] = popt[0] * ic_pl(*popt, tmax_ic, tcen, RR_counts)
            if ic_corr == 'binned':
                ic[isub] = popt[0] * ic_binned(tcen[sel], wcorr[isub, sel], RR_counts[sel])
            if ic_corr == 'None':
                ic[isub] = 0
            color = next(ax._get_lines.prop_cycler)['color']
            plt.errorbar(tcen, wcorr[isub, :], w_err[isub, :], fmt='o', color=color,
                         label=lbl[isub] + f' {popt[0]:4.3f} {popt[1]:4.3f}')
            plt.plot(tcen[sel], power_law(tcen[sel], *popt), color=color)
            if t2lo:
                sel = (tcen >= t2lo) * (tcen < t2hi)
                popt2, pcov2 = scipy.optimize.curve_fit(
                    power_law, tcen[sel], wcorr[isub, sel], p0=p02,
                    sigma=w_err[isub, sel])
                plt.plot(tcen[sel], power_law(tcen[sel], *popt2), color=color)
                print(lbl[isub], popt, popt2, ic[isub])
            else:
                print(lbl[isub], popt, ic[isub])
                
#         plt.axis((0.01, 0.5, 0.01, 1))
        plt.loglog()
        plt.legend()
        plt.xlabel(r'$\theta$ [degrees]')
        plt.ylabel(r'w($\theta$)')
        plt.show()


# def wplot_scale_old(cosmo, infile='w_hsc.pkl', gamma1=1.7, gamma2=4, r0=5.0, eps=0,
#                 maglims=np.linspace(20, 26, 7), Mmin=-24, Mmax=-12,
#                 alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
#                 phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67]):
#     """Plot w(theta) results for multiple samples with Limber scaling."""

#     (tcen, w, w_err, wj, DD_counts, DR_counts, RR_counts, lbl) = pickle.load(open(infile, 'rb'))       
#     plt.clf()
#     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=1)
#     fig.set_size_inches(6, 4)
#     fig.subplots_adjust(hspace=0, wspace=0)
#     mmin = maglims[0]
#     mmax = maglims[1]
#     selref = util.SelectionFunction(
#         cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar, mmin=mmin, mmax=mmax,
#         Mmin=Mmin, Mmax=Mmax, nksamp=0,
#         kcoeffs=kcoeffs)
#     for isub in range(len(lbl)):
#         mmin = maglims[isub]
#         mmax = maglims[isub+1]
#         selfn = util.SelectionFunction(
#             cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
#             mmin=mmin, mmax=mmax, Mmin=Mmin, Mmax=Mmax, nksamp=0, kcoeffs=kcoeffs)
#         dlgt, dlgw = limber.limber_scale(
#             cosmo, selfn, selref, gamma1=gamma1, gamma2=gamma2, r0=r0, eps=eps)
#         print(dlgt, dlgw)
#         axes[0].errorbar(tcen, w[isub, :], w_err[isub, :], fmt='o', 
#                          label=lbl[isub])
#         axes[1].errorbar(tcen * 10**dlgt, w[isub, :] * 10**dlgw,
#                          w_err[isub, :] * 10**dlgw, fmt='o', label=lbl[isub])
#     axes[0].loglog()
#     axes[1].loglog()
#     axes[0].legend()
#     axes[0].set_xlabel(r'$\theta$ [degrees]')
#     axes[1].set_xlabel(r'$\theta$ [degrees]')
#     axes[0].set_ylabel(r'w($\theta$)')
#     plt.show()


def plot_sel(cosmo, wcorrs, gamma1=1.7, gamma2=4, r0=5.0, eps=0,
                Mmin=-24, Mmax=-12,
                alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
                phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67],
                plot_sel=0):
    """Plot selection functions."""

    plt.clf()
    fig, ax = plt.subplots(1, 1, num=1)
    for wcorr in wcorrs:
        selfn = util.SelectionFunction(
            cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
            mlo=wcorr.mlo, mhi=wcorr.mhi, Mmin=Mmin, Mmax=Mmax,
            nksamp=0, kcoeffs=kcoeffs)
        selfn.plot_dNdz(ax)
    plt.show()

    
def wplot_scale(cosmo, wcorrs, gamma1=1.7, gamma2=4, r0=5.0, eps=0,
                Mmin=-24, Mmax=-12,
                alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
                phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67],
                plot_sel=0):
    """Plot w(theta) results for multiple samples with Limber scaling."""

    plt.clf()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=1)
    if plot_sel:
        fig2, ax2 = plt.subplots(1, 1, num=2)
    fig.set_size_inches(6, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    selref = util.SelectionFunction(
        cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
        mlo=wcorrs[0].mlo, mhi=wcorrs[0].mhi,
        Mmin=Mmin, Mmax=Mmax, nksamp=0, kcoeffs=kcoeffs)
    for wcorr in wcorrs:
        selfn = util.SelectionFunction(
            cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
            mlo=wcorr.mlo, mhi=wcorr.mhi, Mmin=Mmin, Mmax=Mmax,
            nksamp=0, kcoeffs=kcoeffs)
        if plot_sel:
            selfn.plot_Nz(ax2)
        dlgt, dlgw = limber.limber_scale(
            cosmo, selfn, selref, gamma1=gamma1, gamma2=gamma2, r0=r0, eps=eps)
        print(dlgt, dlgw)
        lbl = f'm = [{wcorr.mlo:4.2f}, {wcorr.mhi:4.2f}]'
        wcorr.plot(ax=axes[0], label=lbl)
        wcorr.plot(ax=axes[1], tscale=10**dlgt, wscale=10**dlgw, label=lbl)
    axes[0].loglog()
    axes[1].loglog()
    axes[0].legend()
    axes[0].set_xlabel(r'$\theta$ [degrees]')
    axes[1].set_xlabel(r'$\theta$ [degrees]')
    axes[0].set_ylabel(r'w($\theta$)')
    plt.show()


def w_pred(gamma1=1.7, gamma2=4, r0=5.0, eps=0, mbins = np.linspace(15, 20, 6),
                Mmin=-24, Mmax=-12, theta=np.logspace(-2, 1, 20),
                alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
                phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67],
                plot_sel=0):
    """Plot predicted w(theta) from Maddox+1996 eqn 31."""

    # Flagship2 cosomology, converting to h=1 units
    h = 1
    Om0 = 0.319
    cosmo = util.CosmoLookup(h, Om0)
    
    plt.clf()
    for mlo in mbins[:-1]:
        mhi = mlo + 1
        selfn = util.SelectionFunction(
            cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
            mlo=mlo, mhi=mhi, Mmin=Mmin, Mmax=Mmax,
            nksamp=0, kcoeffs=kcoeffs)
        w = []
        for t in theta:
            wp = limber.w_lum(cosmo, selfn, t, 0.5*(mlo+mhi))
            w.append(wp)
            print(mlo, mhi, t, wp)
                 
        plt.plot(theta, w, label=f'{mlo:2.0f} < m < {mhi:2.0f}')
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ [degrees]')
    plt.ylabel(r'w($\theta$)')
    plt.show()


def mcmc(cosmo, wcorrs, gamma1=1.7, gamma2=4, r0=5.0, eps=0,
         Mmin=-24, Mmax=-12,
         alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
         phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67],
         nstep=[50, 100]):
    """MCMC exploration of LF, xi(r) parameter space that searches for minimum
    in chi2 residuals after Limber scaling.  Note that scaling is independent
    of phi*_0 and r_0."""

    def lnprob(x, verbose=0):
        """Returns -0.5*chi^2 residual between scaled w(theta) estimates."""

        alpha, Mstar, phistar[1] = x[:2], x[2:4], x[4]
        gamma1, gamma2, eps = x[5], x[6], x[7]

        # Parameter limits
        bad = 1
        if (-5 < alpha[0] < 5 and -5 < alpha[1] < 5 and
            -25 < Mstar[0] < -15 and  -5 < Mstar[1] < 5 and
            -5 < phistar[1] < 5 and
            1 < gamma1 < 2 and 2 < gamma2 < 8 and -5 < eps < 5):
            bad = 0
        if bad:
            return -np.inf
           
        chi2 = 0
        selref = util.SelectionFunction(
            cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
            mlo=wcorrs[0].mlo, mhi=wcorrs[0].mhi,
            Mmin=Mmin, Mmax=Mmax, nksamp=0, kcoeffs=kcoeffs)
        for wcorr in wcorrs[1:]:
            selfn = util.SelectionFunction(
                cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
                mlo=wcorr.mlo, mhi=wcorr.mhi, Mmin=Mmin, Mmax=Mmax,
                nksamp=0, kcoeffs=kcoeffs)
            dlgt, dlgw = limber.limber_scale(
                cosmo, selfn, selref, gamma1=gamma1, gamma2=gamma2, r0=r0, eps=eps)
            use = wcorr.err > 0
            w_interp = 10**(lgw_interp(wcorr.lgsep[use] + dlgt) - dlgw)
            chi2 += np.sum(((wcorr.est_corr()[use] - w_interp)/wcorr.err[use])**2)
            if verbose:
                print(((wcorr.est_corr()[use] - w_interp)/wcorr.err[use])**2)
#                pdb.set_trace()
        if np.isnan(chi2):
            return -np.inf
        if verbose:
            print(x, chi2)
        return -0.5*chi2

    use = (wcorrs[0].est_corr() > 0) * (wcorrs[0].sep < 7)
    lgw_interp = scipy.interpolate.interp1d(
        wcorrs[0].lgsep[use], np.log10(wcorrs[0].est_corr()[use]),
        fill_value='extrapolate')
#     pdb.set_trace()
    
    ndim = 8
    nwalkers = 5*ndim
    x0 = [*alpha, *Mstar, phistar[1], gamma1, gamma2, eps]
    print('lnprob (default params) =', lnprob(x0, verbose=1))
    
    pos = np.tile(x0, (nwalkers, 1)) + 1e-5*np.random.randn(nwalkers, ndim)
#     with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    for ichain in range(len(nstep)):
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep[ichain])
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        # try:
        #     print("Autocorrelation time:", sampler.get_autocorr_time())
        # except AutocorrError:
        #     pass

        fig, axes = plt.subplots(ndim, figsize=(8, 8), sharex=True)
        samples = sampler.get_chain()
        labels = ['alpha_0', 'alpha_z', 'M*_0', 'M*_z', 'phi*_z',
                  'gamma1', 'gamma2', 'eps']
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.show()
        plt.savefig(f'chain{ichain}.png')
        
        res = np.array(np.percentile(sampler.flatchain, [50, 16, 84], axis=0))
    alpha, Mstar, phistar[1] = res[0, :2], res[0, 2:4], res[0, 4]
    gamma1, gamma2, eps = res[0, 5], res[0, 6], res[0, 7]
    print('Median parameter values')
    print('alpha, M*, phi* =', alpha, Mstar, phistar)
    print('gamma1, gamma2, r0, eps = ', gamma1, gamma2, r0, eps)
    print('lnprob (median params) =', lnprob(res[0, :], verbose=1))

    plt.clf()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(6, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    selref = util.SelectionFunction(
        cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
        mlo=wcorrs[0].mlo, mhi=wcorrs[0].mhi,
        Mmin=Mmin, Mmax=Mmax, nksamp=0, kcoeffs=kcoeffs)
    for wcorr in wcorrs:
        selfn = util.SelectionFunction(
            cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
            mlo=wcorr.mlo, mhi=wcorr.mhi, Mmin=Mmin, Mmax=Mmax,
            nksamp=0, kcoeffs=kcoeffs)
        dlgt, dlgw = limber.limber_scale(
            cosmo, selfn, selref, gamma1=gamma1, gamma2=gamma2, r0=r0, eps=eps)
        print(dlgt, dlgw)
        lbl = f'm = [{wcorr.mlo:4.2f}, {wcorr.mhi:4.2f}]'
        wcorr.plot(ax=axes[0], label=lbl)
        wcorr.plot(ax=axes[1], tscale=10**dlgt, wscale=10**dlgw, label=lbl)
    lgtp = np.linspace(-2, 2, 50)
    axes[1].plot(10**lgtp, 10**lgw_interp(lgtp), c='b')
    axes[0].loglog()
    axes[1].loglog()
    axes[0].legend()
    axes[0].set_xlabel(r'$\theta$ [degrees]')
    axes[1].set_xlabel(r'$\theta$ [degrees]')
    axes[0].set_ylabel(r'w($\theta$)')
    plt.show()
    plt.savefig('w_mcmc.png')

    flat_samples = sampler.get_chain(flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()
    plt.savefig('corner.png')

def xi_plot_samp(infile='w_hsc.pkl', rfitlo=0.01, rfithi=0.05, p0=[5, 1.7],
                 rmax_ic=0.1, ic_corr='None', niter=3):
    """Plot xi(r) results for multiple samples."""

    def power_law(r, r0, gamma):
        """Power law xi(r) = (r0/r)**gamma."""
        return (r0/r)**gamma
    
    def xi_ev(z, A, eps):
        """Amplitude evolution of xi(r) = A * (1+z)**(-(3 + eps)."""
        return A * (1+z)**(-(3 + eps))
    
    xi_dict_list = pickle.load(open(infile, 'rb'))
    nsamp = len(xi_dict_list)
    ic = np.zeros(nsamp)
    for iter in range(niter):
        plt.clf()
        ax = plt.gca()
        r0, r0_err, gamma, gamma_err = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
        for isub in range(nsamp):
            xi_dict = xi_dict_list[isub]
            print(xi_dict.keys())
            rcen = 10**(0.5*(np.log10(xi_dict['DD']['rmin']) + np.log10(xi_dict['DD']['rmax'])))
            xi = xi_dict['xi']
            xi_err = xi_dict['xi_err']
            RR = xi_dict['RR']
            xicorr = xi + ic[isub]
            sel = (rcen >= rfitlo) * (rcen < rfithi) * np.isfinite(xicorr)
            try:
                popt, pcov = scipy.optimize.curve_fit(
                power_law, rcen[sel], xicorr[sel], p0=p0, sigma=xi_err[sel])
            except RuntimeError:
                popt = [0, 0]    
            r0[isub], gamma[isub] = popt[0], popt[1]
            r0_err[isub], gamma_err[isub] = pcov[0,0]**0.5, pcov[1,1]**0.5
            if ic_corr == 'pl':
                ic[isub] = popt[0] * ic_pl(*popt, rmax_ic, rcen, RR)
            if ic_corr == 'binned':
                ic[isub] = popt[0] * ic_binned(rcen[sel], xicorr[sel], RR[sel])
            if ic_corr == 'None':
                ic[isub] = 0
            color = next(ax._get_lines.prop_cycler)['color']
            plt.errorbar(rcen, xicorr, xi_err, fmt='o', color=color,
                         label=xi_dict['lbl'] + f' {popt[0]:3.2f} {popt[1]:3.2f}')
            plt.plot(rcen[sel], power_law(rcen[sel], *popt), color=color)
            print(xi_dict['lbl'], popt, pcov, ic[isub])
                
#         plt.axis((0.01, 0.5, 0.01, 1))
        plt.loglog()
        plt.legend()
        plt.xlabel(r'$r$ [Mpc/h]')
        plt.ylabel(r'$\xi(r)$')
        plt.show()

        zcen = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
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
        axes[1].set_ylabel('gamma')
        axes[1].set_xlabel('z')
        plt.show()

        
def transpose(infile, outfile):
    """Transpose arrays in .pkl files so that theta bin corresponds
    to last dimension."""
    (tcen, w, w_err, wj, DD_counts, DR_counts, RR_counts, lbl) = pickle.load(open(infile, 'rb'))       
    pickle.dump((tcen, w.T, w_err.T, wj.T, DD_counts, DR_counts, RR_counts, lbl), open(outfile, 'wb'))


def kpoly(z, clr, p):
    """K-corrections as a 2d polynomial, constrained to pass through (zref, kref)."""

    zref = 0.1
    kref = -2.5*math.log10(1+zref)
    
def poly_test():
    """Test polynomial fitting."""
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 5, 8, 15, 28])
    p = Polynomial.fit(x, y, deg=[1, 2], window=[0, 5])
    yfit = p(x)
    plt.scatter(x, y)
    plt.plot(x, yfit)
    plt.show()


def sel_test():
    cosmo = util.CosmoLookup()
    selfn = util.SelectionFunction(cosmo, plot=True)
    
def corrfunc_test(nran=1000, tmin=0.01, tmax=10, nbins=20):
    """Random-random pair count test."""

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    ra = 20*rng.random(nran)
    dec = 20*rng.random(nran)
    counts = Corrfunc.mocks.DDtheta_mocks(1, 1, bins, ra, dec)
    print(counts)

    counts = Corrfunc.mocks.DDtheta_mocks(1, 1, bins, ra.astype('float32'), dec.astype('float32'))
    print(counts)

    theta = np.linspace(0, 180, 181)
    plt.clf()
    plt.plot(theta, np.sin(np.deg2rad(theta)/2)**2)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\sin^2(\theta/2)$')
    plt.show()
    
