# Utilities for LF and clustering codes

import glob
import math
import mpmath
import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import default_rng
import pdb
import pickle
import pylab as plt
from scipy import interpolate
import scipy.special
import subprocess
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

rng = default_rng()

class CosmoLookup(object):
    """Cosmology look-up tables."""
    
    def __init__(self, h=1, Om0=0.319, zbins=np.linspace(0.0001, 1, 100)):
        cosmo = FlatLambdaCDM(H0=100*h, Om0=Om0)
        self._z = zbins
        self._nz = self._z.size
        self._x = cosmo.comoving_distance(self._z).value
        self._distmod = cosmo.distmod(self._z).value
        self._comoving_volume = cosmo.comoving_volume(self._z).value
        self._differential_comoving_volume = cosmo.differential_comoving_volume(self._z).value
        self._dxdz = np.gradient(self._x, zbins[1]-zbins[0])
        self.xmin, self.xmax = self._x[0], self._x[-1]

    def z_at_dist(self, x):
        return np.interp(x, self._x, self._z)

    def z_at_distmod(self, dm):
        return np.interp(dm, self._distmod, self._z)

    def z_at_pdist(self, x):
        return np.interp(x, self._x/(1 + self._z), self._z)

    def dc(self, z):
        return np.interp(z, self._z, self._x)

    def dxdz(self, z):
        return np.interp(z, self._z, self._dxdz)

    def distmod(self, z):
        return np.interp(z, self._z, self._distmod)

    def comoving_volume(self, z):
        return np.interp(z, self._z, self._comoving_volume)

    def differential_comoving_volume(self, z):
        return np.interp(z, self._z, self._differential_comoving_volume)


class SelectionFunction(object):
    """Selection function and N(z) look-up tables."""
    
    def __init__(self, cosmo, lf_pars, mlo=12, mhi=20,
                 solid_angle=1, dz=1, interp=0,
                 sax=None, nax=None):

        def lf_interp(M):
            """Interpolation of lg phi(M, z).  Limit to 0.2,
            to avoid wacky extrapolation to very faint magnitudes."""
            return min(0.2, 10**interpolate.interpn(
                (Mcen, zmean), lgphi, [M, self._z[iz]],
                bounds_error=False, fill_value=None))

        kpoly, lf_dict, Mcen, zmean, lgphi = pickle.load(open(lf_pars, 'rb'))
        zfun = lf_dict['zfun']
        
        self._z = cosmo._z
        self._sel, self._Nz = np.zeros(cosmo._nz), np.zeros(cosmo._nz)
        self._zmean, self._Mmean, self._kmean = np.zeros(cosmo._nz), np.zeros(cosmo._nz), np.zeros(cosmo._nz)
        dmod = cosmo.distmod(self._z)
        k = kpoly(self._z)
        for iz in range(cosmo._nz):
            Mlo = mlo - dmod[iz] - k[iz]
            Mhi = mhi - dmod[iz] - k[iz]
            if interp:
                gam = scipy.integrate.quad(lf_interp, Mlo, Mhi,
                                           epsabs=0.01, epsrel=1e-3)[0]
            else:
                a1 = zfun(self._z[iz], lf_dict['alpha']) + 1
                Mst = zfun(self._z[iz], lf_dict['Mstar'])
                pst = 10**zfun(self._z[iz], lf_dict['lgphistar'])
                Llo, Lhi = 10**(0.4*(Mst - Mlo)), 10**(0.4*(Mst - Mhi))
                gam = pst * mpmath.gammainc(a1, Lhi, Llo)
                
            self._sel[iz] = gam * (1+self._z[iz])**3
            self._Nz[iz] = gam * cosmo._differential_comoving_volume[iz] * solid_angle * dz
            # if self._Nz[iz] > 2e5:
            #     pdb.set_trace()
        # self._dNdz = np.gradient(self._Nz, self._z[1]-self._z[0])
        if sax:
            sax.plot(self._z, self._sel,
                     label=f'{mlo} < m < {mhi}, {Mmin} < M < {Mmax}')
        if nax:
            nax.plot(self._z, self._Nz,
                     label=f'{mlo} < m < {mhi}, {Mmin} < M < {Mmax}')
        
    def sel(self, z):
        """Linear interpolation of selection function"""
        return np.interp(z, self._z, self._sel)

    def Nz(self, z):
        """Linear interpolation of N(z)."""
        return np.interp(z, self._z, self._Nz)

    # def dNdz(self, z):
    #     """Linear interpolation of N(z)."""
    #     return np.interp(z, self._z, self._dNdz)

    def plot_Nz(self, ax, **kwargs):
        """Plot N(z)."""
        ax.plot(self._z, self._Nz, **kwargs)


class NzCounts(object):
    """Observed N(z) counts."""
    
    def __init__(self, spline):
        self._spline = spline

    def __call__(z):
        return self._spline(z)


def ran_dist(x, p, nran):
    """Generate nran random points according to distribution p(x)"""

    if np.amin(p) < 0:
        print('ran_dist warning: clipping negative pdf values to zero.')
        p = np.clip(p, 0, None)
    cp = np.cumsum(p)
    y = (cp - cp[0]) / (cp[-1] - cp[0])
    r = rng.random(nran)
    return np.interp(r, y, x)


def ran_fun(f, xmin, xmax, nran, args=None, nbin=1000):
    """Generate nran random points according to pdf f(x)"""

    x = np.linspace(xmin, xmax, nbin)
    if args is not None:
        p = f(x, *args)
    else:
        p = f(x)
    return ran_dist(x, p, nran)


def Nofz(z, N0, z0, alpha, beta):
    """Blake+2013 fit to N(z)."""
    return N0 * (z/z0)**alpha * np.exp(-(z/z0)**beta)


def Nofz_plot(N0=1, z0=0.5, alpha=1, beta=1, z=np.linspace(0.0, 1.0, 51)):
    h = Nofz(z, N0, z0, alpha, beta)
    plt.clf()
    plt.plot(z, h)
    plt.show()


