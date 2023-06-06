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
import scipy.special
import subprocess
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

rng = default_rng()

class CosmoLookup(object):
    """Cosmology look-up tables."""
    
    def __init__(self, h=1, Om0=0.319, zbins=np.linspace(0.01, 1, 100)):
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
    
    def __init__(self, cosmo, alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
                 phistar=[9.81e-4, -3.24e-4], mlo=12, mhi=20,
                 Mmin=-24, Mmax=-12, ktable=None, nksamp=100,
                 kcoeffs=[0, 0.65, 0.08],
                 sax=None, nax=None):
            
        self._z = cosmo._z
        self._sel, self._Nz = np.zeros(cosmo._nz), np.zeros(cosmo._nz)
        self._zmean, self._Mmean, self._kmean = np.zeros(cosmo._nz), np.zeros(cosmo._nz), np.zeros(cosmo._nz)
        dmod = cosmo.distmod(self._z)
        if kcoeffs is not None:
            kp = Polynomial(kcoeffs, domain=[0, 1], window=[0, 1])
            k = kp(self._z)
        for iz in range(cosmo._nz):
            a1 = alpha[0] + alpha[1]*self._z[iz] + 1
            Mst = Mstar[0] + Mstar[1]*self._z[iz]
            pst = phistar[0] + phistar[1]*self._z[iz]
            Lmin, Lmax = 10**(0.4*(Mst - Mmin)), 10**(0.4*(Mst - Mmax))
            gamd = mpmath.gammainc(a1, Lmax, Lmin)
            if nksamp > 0:
                # Average gamma function for luminosity limits corresponding to
                # k-corrections for nksamp galaxies closest in redshift
                gamn = 0
                dz = np.abs(ktable['z'] - self._z[iz])
                sort = np.argsort(dz)
                for ik in range(nksamp):
                    k = ktable['k'][sort][ik]
                    Mlo = np.clip(mlo - dmod - k, Mmin, Mmax)
                    Mhi = np.clip(mhi - dmod - k, Mmin, Mmax)
                    Llo, Lhi = 10**(0.4*(Mst - Mlo[iz])), 10**(0.4*(Mst - Mhi[iz]))
                    gamn += mpmath.gammainc(a1, Lhi, Llo)
                gamn /= nksamp
                self._zmean[iz] = np.mean(ktable['z'][sort][:nksamp])
                self._Mmean[iz] = np.mean(ktable['M'][sort][:nksamp])
                self._kmean[iz] = np.mean(ktable['k'][sort][:nksamp])
            else:
                # Use polynomial fit to K(z)
                Mlo = np.clip(mlo - dmod - k[iz], Mmin, Mmax)
                Mhi = np.clip(mhi - dmod - k[iz], Mmin, Mmax)
                Llo, Lhi = 10**(0.4*(Mst - Mlo[iz])), 10**(0.4*(Mst - Mhi[iz]))
                gamn = mpmath.gammainc(a1, Lhi, Llo)
                
            self._sel[iz] = gamn / gamd
            self._Nz[iz] = pst * gamn * cosmo._differential_comoving_volume[iz]
        # self._dNdz = np.gradient(self._Nz, self._z[1]-self._z[0])
        # pdb.set_trace()
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
        ax.plot(self._z, self._Nz)


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


