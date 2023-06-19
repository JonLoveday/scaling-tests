# Limber scaling for angular correlation functions

import glob
import math
import mpmath
import numpy as np
import pdb
import pickle
import pylab as plt
import scipy.special
import subprocess
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
import util

def w_a_sel(cosmo, selfn, gamma=1.7, r0=5.0, eps=0, plotint=0):
    """Returns w(theta) amplitude for power-law xi(r) and given sel fns."""
    
    def denfun(x):
        """Denominator of Maddox+ eqn 33b."""
        z = cosmo.z_at_dist(x)
        a = 1/(1+z)
        return x**2 * a**3 * selfn.sel(z)

    def xifun(x):
        """Numerator of Maddox+ eqn 33b."""
        z = cosmo.z_at_dist(x)
        a = 1/(1+z)
        return x**(5-gamma) * a**6 * selfn.sel(z)**2 * (1 +z)**(gamma - 3 - eps)


    gfac = (math.pi**0.5 * scipy.special.gamma((gamma-1)/2) * r0**gamma /
            scipy.special.gamma(gamma/2))
    xmin, xmax = cosmo.xmin, cosmo.xmax
    if plotint:
        xp = np.linspace(xmin, xmax, 100)
        plt.clf()
        plt.plot(xp, denfun(xp))
        plt.xlabel('x')
        plt.ylabel('denfun')
        plt.show()
        plt.clf()
        plt.plot(xp, xifun(xp))
        plt.xlabel('x')
        plt.ylabel('xifun')
        plt.show()
            
    res = scipy.integrate.quad(xifun, xmin, xmax, full_output=0,
                               epsabs=1e3, epsrel=1e-3)
    if len(res) > 3:
        pdb.set_trace()
    num = res[0]
    res = scipy.integrate.quad(denfun, xmin, xmax, full_output=0,
                               epsabs=1e3, epsrel=1e-3)
    if len(res) > 3:
        pdb.set_trace()
    den = res[0]**2
    B = num/den
    A = gfac * B
    return A


def w_a(cosmo, selfn, gamma=1.7, r0=5.0, eps=0, plotint=0):
    """Returns w(theta) amplitude for power-law xi(r) and given sel fns.
    This version uses N(z) rather than S(z)."""
    
    def denfun(z):
        """Denominator of Maddox+ eqn 35a."""
        return selfn.Nz(z)

    def xifun(z):
        """Numerator of Maddox+ eqn 35a."""
        x = cosmo.dc(z)
        return x**(1-gamma) * selfn.Nz(z)**2 * (1 +z)**(gamma - 3 - eps)/cosmo.dxdz(z)

    zmin, zmax = cosmo._z[0], cosmo._z[-1]
    gfac = (math.pi**0.5 * scipy.special.gamma((gamma-1)/2) * r0**gamma /
            scipy.special.gamma(gamma/2))
    if plotint:
        zp = np.linspace(zmin, zmax, 100)
        plt.clf()
        plt.plot(zp, denfun(zp))
        plt.xlabel('z')
        plt.ylabel('denfun')
        plt.show()
        plt.clf()
        plt.plot(zp, xifun(zp))
        plt.xlabel('z')
        plt.ylabel('xifun')
        plt.show()
            
    res = scipy.integrate.quad(xifun, zmin, zmax, full_output=0,
                               epsabs=1e3, epsrel=1e-3)
    if len(res) > 3:
        pdb.set_trace()
    num = res[0]
    res = scipy.integrate.quad(denfun, zmin, zmax, full_output=0,
                               epsabs=1e3, epsrel=1e-3)
    if len(res) > 3:
        pdb.set_trace()
    den = res[0]**2
    B = num/den
    A = gfac * B
    return A


def limber_scale(cosmo, selfn, selref, gamma1=1.7, gamma2=4, r0=5.0, eps=0):
    """Scale w(theta) to reference depth for two power-law model."""
    
    A = w_a(cosmo, selfn, gamma=gamma1, r0=r0, eps=eps)
    B = w_a(cosmo, selfn, gamma=gamma2, r0=r0, eps=eps)
    Aref = w_a(cosmo, selref, gamma=gamma1, r0=r0, eps=eps)
    Bref = w_a(cosmo, selref, gamma=gamma2, r0=r0, eps=eps)
    dlgA, dlgB = np.log10(A/Aref), np.log10(B/Bref)
    dlgt = (dlgA - dlgB) / (gamma2 - gamma1)
    dlgw = (gamma1-1)*dlgt - dlgA
    return dlgt, dlgw


def limber_scale_mult(gamma1=1.7, gamma2=4, r0=5.0, eps=0,
                 maglims=np.linspace(20, 26, 7),
                 zbins=np.linspace(0.0, 0.5, 100), plotsel=0, plotint=1):
    """Scale w(theta) to reference depth for two power-law model."""
    
    cosmo = util.CosmoLookup(zbins=zbins, plot=0)
    nmag = len(maglims)-1
    A, B = np.zeros(nmag), np.zeros(nmag)
    if plotsel:
        plt.clf()
    print('mlo  mhi  A  B   dlgt  dlgw')
    for im in range(nmag):
        mlo, mhi = maglims[im], maglims[im+1]
        cosmo.set_selfn(mlo=mlo, mhi=mhi, plot=plotsel)
        A[im] = w_a(cosmo, gamma=gamma1, r0=r0, eps=eps, plotint=plotint)
        B[im] = w_a(cosmo, gamma=gamma2, r0=r0, eps=eps, plotint=plotint)
        if im > 0:
            dlgA, dlgB = np.log10(A[im]/A[0]), np.log10(B[im]/B[0])
            dlgt = (dlgA - dlgB) / (gamma2 - gamma1)
            dlgw = dlgA - (1-gamma1)*dlgt
        else:
            dlgt, dlgw = 0, 0
        print(f'{mlo} {mhi} {A[im]:5.4f} {B[im]:5.4f} {dlgt:5.4f} {dlgw:5.4f}')
    if plotsel:
        plt.xlabel('z')
        plt.ylabel('S(z)')
        plt.legend()
        plt.show()
    return dlgt, dlgw


def w_lum(cosmo, selfn, theta, m, gamma=1.7, r0=5.0, eps=0, plotint=0,
          rmin=1, rmax=3000):
    """Returns w(theta) for lum-dependent xi(r) correlation length and index
    by evaluating Maddox+1996 eqn 31."""
    
    def denfun(r):
        """Denominator of Maddox+ eqn 31."""
        z = cosmo.z_at_pdist(r)
        return r**2 * selfn.sel(z)

    def ximod(r, z, M):
        """Luminosity-dependent evolving xi(r) model.  No lum dependence yet."""
        return (r0/r)**gamma * (1+z)**-(3+eps)

    def K(z):
        return 3*z
    
    def xifun(r1, r2):
        """Numerator of Maddox+ eqn 31."""
        z = cosmo.z_at_pdist(0.5*(r1+r2))
        z1 = cosmo.z_at_pdist(r1)
        z2 = cosmo.z_at_pdist(r2)
        a = 1/(1+z)
        r12 = (r1**2 + r2**2 - 2*r1*r2*np.cos(np.deg2rad(theta)))**0.5/a
        M = m - cosmo.distmod(z) - K(z)
        return r1**2 * r2**2 * selfn.sel(z1) * selfn.sel(z2) * ximod(r12, z, M)

    denom = scipy.integrate.quad(denfun, rmin, rmax, epsabs=1e5, epsrel=0.01)[0]**2
    num = scipy.integrate.dblquad(xifun, rmin, rmax, lambda x: max(rmin, x-100),
                                  lambda x: min(rmax, x+100), epsabs=1e5, epsrel=0.01)[0]

    return num/denom
