# Limber scaling for angular correlation functions

import glob
import math
import matplotlib.pyplot as plt
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


def w_lum(cosmo, selfn, theta, m, xi_pars, plotint=0, pdf=None, plot_den=0):
    """Returns w(theta) for lum-dependent xi(r) correlation length and index
    by evaluating Maddox+1996 eqn 31 using comoving coords."""
    
    def denfun(x):
        """Denominator of Maddox+ eqn 31."""
        z = cosmo.z_at_dist(x)
        return 1e-5 * x**2 * selfn.sel(z) / (1+z)**3

    def ximod(r, z, M):
        """Luminosity-dependent evolving xi(r) model."""
        # Mz = np.ma.vstack((M.flatten(), z.flatten()))
        r0 = xi_dict['r0_fun'](xi_dict['r0_pars'], M, z)
        gamma = xi_dict['gam_fun'](xi_dict['gam_pars'], M, z)
        return (r0/r)**gamma

    def K(z):
        return 0
    
    def xifun(x1, x2):
        """Numerator of Maddox+ eqn 31."""
        z = cosmo.z_at_dist(0.5*(x1+x2))
        z1 = cosmo.z_at_dist(x1)
        z2 = cosmo.z_at_dist(x2)
        a = 1/(1+z)
        x12 = (x1**2 + x2**2 - 2*x1*x2*np.cos(np.deg2rad(theta)))**0.5
        M = m - cosmo.distmod(z) - K(z)
        return 1e-10 * x1**2/(1+z1)**3 * x2**2/(1+z2)**3 * selfn.sel(z1) * selfn.sel(z2) * ximod(x12, z, M)

    xmin, xmax = cosmo._x[0], cosmo._x[-1]
    x = np.linspace(xmin, xmax, 100)
    xi_dict = pickle.load(open(xi_pars, 'rb'))


    if pdf:
        if plot_den:
            fig = plt.figure()
            plt.plot(x, denfun(x))
            plt.xlabel('x [Mpc]')
            plt.ylabel('denfun')
            plt.title(f'mag = {m:4.2f}')
            pdf.savefig(plt.gcf().number)
            plt.close(fig)
        
        fig = plt.figure()
        x1, x2 = np.meshgrid(x, x)
        plt.imshow(xifun(x1, x2), extent=(xmin, xmax, xmax, xmin))
        plt.xlabel('x2 [Mpc]')
        plt.ylabel('x1 [Mpc]')
        plt.title(f'mag = {m:4.2f}, theta = {theta:4.3f}')
        plt.colorbar()
        pdf.savefig(plt.gcf().number)
        plt.close(fig)


    quad = scipy.integrate.quad(denfun, xmin, xmax, epsrel=0.01)
    denom = quad[0]**2
    print('denom', m, theta, quad)

    # num = scipy.integrate.dblquad(xifun, rmin, rmax, lambda x: max(rmin, x-100),
    #                               lambda x: min(rmax, x+100), epsabs=1e5, epsrel=0.01)[0]
    quad = scipy.integrate.dblquad(xifun, xmin, xmax, xmin, xmax, epsrel=0.01)
    num = quad[0]
    print('num', m, theta, quad)

    return num/denom


def w_lum_Nz(cosmo, selfn, theta, m, xi_pars, kpoly, plotint=0, pdf=None, plot_den=0):
    """Returns w(theta) for lum-dependent xi(r) correlation length and index
    by evaluating Maddox+1996 eqn 31 using number counts N(z)."""
    
    def denfun(z):
        """Denominator of Maddox+ eqn 31."""
        return selfn.Nz(z)

    def ximod(r, z, M):
        """Luminosity-dependent evolving xi(r) model."""
        # Mz = np.ma.vstack((M.flatten(), z.flatten()))
        r0 = np.clip(xi_dict['r0_fun'](xi_dict['r0_pars'], M, z), 1, 10)
        gamma = np.clip(xi_dict['gam_fun'](xi_dict['gam_pars'], M, z), 1, 3)
        xi = (r0/r)**gamma
        return xi

    def ximod_2d(r, z, M):
        """2d interpolation of r0(M, z) and gamma(M, z)."""
        r0 = xi_dict['r0_lin_interp'](M, z)
        bad =  np.isnan(r0)
        r0[bad] = xi_dict['r0_nn_interp'](M, z)[bad]
        gamma = xi_dict['gamma_lin_interp'](M, z)
        bad = np.isnan(gamma)
        gamma[bad] = xi_dict['gamma_nn_interp'](M, z)[bad]
        return (r0/r)**gamma

    def xifun(z1, z2):
        """Numerator of Maddox+ eqn 31."""
        z = 0.5*(z1+z2)
        x1 = cosmo.dc(z1)
        x2 = cosmo.dc(z2)
        x12 = (x1**2 + x2**2 - 2*x1*x2*np.cos(np.deg2rad(theta)))**0.5
        M = m - cosmo.distmod(z) - kpoly(z)
        return denfun(z1) * denfun(z2) * ximod(x12, z, M)

    zmin, zmax = cosmo._z[0], cosmo._z[-1]
    z = np.linspace(zmin, zmax, 100)
    xi_dict = pickle.load(open(xi_pars, 'rb'))


    if pdf:
        if plot_den:
            fig = plt.figure()
            plt.plot(z, denfun(z))
            plt.xlabel('Redshift')
            plt.ylabel('denfun')
            plt.title(f'mag = {m:4.2f}')
            pdf.savefig(plt.gcf().number)
            plt.close(fig)
        
        fig = plt.figure()
        z1, z2 = np.meshgrid(z, z)
        plt.imshow(xifun(z1, z2), extent=(zmin, zmax, zmax, zmin), norm='log')
        plt.xlabel('z2')
        plt.ylabel('z1')
        plt.title(f'mag = {m:4.2f}, theta = {theta:4.3f}')
        plt.colorbar()
        pdf.savefig(plt.gcf().number)
        plt.close(fig)


    quad = scipy.integrate.quad(denfun, zmin, zmax, epsrel=0.01)
    denom = quad[0]**2
    print('denom', m, theta, quad)

    # num = scipy.integrate.dblquad(xifun, rmin, rmax, lambda x: max(rmin, x-100),
    #                               lambda x: min(rmax, x+100), epsabs=1e5, epsrel=0.01)[0]
    quad = scipy.integrate.dblquad(xifun, zmin, zmax, zmin, zmax, epsrel=0.01)
    num = quad[0]
    if np.isnan(num):
        pdb.set_trace()
    print('num', m, theta, quad)

    return num/denom
