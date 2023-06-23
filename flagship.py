# Clustering/LF measurements for Euclid flagship

import glob
import math
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import numpy.ma as ma
from numpy.polynomial import Polynomial
from numpy.random import default_rng
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
import pymangle

import calc_kcor
import limber
import util
import wcorr

ln10 = math.log(10)
rng = default_rng()

# Flagship2 cosomology, converting to h=1 units
h = 1
Om0 = 0.319
cosmo = util.CosmoLookup(h, Om0)

def wcounts(infile='14516.fits', mask_file='mask.ply', out_pref='w_mag/',
            limits=(180, 200, 0, 20), nran=100000, nra=4, ndec=4,
            tmin=0.01, tmax=10, nbins=20,
            magbins=np.linspace(15, 20, 6), plots=1):
    """Angular pair counts in mag bins."""

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    t = Table.read(infile)
    sel = t['hmag'] < magbins[-1]
    t = t[sel]
    ra, dec, mag = t['ra_gal'], t['dec_gal'], t['hmag']
    mmean = np.zeros(len(magbins)-1)
    sub = np.zeros(len(ra), dtype='int8')
    if plots:
        plt.clf()
        fig, axes = plt.subplots(5, 1, sharex=True, num=1)
        fig.set_size_inches(5, 6)
        fig.subplots_adjust(hspace=0, wspace=0)
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= mag) * (mag < magbins[imag+1])
        sub[sel] = imag
        mmean[imag] = np.mean(mag[sel])
        print(imag, mmean[imag], np.percentile(t['habs'][sel], (5, 50, 95)))
        if plots:
            ax = axes[imag]
            ax.hist(t['habs'][sel], bins=np.linspace(-24, -15, 19))
            ax.text(0.7, 0.8, rf'm = {magbins[imag]:3.1f}-{magbins[imag+1]:3.1f}',
                    transform=ax.transAxes)

    if plots:
        plt.xlabel(r'$M_r$')
        plt.show()
    galcat = wcorr.Cat(ra, dec, sub=sub)
    galcat.assign_jk(limits, nra, ndec)

    mask = pymangle.Mangle(mask_file)
    ra, dec = mask.genrand_range(nran, *limits)
    rancat = wcorr.Cat(ra.astype('float64'), dec.astype('float64'))
    rancat.assign_jk(limits, nra, ndec)

    print(galcat.nobj, rancat.nobj, 'galaxies and randoms')

    njack = nra*ndec
    for ijack in range(njack+1):
        rcoords = rancat.sample(ijack)
        plt.clf()
        plt.scatter(rcoords[0], rcoords[1], s=0.1)
        plt.show()
        info = {'Jack': ijack, 'Nran': len(rcoords[0]), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_J{ijack}.pkl'
        pool.apply_async(wcorr.wcounts, args=(*rcoords, bins, info, outfile))
        for imag in range(len(magbins) - 1):
            print(ijack, imag)
            mlo, mhi = magbins[imag], magbins[imag+1]
            gcoords = galcat.sample(ijack, sub=imag)
            info = {'Jack': ijack, 'mlo': mlo, 'mhi': mhi, 'mmean': mmean[imag],
                    'Ngal': len(gcoords[0]), 'Nran': len(rcoords[0]),
                    'bins': bins, 'tcen': tcen}
            outfile = f'{out_pref}GG_J{ijack}_m{imag}.pkl'
            pool.apply_async(wcorr.wcounts,
                             args=(*gcoords, bins, info, outfile))
            outfile = f'{out_pref}GR_J{ijack}_m{imag}.pkl'
            pool.apply_async(wcorr.wcounts,
                             args=(*gcoords, bins, info,  outfile, *rcoords))
    pool.close()
    pool.join()


def xir_counts(infile='14516.fits', mask_file='mask.ply',
         Mbins=np.linspace(-26, -12, 8),
         zbins=np.linspace(0, 1, 6), limits=(180, 200, 0, 20),
         ranfac=1, nra=3, ndec=3, rbins=np.logspace(-1, 2, 16),
         randist='shuffle', out_pref='xir_z/', multi=True):
    """Real-space pair counts in magnitude and redshift bins."""

    nM = len(Mbins) - 1
    nz = len(zbins) - 1
    rcen = 10**(0.5*np.diff(np.log10(rbins)) + np.log10(rbins[:-1]))
    if multi:
        ncpu = mp.cpu_count()
        pool = mp.Pool(ncpu)
        print('Using', ncpu, 'CPUs')
    
    t = Table.read(infile)
    ra, dec, Mag = t['ra_gal'], t['dec_gal'], t['habs']
    redshift = t['true_redshift_gal']
    r = cosmo.dc(redshift)

    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        for im in range(nM):
            Mlo, Mhi = Mbins[im], Mbins[im+1]
            sel = ((zlo <= redshift) * (redshift < zhi) *
                   (Mlo <= Mag) * (Mag < Mhi))
            Mmean = np.mean(Mag[sel])
            zmean = np.mean(redshift[sel])
            galcat = wcorr.Cat(ra[sel], dec[sel], r=r[sel])
            galcat.assign_jk(limits, nra, ndec)
            galcat.gen_cart()
            ngal = len(ra[sel])
            if ngal > 0:
                nran = int(ranfac*ngal)
                mask = pymangle.Mangle(mask_file)
                rar, decr = mask.genrand_range(nran, *limits)
                rr = rng.choice(r[sel], nran, replace=True)
                rancat = wcorr.Cat(rar.astype('float64'), decr.astype('float64'), r=rr)
                rancat.assign_jk(limits, nra, ndec)
                rancat.gen_cart()

                print(f'Redshift bin {iz}, Mag bin {im} ngal = {ngal}, nran = {nran}')
                for jack in range(galcat.njack+1):
                    x, y, z = galcat.sample(jack=jack, cart=True)
                    xr, yr, zr = rancat.sample(jack=jack, cart=True)
                    info = {'jack': jack, 'Ngal': ngal, 'Nran': nran,
                            'rbins': rbins, 'rcen': rcen,
                            'Mlo': Mlo, 'Mhi': Mhi, 'zlo': zlo, 'zhi': zhi,
                            'Mmean': Mmean, 'zmean': zmean}
                    outgg = f'{out_pref}GG_j{jack}_z{iz}_M{im}.pkl'
                    outgr = f'{out_pref}GR_j{jack}_z{iz}_M{im}.pkl'
                    outrr = f'{out_pref}RR_j{jack}_z{iz}_M{im}.pkl'
                    if multi:
                        pool.apply_async(wcorr.xir_counts,
                                         args=(x, y, z, rbins, info, outgg))
                        pool.apply_async(wcorr.xir_counts,
                                         args=(xr, yr, zr, rbins, info, outrr))
                        pool.apply_async(wcorr.xir_counts,
                                         args=(x, y, z, rbins, info, outgr, xr, yr, zr))
                    else:
                        wcorr.xir_counts(x, y, z, rbins, info, outgg)
                        wcorr.xir_counts(xr, yr, zr, rbins, info, outrr)
                        wcorr.xir_counts(x, y, z, rbins, info, outgr, xr, yr, zr)
    if multi:
        pool.close()
        pool.join()


def hists(infile='14516.fits', Mbins=np.linspace(-26, -16, 41),
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
        M = t['habs'][sel]
        print(iz, np.percentile(M, (5, 50, 95)))
        ax = axes[iz]
        ax.semilogy()
        ax.hist(M, Mbins)
        ax.text(0.05, 0.8, rf'z = {zbins[iz]:3.1f}-{zbins[iz+1]:3.1f}',
            transform=ax.transAxes)
    plt.xlabel(r'$M_h$')
    plt.show()


def w_plot(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7], prefix='w_mag/',
           avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7,
           alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
           phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67], ic_corr=0):
    """w(theta) from angular pair counts in mag bins."""

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
        if ic_corr:
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
                      r0=r0, eps=eps, alpha=alpha, Mstar=Mstar,
                      phistar=phistar, kcoeffs=kcoeffs)

def w_plot_pred(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7], prefix='w_mag/',
           avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7,
           Mmin=-24, Mmax=-12, alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
           phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67], ic_corr=0,
                pltfile='intergrand_plots.pdf'):
    """Plot observed and predicted w(theta) in mag bins."""

    if pltfile:
        pdf = matplotlib.backends.backend_pdf.PdfPages(pltfilename)
    else:
        pdf = None
    fig = plt.figure()
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
        if ic_corr:
            corr.ic_calc(fit_range, p0, 5)
        corr_slices.append(corr)
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax, color=color, label=f"m = [{info['mlo']}, {info['mhi']}]")

        selfn = util.SelectionFunction(
            cosmo, alpha=alpha, Mstar=Mstar, phistar=phistar,
            mlo=info['mlo'], mhi=info['mhi'], Mmin=Mmin, Mmax=Mmax,
            nksamp=0, kcoeffs=kcoeffs)
        w = []
        for t in corr.sep:
            wp = limber.w_lum(cosmo, selfn, t, 0.5*(info['mlo']+info['mhi']),
                              pdf=pdf)
            w.append(wp)
        ax.plot(corr.sep, w, color=color)
        # popt, pcov = corr.fit_w(fit_range, p0, ax, color)
        # print(popt, pcov)

    if pdf:
        pdf.close()
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()
    plt.close(fig)


def mcmc(nmag=5, njack=16, fit_range=[0.01, 1], p0=[0.05, 1.7], prefix='w_mag/',
         avgcounts=False, gamma1=1.67, gamma2=3.8, r0=6.0, eps=-2.7,
         alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
         phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67],
         nstep=[500, 10000]):
    """MCMC fit of LF and xi(r) params to optimise Limber scaling."""

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

    wcorr.mcmc(cosmo, corr_slices, gamma1=gamma1, gamma2=gamma2,
               r0=r0, eps=eps, alpha=alpha, Mstar=Mstar,
               phistar=phistar, kcoeffs=kcoeffs, nstep=nstep)

def xir_z_plot(nz=5, njack=8, fit_range=[0.1, 20], p0=[5, 1.7], prefix='xir_z/',
             avgcounts=False):
    """xi(r) from pair counts in redshift bins."""

    def xi_ev(z, A, eps):
        """Amplitude evolution of xi(r) = A * (1+z)**(-(3 + eps)."""
        return A * (1+z)**(-(3 + eps))
    
    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    zcen, r0, r0_err, gamma, gamma_err = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
    for iz in range(nz):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{prefix}GG_j{ijack}_z{iz}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GR_j{ijack}_z{iz}.pkl'
            (info, DR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}RR_j{ijack}_z{iz}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            corrs.append(
                wcorr.Corr1d(info['Ngal'], info['Nran'],
                             DD_counts, DR_counts, RR_counts))
        corr = corrs[0]
        corr.err = np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        color = next(ax._get_lines.prop_cycler)['color']
        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
        print(popt, pcov)
        corr.plot(ax, color=color,
                  label=rf"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}], $r_0 = {popt[0]:3.2f} \pm {pcov[0][0]**0.5:3.2f}$, $\gamma = {popt[1]:3.2f} \pm {pcov[1][1]**0.5:3.2f}$")
        zcen[iz] = 0.5*(info['zlo'] + info['zhi'])
        r0[iz], gamma[iz] = popt[0], popt[1]
        r0_err[iz], gamma_err[iz] = pcov[0,0]**0.5, pcov[1,1]**0.5
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$r$ [Mpc/h]')
    plt.ylabel(r'$\xi(r)$')
    plt.show()

    gav = np.average(gamma, weights=gamma_err**-2)
    gerr = np.sum(gamma_err**-2)**-0.5
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
    axes[1].text(0.5, 0.8,
                 rf'$\gamma = {gav:5.3f} \pm {gerr:5.3f}$',
                 transform=axes[1].transAxes)
    axes[1].set_ylabel('gamma')
    axes[1].set_xlabel('z')
    plt.show()


def xir_M_z_plot(nm=6, nz=5, njack=9, fit_range=[0.1, 20], p0=[5, 1.7],
                 prefix='xir_z/', avgcounts=False):
    """xi(r) from pair counts in Magnitude and redshift bins."""

    def xi_ev(z, A, eps):
        """Amplitude evolution of xi(r) = A * (1+z)**(-(3 + eps)."""
        return A * (1+z)**(-(3 + eps))
    
    def r0_fun(Mz, r0_0, am, Mstar, eps):
        """r0 = r0_0 * (1+z)**(-(3 + eps) + am * L/L*."""
        M, z = Mz[0, :], Mz[1, :]
        return r0_0 * (1+z)**(-(3 + eps)) + am * 10**(0.4*(Mstar - M)) 
    
    # def gam_fun(Mz, gam_0, am, Mstar, eps):
    #     """gamma = gam_0 * (1+z)**(-(3 + eps) * am * L/L*."""
    #     M, z = Mz[0, :], Mz[1, :]
    #     return gam_0 * (1+z)**(-(3 + eps)) + am * 10**(0.4*(Mstar - M)) 
    
    # def r0_fun(Mz, r0_0, a1, a2, eps):
    #     """r0 = r0_0 * (1+z)**(-(3 + eps) + a1*M + a2*M^2."""
    #     M, z = Mz[0, :], Mz[1, :]
    #     return r0_0 * (1+z)**(-(3 + eps)) + a1*M + a2*M**2
    
    def gam_fun(Mz, gam_0, a1, a2, eps):
        """gamma = gam_0 * (1+z)**(-(3 + eps) * a1*M + a2*M**2."""
        M, z = Mz[0, :], Mz[1, :]
        return gam_0 * (1+z)**(-(3 + eps)) + a1 + a2*M
    
    plt.clf()
    fig, axes = plt.subplots(1, nz, sharex=True, sharey=True, num=1)
    fig.set_size_inches(15, 5)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].set_ylabel(r'$\xi(r)$')
    axes[2].set_xlabel(r'$r$ [Mpc/h]')
    corr_slices = []
    Mmean, zmean, r0, r0_err, gamma, gamma_err = np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz)), np.zeros((nm, nz))
    zlo, zhi =  np.zeros(nz), np.zeros(nz)
    for iz in range(nz):
        ax = axes[iz]
        for im in range(nm):
            try:
                corrs = []
                for ijack in range(njack+1):
                    infile = f'{prefix}GG_j{ijack}_z{iz}_M{im}.pkl'
                    if ijack == 0:
                        (info, DD_counts) = pickle.load(open(infile, 'rb'))
                    else:
                        (_, DD_counts) = pickle.load(open(infile, 'rb'))
                    infile = f'{prefix}GR_j{ijack}_z{iz}_M{im}.pkl'
                    (_, DR_counts) = pickle.load(open(infile, 'rb'))
                    infile = f'{prefix}RR_j{ijack}_z{iz}_M{im}.pkl'
                    (_, RR_counts) = pickle.load(open(infile, 'rb'))
                    corrs.append(
                        wcorr.Corr1d(info['Ngal'], info['Nran'],
                                     DD_counts, DR_counts, RR_counts))
                if info['Ngal'] > 100:
                    corr = corrs[0]
                    corr.err = np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
                    color = next(ax._get_lines.prop_cycler)['color']
                    popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
                    ax.errorbar(corr.sep, corr.est, corr.err, color=color, fmt='o',
                                label=rf"M = [{info['Mlo']:3.1f}, {info['Mhi']:3.1f}]")
                    # corr.plot(ax, color=color,
                    #           label=rf"M = [{info['Mlo']:3.1f}, {info['Mhi']:3.1f}], $r_0 = {popt[0]:3.2f} \pm {pcov[0][0]**0.5:3.2f}$, $\gamma = {popt[1]:3.2f} \pm {pcov[1][1]**0.5:3.2f}$")
                    Mmean[im, iz] = info['Mmean']
                    zmean[im, iz] = info['zmean']
                    r0[im, iz], gamma[im, iz] = popt[0], popt[1]
                    r0_err[im, iz], gamma_err[im, iz] = pcov[0,0]**0.5, pcov[1,1]**0.5
                    print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'],
                          info['Ngal'], r0[im, iz], gamma[im, iz])
                    # if iz == 0 and im == 0:
                    #     pdb.set_trace()
                else:
                    print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'],
                          info['Ngal'])
            except FileNotFoundError:
                print(info['zlo'], info['zhi'], info['Mlo'], info['Mhi'], ' no data')

        ax.legend()
        zlo[iz], zhi[iz] = info['zlo'], info['zhi']
        ax.text(0.5, 0.9, rf"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}]",
                transform=ax.transAxes)
    plt.loglog()
    plt.show()

    mask = Mmean > -5
    Mmean = ma.masked_array(Mmean, mask)
    zmean = ma.masked_array(zmean, mask)
    gamma = ma.masked_array(gamma, mask)
    gamma_err = ma.masked_array(gamma_err, mask)
    r0 = ma.masked_array(r0, mask)
    r0_err = ma.masked_array(r0_err, mask)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=2)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    scatter = axes[0].scatter(Mmean, zmean, c=gamma)
    plt.colorbar(scatter, ax=axes[0], label='gamma', location='top')
    axes[0].set_xlabel('M')
    axes[0].set_ylabel('z')
    scatter = axes[1].scatter(Mmean, zmean, c=r0)
    plt.colorbar(scatter, ax=axes[1], label='r0', location='top')
    axes[1].set_xlabel('M')
    # axes[1].set_yticklabels([])
    plt.show()

    ok = ~Mmean.mask
    Mz = np.ma.vstack((Mmean[ok].flatten(), zmean[ok].flatten()))
    r0_popt, r0_pcov = scipy.optimize.curve_fit(
        r0_fun, Mz, r0[ok].flatten(), p0=(4, 0.1, -21, 0), sigma=r0_err[ok].flatten())
    gam_popt, gam_pcov = scipy.optimize.curve_fit(
        gam_fun, Mz, gamma[ok].flatten(), p0=(4, 0.1, -21, 0), sigma=gamma_err[ok].flatten())
    print('r0 fit pars:', r0_popt)
    print('gamma fit pars:', gam_popt)

    fig, axes = plt.subplots(2, nz, sharex=True, sharey='row', num=3)
    fig.set_size_inches(16, 8)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0, 0].set_ylabel(r'$r_0$')
    axes[1, 0].set_ylabel(r'$\gamma$')
    axes[1, 2].set_xlabel(r'$M_r$')
    for iy in range(2):
        for iz in range(nz):
            # pdb.set_trace()
            ax = axes[iy, iz]
            if iy == 0:
                ax.text(0.5, 0.9, rf"z = [{zlo[iz]:3.1f}, {zhi[iz]:3.1f}]",
                transform=ax.transAxes)
                ax.errorbar(Mmean[:, iz], r0[:, iz], r0_err[:, iz])
                ax.plot(Mmean[:, iz],
                         r0_fun(np.vstack((Mmean[:, iz], zmean[:, iz])), *r0_popt))
            else:
                ax.errorbar(Mmean[:, iz], gamma[:, iz], gamma_err[:, iz])
                ax.plot(Mmean[:, iz],
                         gam_fun(np.vstack((Mmean[:, iz], zmean[:, iz])), *gam_popt))
    plt.show()


def kcorr(infile='14516.fits', M_bins=np.linspace(-24, -16, 5),
          nplot=100000):
    """Empirically determine flagship K-corrections using k = m - M - DM."""

    t = Table.read(infile)
    sel = t['true_redshift_gal'] < 1
    t = t[sel]

    m, M, z = t['hmag'], t['habs'], t['true_redshift_gal']
    dm = cosmo.distmod(z)
    k = m - M - dm
    # kc = calc_kcor.calc_kcor('r', z, 'g - r', g-r)
    sel = np.isfinite(k)
    z, k, M = z[sel], k[sel], M[sel]
    
    # Polynomial fit to K(z), constrained to pass through origin
    zp = np.linspace(0, 1, 21)
    plt.clf()
    ip = rng.choice(len(z), nplot)

    plt.scatter(z[ip], k[ip], s=0.01, c=M[ip])
    # for deg in [[1,], [1,2], [1,2,3], [1,2,3,4]]:
    #     p = Polynomial.fit(z, k, deg=deg, domain=[0, 1], window=[0, 1])
    #     kfit = p(zp)
    #     plt.plot(zp, kfit, label=str(deg))
    #     print(p.coef)
    deg = [1,2]
    p = Polynomial.fit(z, k, deg=deg, domain=[0, 1], window=[0, 1])
    kfit = p(zp)
    plt.plot(zp, kfit, label='All')
    print(p.coef)
    for im in range(len(M_bins)-1):
        sel = (M_bins[im] <= M) * (M < M_bins[im+1])
        p = Polynomial.fit(z[sel], k[sel], deg=deg, domain=[0, 1], window=[0, 1])
        kfit = p(zp)
        plt.plot(zp, kfit, label=f'[{M_bins[im]},  {M_bins[im+1]}]')
        print(p.coef)
    plt.colorbar(label=r'$M_h$')
    plt.legend()
    plt.ylabel('Kcorr')
    # plt.subplot(212)
    # plt.scatter(z, k-kc, c=g-r, s=0.01)
    # # plt.scatter(z, k, s=0.01)
    # plt.colorbar(label='g-r')
    # plt.xlabel('Redshift')
    # plt.ylabel('Delta Kcorr')
    plt.show()


def lf(infile='14516.fits', zbins=np.linspace(0.0, 1.0, 6),
       magbins=np.linspace(-26, -18, 33), p0=[-1.45, -21, 1e-3],
       bounds=([-1.451, -23, 1e-4], [-1.449, -19, 1e-2]), zfit=0):
    """Flagship h-band LF in redshift slices, assuming it's volume-limited 
    to M_h ~ -18 at z = 1.0."""

    def Schechter(M, alpha, Mstar, phistar):
        L = 10**(0.4*(Mstar-M))
        schec = 0.4*ln10*phistar*L**(alpha+1)*np.exp(-L)
        return schec

    t = Table.read(infile)
    Mcen = magbins[:-1] + np.diff(magbins)
    area_frac = 20*20*(math.pi/180)**2/(4*math.pi)
    nz = len(zbins)-1
    zmean = np.zeros(nz)
    alpha, Mstar, phistar = np.zeros(nz), np.zeros(nz), np.zeros(nz)
    alpha_err, Mstar_err, phistar_err = np.zeros(nz), np.zeros(nz), np.zeros(nz)
    plt.clf()
    ax = plt.gca()
    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        vol = area_frac * (cosmo.comoving_volume(zhi) - cosmo.comoving_volume(zlo))
        sel = (zlo <= t['true_redshift_gal']) * (t['true_redshift_gal'] < zhi)
        zmean[iz] = np.mean(t['true_redshift_gal'][sel])
#        zmean[iz] = np.log10(1 + zmean[iz])
        hist, edges = np.histogram(t['habs'][sel], magbins)
        phi = hist/vol
        phi_err = phi/hist**0.5
        use = phi > 0
        popt, pcov = scipy.optimize.curve_fit(
            Schechter, Mcen[use], phi[use], p0=p0, sigma=phi_err[use])
#            absolute_sigma=True)
#            bounds=bounds)
        alpha[iz], Mstar[iz], phistar[iz] = popt[0], popt[1], popt[2]
        alpha_err[iz], Mstar_err[iz], phistar_err[iz] = pcov[0][0]**0.5, pcov[1][1]**0.5, pcov[2][2]**0.5, 
        lbl = f'z = {zlo:3.1f} - {zhi:3.1f}; {popt[0]:3.2f} {popt[1]:3.2f} {popt[2]:3.2e}'
        color = next(ax._get_lines.prop_cycler)['color']
        plt.errorbar(Mcen, phi, phi_err, fmt='o', color=color, label=lbl)
        plt.plot(Mcen, Schechter(Mcen, *popt), color=color)
#        print(phi, phi_err)
    plt.semilogy(base=10)
    plt.ylim(1e-8, 1e-2)
    plt.xlabel(r'$M_h$')
    plt.ylabel(r'$\Phi(M_h)$')
    plt.legend()
    plt.show()

    # Schechter parameters as function of z (zfit=0) or lg(1+z) if zfit=1
    zp = np.array([zbins[0], zbins[-1]])
    zlbl = 'z'
    if zfit:
        zmean = np.log10(1+zmean)
        zp = np.log10(1+zp)
        zlbl = 'lg(1+z)'
    fig, axes = plt.subplots(3, 1, sharex=True, num=2)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)

    ax = axes[0]
    ax.errorbar(zmean, alpha, alpha_err)
    p = Polynomial.fit(zmean, alpha, deg=1, w=alpha_err**-2)
    yp = p(zp)
    ax.plot(zp, yp)
    ax.text(0.4, 0.85, rf'$\alpha = {p.coef[0]:5.3f} + {p.coef[1]:5.3f} {zlbl}$',
            transform=ax.transAxes)
    ax.set_ylabel(r'$\alpha$')

    ax = axes[1]
    ax.errorbar(zmean, Mstar, Mstar_err)
    p = Polynomial.fit(zmean, Mstar, deg=1, w=Mstar_err**-2)
    yp = p(zp)
    ax.plot(zp, yp)
    ax.text(0.4, 0.85, rf'$M^* = {p.coef[0]:5.3f} + {p.coef[1]:5.3f} {zlbl}$',
            transform=ax.transAxes)
    ax.set_ylabel(r'$M^*$')
    
    ax = axes[2]
    ax.errorbar(zmean, phistar, phistar_err)
    p = Polynomial.fit(zmean, phistar, deg=1, w=phistar_err**-2)
    yp = p(zp)
    ax.plot(zp, yp)
    ax.text(0.4, 0.85, rf'$\phi^* = {p.coef[0]:3.2e} + {p.coef[1]:3.2e} {zlbl}$',
            transform=ax.transAxes)
    ax.set_ylabel(r'$\Phi^*$')
    ax.set_xlabel('log(1+z)')
    plt.show()


def gen_selfn(infile='14516.fits', outfile='sel_14516.pkl',
              M_bins=np.linspace(-24, -16, 5),
              zbins=np.linspace(0, 1, 6), nksamp=100):
    """Generate selection functions in luminosity bins."""

    # K-correction coefficients for mag bins defined by M_bins
    kcoeffs = [[0, 1.70, 0.64], [0, 0.77, 0.94], [0, 0.65, 0.08], [0, -0.26, 0.53]]

    t = Table.read(infile)
    sel = t['true_redshift_gal'] < 1
    t = t[sel]

    m, M, z = t['hmag'], t['habs'], t['true_redshift_gal']
    dm = cosmo.distmod(z)
    k = m - M - dm
    ktable = Table([M, z, k], names=('M', 'z', 'k'))
    # kc = calc_kcor.calc_kcor('r', z, 'g - r', g-r)
    sel = np.isfinite(k)
    ktable = ktable[sel]

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    ax = axes[0]
    ax.set_ylabel('S(z)')
    ax = axes[1]
    ax.set_xlabel('z')
    ax.set_ylabel('N(z)')
    sel_dict = {}
    for im in range(len(Mr_bins) - 1):
        Mmin, Mmax = Mr_bins[im], Mr_bins[im+1]
        sel = (Mmin <= ktable['M']) * (ktable['M'] < Mmax)
        selfn = util.SelectionFunction(
            cosmo, alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
            phistar=[9.81e-4, -3.24e-4], mmin=0, mmax=25,
            Mmin=Mmin, Mmax=Mmax, ktable=ktable[sel], nksamp=nksamp,
            kcoeffs=kcoeffs[im], sax=axes[0], nax=axes[1])
        key = f'{Mmin:4.1f}-{Mmax:4.1f}'
        sel_dict[key] = selfn
    pickle.dump(sel_dict, open(outfile, 'wb'))
    axes[0].legend()
    axes[1].legend()
    plt.show()


def selfn_plot(infile='sel_14516.pkl'):
    """plot selection functions in luminosity bins."""

    sel_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    fig, axes = plt.subplots(5, 1, sharex=True, num=1)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    axes[0].set_ylabel(r'$z - \langle \Delta z \rangle$')
    axes[1].set_ylabel(r'$\langle M \rangle$')
    axes[2].set_ylabel(r'$\langle k \rangle$')
    axes[3].set_ylabel('S(z)')
    axes[4].set_ylabel('N(z)')
    axes[4].set_xlabel('z')
    for key in sel_dict.keys():
        selfn = sel_dict[key]
        axes[0].plot(selfn._z, selfn._z - selfn._zmean, label=key)
        axes[1].plot(selfn._z, selfn._Mmean, label=key)
        axes[2].plot(selfn._z, selfn._kmean, label=key)
        axes[3].plot(selfn._z, selfn._sel, label=key)
        axes[4].plot(selfn._z, selfn._Nz, label=key)
    plt.legend()
    plt.show()
