# Routines for analysing GAMA v1 mocks

from astropy.table import Table
import Corrfunc
import multiprocessing as mp
import numpy as np
from numpy.random import default_rng
import pdb
import pickle
import pylab as plt
import pymangle

import util
import wcorr

rng = default_rng()

# WMAP7 cosomology, converting to h=1 units
h = 1
Om0 = 0.272
cosmo = util.CosmoLookup(h, Om0)

def make_masks():
    """Masks for gama mock groups."""
    wcorr.make_rect_mask((129.0, 141.0, -2.0, 3.0), 'G1.ply')
    wcorr.make_rect_mask((174.0, 186.0, -3.0, 2.0), 'G2.ply')
    wcorr.make_rect_mask((211.5, 223.5, -2.0, 3.0), 'G3.ply')


def wcounts(out_pref='w_mag/',
            ranfac=1, tmin=0.01, tmax=10, nbins=20,
            magbins=np.linspace(15.8, 19.8, 5)):
    """Angular pair counts in mag bins."""

    nreg = 3
    nreal = 26
    limits = [[129.0, 141.0, -2.0, 3.0],
              [174.0, 186.0, -3.0, 2.0],
              [211.5, 223.5, -2.0, 3.0]]
    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    for ireg in range(nreg):
        region = ireg + 1
        t = Table.read(f'G{region}.fits')
        rmag = t['SDSS_r_obs_app']
        ngal = len(t)/nreal
        nran = int(ranfac * ngal)
        mask = pymangle.Mangle(f'G{region}.ply')
        rar, decr = mask.genrand_range(nran, *limits[ireg])
        rar, decr = rar.astype(float), decr.astype(float)
        print(f'Region {region} nran = {len(rar)}')
        info = {'Region': region, 'Nran': len(rar), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_G{region}.pkl'
        pool.apply_async(wcorr.wcounts, args=(rar, decr, bins, info, outfile))
        for ireal in range(26):
            for imag in range(len(magbins) - 1):
                mlo, mhi = magbins[imag], magbins[imag+1]
                sel = (t['ireal'] == ireal) * (mlo <= rmag) *  (rmag < mhi)
                ra, dec = t['ra'][sel].value, t['dec'][sel].value
                print(ireal, imag, len(ra))
                info = {'Region': region, 'ireal': ireal, 'mlo': mlo, 'mhi': mhi,
                        'Ngal': len(ra), 'Nran': len(rar), 'bins': bins, 'tcen': tcen}
                outfile = f'{out_pref}GG_G{region}_V{ireal}_m{imag}.pkl'
                pool.apply_async(wcorr.wcounts,
                                 args=(ra, dec, bins, info, outfile))
                outfile = f'{out_pref}GR_G{region}_V{ireal}_m{imag}.pkl'
                pool.apply_async(wcorr.wcounts,
                                 args=(ra, dec, bins, info,  outfile, rar, decr))
    pool.close()
    pool.join()


def xir_counts(infile='Gonzalez.fits', out_pref='xir_z/',
               ranfac=10, rmin=0.1, rmax=100, nbins=20,  Mr_lims=[-23, -20],
               zbins=np.linspace(0.0, 0.5, 6), multi=True):
    """Real-space pair counts in redshift bins."""

    count_fn = wcorr.xir_counts
    nz = len(zbins) - 1
    nreg = 3
    nreal = 26
    limits = [[129.0, 141.0, -2.0, 3.0],
              [174.0, 186.0, -3.0, 2.0],
              [211.5, 223.5, -2.0, 3.0]]
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    if multi:
        ncpu = mp.cpu_count()
        pool = mp.Pool(ncpu)
        print('Using', ncpu, 'CPUs')
    
    t = Table.read(infile)
    # Select L* galaxies, well-sampled in all z slices
    rabs = t['SDSS_r_rest_abs']
    sel = (Mr_lims[0] < rabs) * (rabs < Mr_lims[1])
    t = t[sel]
    z = t['redshift_cos']
    rabs = t['SDSS_r_rest_abs']
    for iz in range(nz):
        zlo, zhi = zbins[iz], zbins[iz+1]
        selz = (zlo <= z) * (z < zhi)
        Mmean = np.mean(rabs[selz])
        for ireg in range(nreg):
            region = ireg + 1
            selr = selz * (limits[ireg][0] <= t['ra']) * (t['ra'] <= limits[ireg][1])
            ngal = len(t[selr])/nreal
            nran = int(ranfac * ngal)
            mask = pymangle.Mangle(f'G{region}.ply')
            rar, decr = mask.genrand_range(nran, *limits[ireg])
            rar, decr = rar.astype(float), decr.astype(float)
            rr = cosmo.dc(rng.choice(z[selz], nran, replace=True))
            rancat = wcorr.Cat(rar.astype('float32'), decr.astype('float32'), r=rr)
            rancat.gen_cart()
            rancat.write(f'{out_pref}ran_G{region}_z{iz}.txt')
            print(f'Redshift bin {iz}, region {region} nran = {len(rar)}')
            info = {'Region': region, 'Nran': len(rar), 'bins': bins,
                    'rcen': rcen, 'zlo': zlo, 'zhi': zhi, 'Mmean': Mmean}
            outfile = f'{out_pref}RR_G{region}_z{iz}.pkl'
            if multi:
                pool.apply_async(count_fn,
                                 args=(rancat.x, rancat.y, rancat.z,
                                       bins, info, outfile))
            else:
                count_fn(rancat.x, rancat.y, rancat.z,
                         bins, info, outfile)
            for ireal in range(26):
                sel = selr * (t['ireal'] == ireal)
                ra = t['ra'][sel].astype('float32')
                dec = t['dec'][sel].astype('float32')
                r = cosmo.dc(z[sel])
                galcat = wcorr.Cat(ra, dec, r=r)
                galcat.gen_cart()
                galcat.write(f'{out_pref}gal_G{region}_z{iz}_V{ireal}.txt')
                print(ireal, len(ra))
                info = {'Region': region, 'ireal': ireal,
                        'zlo': zlo, 'zhi': zhi, 'Mmean': Mmean,
                        'Ngal': len(ra), 'Nran': len(rar), 'bins': bins,
                        'rcen': rcen}
                outgg = f'{out_pref}GG_G{region}_z{iz}_V{ireal}.pkl'
                outgr = f'{out_pref}GR_G{region}_z{iz}_V{ireal}.pkl'
                # if multi:
                #     pool.apply_async(wcorr.xir_counts,
                #                      args=(galcat.x, galcat.y, galcat.z,
                #                            bins, info, outgg))
                #     pool.apply_async(wcorr.xir_counts,
                #                      args=(galcat.x, galcat.y, galcat.z,
                #                            bins, info,  outgr,
                #                            rancat.x, rancat.y, rancat.z))
                # else:
                #     wcorr.xir_counts(galcat.x, galcat.y, galcat.z,
                #                      bins, info, outgg)
                #     wcorr.xir_counts(galcat.x, galcat.y, galcat.z,
                #                      bins, info,  outgr,
                #                      rancat.x, rancat.y, rancat.z)
                if multi:
                    pool.apply_async(count_fn,
                                     args=(galcat.x, galcat.y, galcat.z,
                                           bins, info, outgg))
                    pool.apply_async(count_fn,
                                     args=(galcat.x, galcat.y, galcat.z,
                                           bins, info,  outgr,
                                           rancat.x, rancat.y, rancat.z))
                else:
                    count_fn(galcat.x, galcat.y, galcat.z,
                                     bins, info, outgg)
                    count_fn(galcat.x, galcat.y, galcat.z,
                                     bins, info,  outgr,
                                     rancat.x, rancat.y, rancat.z)
    if multi:
        pool.close()
        pool.join()


def rplot(iz=0):
    """Plot N(r) histograms for specified redshift shell."""
    (rar, decr, rr) = np.loadtxt(f'ran_G1_z{iz}.txt')
    rg = []
    for ivol in range(26):
        (ra, dec, r) = np.loadtxt(f'gal_G1_z{iz}_V{ivol}.txt')
        rg.append(r)
    rg = [item for sublist in rg for item in sublist]
    
    plt.clf()
    plt.hist(rg, histtype='step', label='galaxies')
    plt.hist(rr, histtype='step', label='randoms', weights=2.6*(np.ones(len(rr))))
    plt.legend()
    plt.xlabel(r'$r$ [Mpc/h]')
    plt.ylabel(r'$N(r)$')
    plt.show()


def plot_sel(nmag=4, fit_range=[0.01, 0.5], p0=[0.05, 1.7], prefix='w_mag/',
           avgcounts=False, gamma1=1.8, gamma2=2.8, r0=5.0, eps=-3,
           alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
                phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67]):
    """Plot selection functions."""
    nreg = 3
    nreal = 26
    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for imag in range(nmag):
        corrs = []
        for ireg in range(nreg):
            region = ireg + 1
            infile = f'{prefix}RR_G{region}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            for ireal in range(nreal):
                infile = f'{prefix}GG_G{region}_V{ireal}_m{imag}.pkl'
                (info, DD_counts) = pickle.load(open(infile, 'rb'))
                infile = f'{prefix}GR_G{region}_V{ireal}_m{imag}.pkl'
                (info, DR_counts) = pickle.load(open(infile, 'rb'))
                corrs.append(
                    wcorr.Corr1d(info['Ngal'], info['Nran'],
                                 DD_counts, DR_counts, RR_counts,
                                 mlo=info['mlo'], mhi=info['mhi']))
        corr = wcorr.Corr1d()
        corr.average(corrs, avgcounts=avgcounts)
        corr_slices.append(corr)
    wcorr.plot_sel(cosmo, corr_slices, gamma1=gamma1, gamma2=gamma2,
                   r0=r0, eps=eps, alpha=alpha, Mstar=Mstar,
                   phistar=phistar, kcoeffs=kcoeffs)

    
def w_plot(nmag=4, fit_range=[0.01, 0.5], p0=[0.05, 1.7], prefix='w_mag/',
           avgcounts=False, gamma1=1.8, gamma2=2.8, r0=5.0, eps=-3,
           alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
                phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67]):
    """w(theta) from angular pair counts in mag bins."""

    nreg = 3
    nreal = 26
    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for imag in range(nmag):
        corrs = []
        for ireg in range(nreg):
            region = ireg + 1
            infile = f'{prefix}RR_G{region}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            for ireal in range(nreal):
                infile = f'{prefix}GG_G{region}_V{ireal}_m{imag}.pkl'
                (info, DD_counts) = pickle.load(open(infile, 'rb'))
                infile = f'{prefix}GR_G{region}_V{ireal}_m{imag}.pkl'
                (info, DR_counts) = pickle.load(open(infile, 'rb'))
                corrs.append(
                    wcorr.Corr1d(info['Ngal'], info['Nran'],
                                 DD_counts, DR_counts, RR_counts,
                                 mlo=info['mlo'], mhi=info['mhi']))
        corr = wcorr.Corr1d()
        corr.average(corrs, avgcounts=avgcounts)
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

# def wplot_scale():
#     """Plot w(theta) results for mag slices with Limber scaling."""
#     wcorr.wplot_scale(cosmo, infile='w12244_mag.pkl', gamma1=1.62, gamma2=2.3,
#                       r0=3.45, eps=-2.1, maglims=np.linspace(15, 20, 6),
#                       alpha=[-0.956, -0.196], Mstar=[-21.135, -0.497],
#                 phistar=[3.26e-3, -1.08e-3], kcoeffs=[0.0, -0.39, 1.67])


def xir_plot(nz=5, fit_range=[0.1, 20], p0=[5, 1.7], prefix='xir_z/', avgcounts=False):
    """xi(r) from pair counts in redshift bins."""

    nreg = 3
    nreal = 26
    plt.clf()
    ax = plt.subplot(111)
    for iz in range(nz):
        corrs = []
        for ireg in range(nreg):
            region = ireg + 1
            infile = f'{prefix}RR_G{region}_z{iz}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            for ireal in range(nreal):
                infile = f'{prefix}GG_G{region}_z{iz}_V{ireal}.pkl'
                (info, DD_counts) = pickle.load(open(infile, 'rb'))
                infile = f'{prefix}GR_G{region}_z{iz}_V{ireal}.pkl'
                (info, DR_counts) = pickle.load(open(infile, 'rb'))
                corrs.append(
                    wcorr.Corr1d(info['Ngal'], info['Nran'],
                                 DD_counts, DR_counts, RR_counts))
        corr = wcorr.Corr1d()
        corr.average(corrs, avgcounts=avgcounts)
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax, color=color,
                  label=f"z = [{info['zlo']:3.1f}, {info['zhi']:3.1f}]")
        popt, pcov = corr.fit_xi(fit_range, p0, ax, color)
        print(popt, pcov)
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$r$ [Mpc/h]')
    plt.ylabel(r'$\xi(r)$')
    plt.show()


def limber_scale(gamma1=1.8, gamma2=2.8, r0=5.0, eps=-3,
                 maglims=np.linspace(15.8, 19.8, 5),
                 zbins=np.linspace(0.0, 0.5, 100), plotsel=0, plotint=1):
    """Scale GAMA mock w(theta) measurements to depth of brightest slice."""
    
    cosmo = util.CosmoLookup(zbins=zbins, plot=0)
    nmag = len(maglims)-1
    A, B = np.zeros(nmag), np.zeros(nmag)
    if plotsel:
        plt.clf()
    print('mlo  mhi  A  B   dlgt  dlgw')
    for im in range(nmag):
        mlo, mhi = maglims[im], maglims[im+1]
        cosmo.set_selfn(mlo=mlo, mhi=mhi, plot=plotsel)
        A[im] = limber.w_a(cosmo, gamma=gamma1, r0=r0, eps=eps, plotint=plotint)
        B[im] = limber.w_a(cosmo, gamma=gamma2, r0=r0, eps=eps, plotint=plotint)
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
