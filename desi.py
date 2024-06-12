# Clustering measurements for DESI EDR and Legacy survey

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
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
import Corrfunc
import pdb
import psutil
import pymangle
import treecorr

import calc_kcor
import cluster_z
import util
import wcorr

ncores = min(32, psutil.cpu_count(logical=False))
print(f'{ncores=}')

ln10 = math.log(10)
rng = default_rng()

# Flagship2 cosomology, converting to h=1 units
# h = 1
# Om0 = 0.319
# cosmo = util.CosmoLookup(h, Om0)

solid_angle_north = 1
def BGS_N_wcounts():
    """Angular pair counts for DESI BGS north."""
    desi_wcounts(galfile='BGS_ANY_N_clustering.dat.fits',
                 ranfile='BGS_ANY_N_0_clustering.ran.fits',
                 out_path='BGS_N_w_z/')

    
def BGS_S_wcounts():
    """Angular pair counts DESI BGS south."""
    desi_wcounts(galfile='BGS_ANY_S_clustering.dat.fits',
                 ranfile='BGS_ANY_S_0_clustering.ran.fits',
                 out_path='BGS_S_w_z/')

    
def desi_wcounts(path='/Users/loveday/Data/DESI/',
                 galfile='BGS_ANY_S_clustering.dat.fits',
                 ranfile='BGS_ANY_S_0_clustering.ran.fits',
                 out_path='/Users/loveday/Data/DESI/BGS_w_z/',
                 tmin=0.001, tmax=10, nbins=20,
                 zbins=np.linspace(0.0, 0.6, 13)):
    """DESI angular auto-pair counts in redshift bins."""

    def create_cat(infile):
        """Create catalogue from specified input file."""
        t = Table.read(path + infile)
        ra, dec, z, rosette = t['RA'], t['DEC'], t['Z'], t['ROSETTE_NUMBER']
        njack, jack = jack_from_rosette(rosette)
        cat = wcorr.Cat(ra, dec, jack=jack)
        cat.z = z

        return cat, njack

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))

    galcat, njack = create_cat(galfile)
    rancat, njack_ran = create_cat(ranfile)
    assert (njack == njack_ran)

    plt.clf()
    plt.subplot(121)
    plt.scatter(galcat.ra, galcat.dec, c=galcat.jack, s=0.1)
    plt.subplot(122)
    plt.scatter(rancat.ra, rancat.dec, c=rancat.jack, s=0.1)
    plt.show()
    
    rcat = treecorr.Catalog(ra=rancat.ra, dec=rancat.dec,
                            ra_units='deg', dec_units='deg',
                            patch=rancat.jack-1)
    print('random cat: ', rancat.nobj)
    rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees') #, var_method='jackknife')
    rr.process(rcat)

    for iz in range(len(zbins) - 1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        sel = (zlo <= galcat.z) * (galcat.z < zhi)
        zmean = np.mean(galcat.z[sel])
        gcat = treecorr.Catalog(ra=galcat.ra[sel], dec=galcat.dec[sel],
                                 ra_units='deg', dec_units='deg',
                                 patch=galcat.jack[sel]-1)

        dr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                  sep_units='degrees') #, var_method='jackknife')
        dr.process(gcat, rcat)
        dd = treecorr.NNCorrelation(
            min_sep=tmin, max_sep=tmax, nbins=nbins,
            sep_units='degrees', var_method='jackknife')
        dd.process(gcat)
        dd.calculateXi(rr=rr, dr=dr)
        xi_jack, w = dd.build_cov_design_matrix('jackknife')
        outfile = f'{out_path}z{iz}.fits'
        dd.write(outfile, rr=rr, dr=dr)
        with fits.open(outfile, mode='update') as hdul:
            hdr = hdul[1].header
            hdr['zlo'] = zlo
            hdr['zhi'] = zhi
            hdr['zmean'] = zmean
            hdr['Ngal'] = gcat.nobj
            hdr['Nran'] = rcat.nobj
            hdul.append(fits.PrimaryHDU(xi_jack))
            hdul.flush()


def BGS_S_xcounts():
    desi_xcounts(
        galfile='BGS_ANY_S_clustering.dat.fits',
        ranfile='BGS_ANY_S_0_clustering.ran.fits',
        out_path='BGS_S_w_z_m/')


def BGS_N_xcounts():
    desi_xcounts(
        galfile='BGS_ANY_N_clustering.dat.fits',
        ranfile='BGS_ANY_N_0_clustering.ran.fits',
        out_path='BGS_N_w_z_m/')


def LRG_S_xcounts():
    desi_xcounts(
        galfile='LRG_S_clustering.dat.fits',
        ranfile='LRG_S_0_clustering.ran.fits',
        out_path='LRG_S_w_z_m/', zbins=np.linspace(0.0, 1.2, 13))


def LRG_N_xcounts():
    desi_xcounts(
        galfile='LRG_N_clustering.dat.fits',
        ranfile='LRG_N_0_clustering.ran.fits',
        out_path='LRG_N_w_z_m/', zbins=np.linspace(0.0, 1.2, 13))


def desi_xcounts(path='/Users/loveday/Data/DESI/',
                 galfile='BGS_ANY_S_clustering.dat.fits',
                 ranfile='BGS_ANY_S_0_clustering.ran.fits',
                 out_path='/Users/loveday/Data/DESI/BGS_w_z_m/',
                 tmin=0.001, tmax=10, nbins=20,
                 zbins=np.linspace(0.0, 0.6, 13),
                 magbins=np.linspace(16, 20, 5)):
    """DESI angular cross-pair counts in redshift-magnitude bins."""

    def create_cat(infile):
        """Create catalogue from specified input file."""
        t = Table.read(path + infile)
        ra, dec, z, rosette = t['RA'], t['DEC'], t['Z'], t['ROSETTE_NUMBER']
        njack, jack = jack_from_rosette(rosette)
        mag = 22.5 - 2.5*np.log10(t['FLUX_Z_DERED'])
        
        cat = wcorr.Cat(ra, dec, jack=jack)
        cat.z = z
        cat.mag = mag
        return cat, njack

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))

    galcat, njack = create_cat(galfile)
    rancat, njack_ran = create_cat(ranfile)
    assert (njack == njack_ran)

    rcat = treecorr.Catalog(ra=rancat.ra, dec=rancat.dec,
                            ra_units='deg', dec_units='deg',
                            patch=rancat.jack-1)
    print('random cat: ', rancat.nobj)
    rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees', var_method='jackknife')
    rr.process(rcat)

    rd_list = []
    for iz in range(len(zbins) - 1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        sel = (zlo <= galcat.z) * (galcat.z < zhi)
        zmean = np.mean(galcat.z[sel])
        gcatz = treecorr.Catalog(ra=galcat.ra[sel], dec=galcat.dec[sel],
                                 ra_units='deg', dec_units='deg',
                                 patch=galcat.jack[sel]-1)
        dr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                    sep_units='degrees', var_method='jackknife')
        dr.process(gcatz, rcat)

        for im in range(len(magbins) - 1):
            mlo, mhi = magbins[im], magbins[im+1]
            sel = (mlo <= galcat.mag) * (galcat.mag < mhi)
            gcatm = treecorr.Catalog(ra=galcat.ra[sel], dec=galcat.dec[sel],
                                     ra_units='deg', dec_units='deg',
                                     patch=galcat.jack[sel]-1)
            if iz == 0:
                rd = treecorr.NNCorrelation(
                    min_sep=tmin, max_sep=tmax, nbins=nbins,
                    sep_units='degrees', var_method='jackknife')
                rd.process(rcat, gcatm)
                rd_list.append(rd)
            else:
                rd = rd_list[im]
                
            dd = treecorr.NNCorrelation(
                min_sep=tmin, max_sep=tmax, nbins=nbins,
                sep_units='degrees', var_method='jackknife')
            dd.process(gcatz, gcatm)
            dd.calculateXi(rr=rr, dr=dr, rd=rd)
            xi_jack, w = dd.build_cov_design_matrix('jackknife')
            outfile = f'{out_path}z{iz}_m{im}.fits'
            dd.write(outfile, rr=rr, dr=dr, rd=rd)
            with fits.open(outfile, mode='update') as hdul:
                hdr = hdul[1].header
                hdr['zlo'] = zlo
                hdr['zhi'] = zhi
                hdr['zmean'] = zmean
                hdr['mlo'] = mlo
                hdr['mhi'] = mhi
                hdr['Ngal1'] = gcatz.nobj
                hdr['Ngal2'] = gcatm.nobj
                hdr['Nran1'] = rcat.nobj
                hdr['Nran2'] = rcat.nobj
                hdul.append(fits.PrimaryHDU(xi_jack))
                hdul.flush()


def legacy_N_wcounts():
    legacy_wcounts(galfile='legacy_N.fits',
                   ranfile='legacy_N_ran-{}.fits',
                   out_path='N_w_m/')

def legacy_wcounts(galfile='legacy_S.fits',
                   ranfile='legacy_S_ran-{}.fits',
                   out_path='S_w_m/', tmin=0.001, tmax=10, nbins=20,
                   magbins=np.linspace(18, 23, 6), nran=1, npatch=9):
    """Legacy angular auto-pair counts in Z-band magnitude bins."""

    # Read Legacy randoms
    ra, dec = np.array([]), np.array([])
    for iran in range(nran):
        t = Table.read(ranfile.format(iran))
        ra = np.append(ra, t['RA'])
        dec = np.append(dec, t['DEC'])
    rcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg',
                            npatch=npatch)
    print(rcat.nobj, 'Randoms')
    del t, ra, dec
    rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                sep_units='degrees')
    rr.process(rcat)
    
    # Read Legacy galaxies
    t = Table.read(galfile)
    sel = (t['LTYPE'] != 'PSF') * (t['Z_MAG'] < magbins[-1])
    t = t[sel]

    # Divide into magnitude bins
    print('imag  ngal')
    for imag in range(len(magbins) - 1):
        mlo, mhi = magbins[imag], magbins[imag+1]
        sel = (mlo <= t['Z_MAG']) * (t['Z_MAG'] < mhi)
        print(imag, len(t[sel]))
        gcat = treecorr.Catalog(ra=t['RA'][sel], dec=t['DEC'][sel],
                                ra_units='deg', dec_units='deg',
                                patch_centers=rcat.patch_centers)
        dr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                    sep_units='degrees')
        dr.process(gcat, rcat)

        dd = treecorr.NNCorrelation(
            min_sep=tmin, max_sep=tmax, nbins=nbins,
            sep_units='degrees', var_method='jackknife')
        dd.process(gcat)
        dd.calculateXi(rr=rr, dr=dr)
        xi_jack, w = dd.build_cov_design_matrix('jackknife')
        outfile = f'{out_path}m{imag}.fits'
        dd.write(outfile, rr=rr, dr=dr)
        with fits.open(outfile, mode='update') as hdul:
            hdr = hdul[1].header
            hdr['mlo'] = mlo
            hdr['mhi'] = mhi
            hdul.append(fits.PrimaryHDU(xi_jack))
            hdul.flush()                    
    del t, gcat, rcat, dd, dr, rr

def BGS_S_legacy_xcounts():
    desi_legacy_xcounts(
        desi_galfile='BGS_ANY_S_clustering.dat.fits',
        desi_ranfile='BGS_ANY_S_0_clustering.ran.fits',
        dpath='/Users/loveday/Data/DESI/',
        lpath='/Users/loveday/Data/Legacy/10.1/',
        legacy_galfile='legacy_S.fits',
        legacy_ranfile='legacy_S_ran-{}.fits',
        out_path='BGS_S_X_L_w_z_m/', plotdist=1)


def BGS_N_legacy_xcounts():
    desi_legacy_xcounts(
        desi_galfile='BGS_ANY_N_clustering.dat.fits',
        desi_ranfile='BGS_ANY_N_0_clustering.ran.fits',
        dpath='/Users/loveday/Data/DESI/',
        lpath='/Users/loveday/Data/Legacy/10.1/',
        legacy_galfile='legacy_N.fits',
        legacy_ranfile='legacy_N_ran-{}.fits',
        out_path='BGS_N_X_L_w_z_m/', plotdist=1)


def LRG_S_legacy_xcounts():
    desi_legacy_xcounts(
        desi_galfile='LRG_S_clustering.dat.fits',
        desi_ranfile='LRG_S_0_clustering.ran.fits',
        dpath='/Users/loveday/Data/DESI/',
        lpath='/Users/loveday/Data/Legacy/10.1/',
        legacy_galfile='legacy_S.fits',
        legacy_ranfile='legacy_S_ran-{}.fits',
        out_path='LRG_S_X_L_w_z_m/', zbins=np.linspace(0.0, 1.2, 13), plotdist=1)


def LRG_N_legacy_xcounts():

    desi_legacy_xcounts(
        desi_galfile='LRG_N_clustering.dat.fits',
        desi_ranfile='LRG_N_0_clustering.ran.fits',
        dpath='/Users/loveday/Data/DESI/',
        lpath='/Users/loveday/Data/Legacy/10.1/',
        legacy_galfile='legacy_N.fits',
        legacy_ranfile='legacy_N_ran-{}.fits',
        out_path='LRG_N_X_L_w_z_m/', zbins=np.linspace(0.0, 1.2, 13), plotdist=1)


def ELG_S_legacy_xcounts():

    desi_legacy_xcounts(
        desi_galfile='ELG_S_clustering.dat.fits',
        desi_ranfile='ELG_S_0_clustering.ran.fits',
        dpath='/Users/loveday/Data/DESI/',
        lpath='/Users/loveday/Data/Legacy/10.1/',
        legacy_galfile='legacy_S.fits',
        legacy_ranfile='legacy_S_ran-{}.fits',
        out_path='ELG_S_X_L_w_z_m/', zbins=np.linspace(0.5, 1.7, 13), plotdist=1)


def ELG_N_legacy_xcounts():

    desi_legacy_xcounts(
        desi_galfile='ELG_N_clustering.dat.fits',
        desi_ranfile='ELG_N_0_clustering.ran.fits',
        dpath='/Users/loveday/Data/DESI/',
        lpath='/Users/loveday/Data/Legacy/10.1/',
        legacy_galfile='legacy_N.fits',
        legacy_ranfile='legacy_N_ran-{}.fits',
        out_path='ELG_N_X_L_w_z_m/', zbins=np.linspace(0.5, 1.7, 13), plotdist=1)


def desi_legacy_xcounts(desi_galfile='BGS_ANY_S_clustering.dat.fits',
                        desi_ranfile='BGS_ANY_S_0_clustering.ran.fits',
                        dpath='/global/cfs/cdirs/desi/public/edr/vac/edr/lss/v2.0/LSScats/clustering/',
                        lpath='/pscratch/sd/l/loveday/Legacy/10.1/',
                        legacy_galfile='legacy.fits',
                        legacy_ranfile='legacy_ran.fits',
                        out_path='/pscratch/sd/l/loveday/Legacy/10.1/bgs_x_l',
                        tmin=0.001, tmax=10, nbins=20,
                        zbins=np.linspace(0.0, 0.6, 13),
                        magbins=np.linspace(18, 20, 5), engine='treecorr',
                        plotdist=0):
    """DESI-Legacy angular cross-pair counts in redshift/magnitude bins."""

    def create_desi_cat(infile):
        """Create catalogue from specified input file."""
        t = Table.read(dpath + infile)
        ra, dec, z, rosette = t['RA'], t['DEC'], t['Z'], t['ROSETTE_NUMBER']
        njack, jack = jack_from_rosette(rosette)

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

    # Read DESI galaxies plus randoms
    dgalcat, njack = create_desi_cat(desi_galfile)
    drancat, njack_ran = create_desi_cat(desi_ranfile)
    assert (njack == njack_ran)

    # Read Legacy sources
    t = Table.read(lpath + legacy_galfile)
    sel = (t['LTYPE'] != 'PSF') * (t['Z_MAG'] < magbins[-1])
    t = t[sel]

    # Divide into magnitude bins
    sub = np.zeros(len(t), dtype='int8')
    print('imag  nobj')
    for imag in range(len(magbins) - 1):
        sel = (magbins[imag] <= t['Z_MAG']) * (t['Z_MAG'] < magbins[imag+1])
        sub[sel] = imag
        print(imag, len(t[sel]))
    lgalcat = wcorr.Cat(t['RA'], t['DEC'], sub=sub)
    print(lgalcat.nobj, 'total Legacy galaxies')

    # Read Legacy randoms
    ra, dec = np.array([]), np.array([])
    for iran in range(20):
        t = Table.read(lpath + legacy_ranfile.format(iran))
        ra = np.append(ra, t['RA'])
        dec = np.append(dec, t['DEC'])
    lrancat = wcorr.Cat(ra, dec)
    print(lrancat.nobj, 'Legacy randoms')
    del t, ra, dec

    if engine == 'corrfunc':
        pool = mp.Pool(ncores)
        lrcoords = lrancat.sample()
        for ijack in range(njack+1):
            drcoords = drancat.sample(ijack)
            info = {'Jack': ijack,
                    'Nran1': len(drcoords[0]), 'Nran2': len(lrcoords[0]),
                    'bins': bins, 'tcen': tcen}
            outfile = f'{out_path}/R1R2_J{ijack}.pkl'
            result = pool.apply_async(
                wcorr.wxcounts, args=(*drcoords, *lrcoords, bins, info, outfile))
            for iz in range(len(zbins) - 1):
                zlo, zhi = zbins[iz], zbins[iz+1]
                dgcoords = dgalcat.sample(ijack, sub=iz)
                for im in range(len(magbins) - 1):
                    mlo, mhi = magbins[im], magbins[im+1]
                    lgcoords = lgalcat.sample(sub=im)
                    info = {'Jack': ijack, 'zlo': zlo, 'zhi': zhi,
                            'mlo': mlo, 'mhi': mhi,
                            'Ngal1': len(dgcoords[0]), 'Ngal2': len(lgcoords[0]),
                            'Nran1': len(drcoords[0]), 'Nran2': len(lrcoords[0]),
                            'bins': bins, 'tcen': tcen}

                    outfile = f'{out_path}/D1D2_J{ijack}_z{iz}_m{im}.pkl'
                    result = pool.apply_async(
                        wcorr.wxcounts, args=(*dgcoords, *lgcoords, bins, info, outfile))
                    outfile = f'{out_path}/D1R2_J{ijack}_z{iz}_m{im}.pkl'
                    result = pool.apply_async(
                        wcorr.wxcounts, args=(*dgcoords, *lrcoords, bins, info,  outfile))
                    outfile = f'{out_path}/D2R1_J{ijack}_z{iz}_m{im}.pkl'
                    result = pool.apply_async(
                        wcorr.wxcounts, args=(*lgcoords, *drcoords, bins, info,  outfile))
        pool.close()
        pool.join()

    if engine == 'treecorr':

        drcat = treecorr.Catalog(ra=drancat.ra, dec=drancat.dec,
                                 ra_units='deg', dec_units='deg',
                                 # npatch=np.max(drancat.jack))
                                 patch=drancat.jack-1)
        lrcat = treecorr.Catalog(ra=lrancat.ra, dec=lrancat.dec,
                                 ra_units='deg', dec_units='deg',
                                 patch_centers=drcat.patch_centers)
        print('random cats: ', drancat.nobj, lrancat.nobj)

        if plotdist:
            plt.ion()
            plt.clf()
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
            fig.set_size_inches(8, 8)
            fig.subplots_adjust(hspace=0, wspace=0)
            axes[0, 0].scatter(drcat.ra, drcat.dec, c=drcat.patch, s=0.1)
            # axes[0, 0].scatter(drancat.ra, drancat.dec, s=0.1)
            # axes[0, 1].scatter(drancat.ra, drancat.dec, s=0.1)
            # axes[0, 1].scatter(dgalcat.ra, dgalcat.dec, s=0.1)
            axes[1, 0].scatter(lrcat.ra, lrcat.dec, c=lrcat.patch, s=0.1)
            # axes[1, 0].scatter(lrancat.ra, lrancat.dec, s=0.1)
            # axes[1, 1].scatter(lrancat.ra, lrancat.dec, s=0.1)
            # axes[1, 1].scatter(lgalcat.ra, lgalcat.dec, s=0.1)
            # plt.show()
    
        rr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                    sep_units='degrees') # , var_method='jackknife')
        rr.process(drcat, lrcat)

        rd_list = []
        print('iz  im    Ngal_DESI   Ngal_Legacy')
        for iz in range(len(zbins) - 1):
            zlo, zhi = zbins[iz], zbins[iz+1]
            dgalsamp = dgalcat.subset(sub=iz)
            dgcat = treecorr.Catalog(ra=dgalsamp.ra, dec=dgalsamp.dec,
                                     ra_units='deg', dec_units='deg',
                                     patch_centers=drcat.patch_centers)
            if plotdist and iz==0:
                axes[0, 1].scatter(dgcat.ra, dgcat.dec, c=dgcat.patch, s=0.1)
                # plt.show()
            dr = treecorr.NNCorrelation(min_sep=tmin, max_sep=tmax, nbins=nbins,
                                        sep_units='degrees') # , var_method='jackknife')
            dr.process(dgcat, lrcat)

            for im in range(len(magbins) - 1):
                mlo, mhi = magbins[im], magbins[im+1]
                lgalsamp = lgalcat.subset(sub=im)
                print(iz, im, dgalsamp.nobj, lgalsamp.nobj)
                lgcat = treecorr.Catalog(ra=lgalsamp.ra, dec=lgalsamp.dec,
                                         ra_units='deg', dec_units='deg',
                                         patch_centers=drcat.patch_centers)
                if plotdist and iz==0 and im==0:
                    axes[1, 1].scatter(lgcat.ra, lgcat.dec, c=lgcat.patch, s=0.1)
                    plt.show()
                if iz == 0:
                    rd = treecorr.NNCorrelation(
                        min_sep=tmin, max_sep=tmax, nbins=nbins,
                        sep_units='degrees') # , var_method='jackknife')
                    rd.process(drcat, lgcat)
                    rd_list.append(rd)
                else:
                    rd = rd_list[im]

                dd = treecorr.NNCorrelation(
                    min_sep=tmin, max_sep=tmax, nbins=nbins,
                    sep_units='degrees', var_method='jackknife')
                dd.process(dgcat, lgcat)
                dd.calculateXi(rr=rr, dr=dr, rd=rd)
                xi_jack, w = dd.build_cov_design_matrix('jackknife')
                outfile = f'{out_path}z{iz}_m{im}.fits'
                dd.write(outfile, rr=rr, dr=dr, rd=rd)
                with fits.open(outfile, mode='update') as hdul:
                    hdr = hdul[1].header
                    hdr['zlo'] = zlo
                    hdr['zhi'] = zhi
                    hdr['mlo'] = mlo
                    hdr['mhi'] = mhi
                    hdr['Ngal1'] = dgcat.nobj
                    hdr['Ngal2'] = lgcat.nobj
                    hdr['Nran1'] = drcat.nobj
                    hdr['Nran2'] = lrcat.nobj
                    hdul.append(fits.PrimaryHDU(xi_jack))
                    hdul.flush()

# def legacy_wcounts(path='/pscratch/sd/l/loveday/Legacy/',
#                    galfile='legacy_desi.fits',
#                    ranfile='legacy_desi_ran.fits',
#                    tmin=0.01, tmax=10, nbins=20,
#                    magbins=np.linspace(18, 23, 6)):
#     """Legacy angular auto-pair counts in Z-band magnitude bins.  Uses output files generated by legacy_desi() in DESI.ipynb"""

#     t = Table.read(path + galfile)
#     sel = (t['LTYPE'] != 'PSF') * (t['Z_MAG'] < magbins[-1])
#     t = t[sel]

#     njack, jack_gal = jack_from_rosette(t['ROSETTE'])
    
#     # Divide into magnitude bins
#     sub = np.zeros(len(jack_gal), dtype='int8')
#     print('imag  nobj')
#     for imag in range(len(magbins) - 1):
#         sel = (magbins[imag] <= t['Z_MAG']) * (t['Z_MAG'] < magbins[imag+1])
#         sub[sel] = imag
#         print(imag, len(jack_gal[sel]))
#     galcat = wcorr.Cat(t['RA'], t['DEC'], sub=sub, jack=jack_gal)
#     print(galcat.nobj, 'total galaxies')
    
#     t = Table.read(path + ranfile)
#     njack_ran, jack_ran = jack_from_rosette(t['ROSETTE'])
#     assert (njack == njack_ran)
#     rancat = wcorr.Cat(t['RA'], t['DEC'], jack=jack_ran)
#     print(rancat.nobj, 'total randoms')

#     bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
#     tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
#     pool = mp.Pool(ncores)
#     out_path = path + 'w_mag/'

#     for ijack in range(njack+1):
#         rcoords = rancat.sample(ijack)
#         info = {'Jack': ijack, 'Nran': len(rcoords[0]), 'bins': bins, 'tcen': tcen}
#         outfile = f'{out_path}RR_J{ijack}.pkl'
#         pool.apply_async(wcorr.wcounts, args=(*rcoords, bins, info, outfile))
#         for imag in range(len(magbins) - 1):
#             print(ijack, imag)
#             mlo, mhi = magbins[imag], magbins[imag+1]
#             gcoords = galcat.sample(ijack, sub=imag)
#             info = {'Jack': ijack, 'mlo': mlo, 'mhi': mhi,
#                     'Ngal': len(gcoords[0]), 'Nran': len(rcoords[0]),
#                     'bins': bins, 'tcen': tcen}
#             outfile = f'{out_path}GG_J{ijack}_m{imag}.pkl'
#             pool.apply_async(wcorr.wcounts,
#                              args=(*gcoords, bins, info, outfile))
#             outfile = f'{out_path}GR_J{ijack}_m{imag}.pkl'
#             pool.apply_async(wcorr.wcounts,
#                              args=(*gcoords, bins, info,  outfile, *rcoords))


def jack_from_rosette(rosette):
    """Assign jackknife regions by rosette number - note these aren't contiguous."""
    rosettes = np.unique(rosette)
    njack = len(rosettes)
    jack = np.zeros(len(rosette))
    for ir in range(njack):
        sel = rosette == rosettes[ir]
        jack[sel] = ir + 1
    return njack, jack


def w_plot(nz=5, njack=10, fit_range=[0.01, 5], p0=[0.05, 1.7],
           prefix='w_N/', avgcounts=False, ic_rmax=0):
    """w(theta) from angular pair counts in redshift bins."""

    plt.clf()
    ax = plt.subplot(111)
    corr_slices = []
    for iz in range(nz):
        corrs = []
        for ijack in range(njack+1):
            infile = f'{prefix}RR_J{ijack}_z{iz}.pkl'
            (info, RR_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GG_J{ijack}_z{iz}.pkl'
            (info, DD_counts) = pickle.load(open(infile, 'rb'))
            infile = f'{prefix}GR_J{ijack}_z{iz}.pkl'
            (info, DR_counts) = pickle.load(open(infile, 'rb'))
            corrs.append(
                wcorr.Corr1d(info['Ngal'], info['Nran'],
                             DD_counts, DR_counts, RR_counts))
        corr = corrs[0]
        corr.err = np.std(np.array([corrs[i].est for i in range(1, njack+1)]), axis=0)
        if ic_rmax > 0:
            corr.ic_calc(fit_range, p0, ic_rmax)
        corr_slices.append(corr)
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax, color=color,
                  label=f"z = [{info['zlo']:2.1f}, {info['zhi']:2.1f}]")
        popt, pcov = corr.fit_w(fit_range, p0, ax, color)
        print(popt, pcov)
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()


def w_a_plot(nz=5, fit_range=[0.01, 1], p0=[0.05, 1.7],
             path='w_z/', avgcounts=False, ic_rmax=0):
    """Plot desi angular correlation results."""

    plt.clf()
    fig, axes = plt.subplots(1, nz, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)

    for iz in range(nz):
        infile = f'{path}z{iz}.fits'
        t = Table.read(infile)
        corr = wcorr.Corr1d()
        corr.sep = t['meanr']
        corr.est = t['xi']
        corr.err = t['sigma_xi']
        corr.r1r2 = t['RR']
        # Check LS estimator NB pair counts already normalised
        # dd = t['DD']/t.meta['NGAL']**2
        # dr = t['DR']/t.meta['NGAL']/t.meta['NRAN']
        # rr = t['RR']/t.meta['NRAN']**2
        # xi_ls = (dd - 2*dr + rr)/rr
        # xi_ls = (t['DD'] - 2*t['DR'] + t['RR'])/t['RR']
            
        ax = axes[iz]
        ax.errorbar(t['meanr'], t['xi'], t['sigma_xi'])
        popt, pcov = corr.fit_w(fit_range, p0, ax)
        print(iz, popt, pcov)
        # ax.errorbar(t['meanr'], xi_ls, t['sigma_xi'], color='red')
        ax.text(0.5, 0.9, f'iz={iz}', transform=ax.transAxes)
    plt.loglog()
    axes[2].set_xlabel(r'$\theta$ / degrees')
    axes[0].set_ylabel(r'$w(\theta)$')
    plt.show()


def w_mag_plot(nm=5, fit_range=[0.001, 1], p0=[0.05, 1.7],
               path='S_w_m/', avgcounts=False, ic_rmax=0):
    """Plot treecorr angular correlation results in mag bins."""

    plt.clf()
    ax = plt.subplot(111)
 
    for im in range(nm):
        infile = f'{path}m{im}.fits'
        t = Table.read(infile)
        corr = wcorr.Corr1d()
        corr.sep = t['meanr']
        corr.est = t['xi']
        corr.err = t['sigma_xi']
        corr.r1r2 = t['RR']
        color = next(ax._get_lines.prop_cycler)['color']
        corr.plot(ax=ax, color=color,
                  label=f"m = [{t.meta['MLO']:3.1f}, {t.meta['MHI']:3.1f}]")
        popt, pcov = corr.fit_w(fit_range, p0, ax, color=color)
        print(im, popt, pcov)
        # ax.errorbar(t['meanr'], xi_ls, t['sigma_xi'], color='red')
        # ax.text(0.5, 0.9, f"m = [{t.meta['MLO']:3.1f}, {t.meta['MHI']:3.1f}]",
        #         transform=ax.transAxes)
    plt.loglog()
    plt.legend()
    ax.set_xlabel(r'$\theta$ / degrees')
    ax.set_ylabel(r'$w(\theta)$')
    plt.show()


def w_x_plot(nz=5, nm=4, fit_range=[0.01, 1], p0=[0.05, 1.7],
             path='bgs_x_l/', avgcounts=False, ic_rmax=0):
    """Plot desi-legacy angular cross-correlation results."""

    plt.clf()
    fig, axes = plt.subplots(nz, nm, sharex=True, sharey=True, num=1)
    fig.set_size_inches(5, 6)
    fig.subplots_adjust(hspace=0, wspace=0)

    print('iz   im    p      cov')
    for iz in range(nz):
        for im in range(nm):
            infile = f'{path}z{iz}_m{im}.fits'
            t = Table.read(infile)
            # corr = wcorr.Corr1d(t.meta['NGAL1'], t.meta['NGAL2'],
            #                     t.meta['NRAN1'], t.meta['NRAN2'],
            #                     t['DD'], t['DR'], t['RD'], t['RR'])
            corr = wcorr.Corr1d()
            corr.sep = t['meanr']
            corr.est = t['xi']
            corr.err = t['sigma_xi']
            corr.r1r2 = t['RR']

            # Check LS estimator NB pair counts already normalised
            # dd = t['DD']/t.meta['NGAL1']/t.meta['NGAL2']
            # dr = t['DR']/t.meta['NGAL1']/t.meta['NRAN2']
            # rd = t['RD']/t.meta['NRAN1']/t.meta['NGAL2']
            # rr = t['RR']/t.meta['NRAN1']/t.meta['NRAN2']
            # xi_ls = (dd - dr - rd +rr)/rr
            # xi_ls = (t['DD'] - t['DR'] - t['RD'] + t['RR'])/t['RR']
            
            ax = axes[iz, im]
            ax.errorbar(t['meanr'], t['xi'], t['sigma_xi'])
            popt, pcov = corr.fit_w(fit_range, p0, ax)
            print(iz, im, popt, pcov)
            # ax.errorbar(t['meanr'], xi_ls, t['sigma_xi'], color='red')
            ax.text(0.1, 0.8, f"z=[{t.meta['ZLO']}, {t.meta['ZHI']}], m=[{t.meta['MLO']}, {t.meta['MHI']}]", transform=ax.transAxes)
    plt.loglog()
    axes[4, 1].set_xlabel(r'$\theta$ / degrees')
    axes[2, 0].set_ylabel(r'$w(\theta)$')
    plt.show()


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

def cz_BGS_N_test():
    cz_test(galfile='BGS_ANY_N_clustering.dat.fits',
            adir='BGS_N_w_z', xdir='BGS_N_w_z_m', njack=10)
    
def cz_test(galfile='BGS_ANY_S_clustering.dat.fits',
            adir='BGS_S_w_z', xdir='BGS_S_w_z_m', njack=10, nz=12, nm=4,
            fit_range=[0.001, 1],
            p0=[0.05, 1.7], rmin=0.01, rmax=10):
    """Test cluster-z on DESI alone."""

    zmean, pmz, pmz_err, mlo, mhi = cluster_z.Nz(
        adir, xdir, njack, nz, nm, fit_range, p0, rmin, rmax)
    
    # Scale to reference N(z) in mag bins
    t = Table.read(galfile)
    z = t['Z']
    mag = 22.5 - 2.5*np.log10(t['FLUX_Z_DERED'])

    plt.clf()
    fig, axes = plt.subplots(1, nm, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 4)
    fig.subplots_adjust(hspace=0, wspace=0)
    for im in range(nm):
        ax = axes[im]
        sel = (mlo[im] <= mag) * (mag < mhi[im])
        zhist, edges = np.histogram(z[sel], bins=np.linspace(0.0, 0.6, 13))
        ax.stairs(zhist, edges)
        scale = zhist.sum()/pmz[:, im].sum()
        pmz[:, im] *= scale
        pmz_err[:, im] *= scale
        ax.errorbar(zmean, pmz[:, im], pmz_err[:, im])
        ax.text(0.1, 1.05, f"m=[{mlo[im]}, {mhi[im]}]",
                transform=ax.transAxes)
    axes[nm//2].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$N(z)$')
    plt.show()

def treecorr_test(ngal=1000, nran=10000, tmin=0.01, tmax=1, nbins=10):
    gcat1 = treecorr.Catalog(ra=rng.random(ngal), dec=rng.random(ngal),
                            ra_units='deg', dec_units='deg',
                            npatch=4)
    gcat2 = treecorr.Catalog(ra=rng.random(ngal), dec=rng.random(ngal),
                            ra_units='deg', dec_units='deg',
                            npatch=4)
    rcat = treecorr.Catalog(ra=rng.random(nran), dec=rng.random(nran),
                            ra_units='deg', dec_units='deg')
    dd = treecorr.NNCorrelation(
        min_sep=tmin, max_sep=tmax, nbins=nbins,
        sep_units='degrees', var_method='jackknife')
    dd.process(gcat1, gcat2)
    dr = treecorr.NNCorrelation(
        min_sep=tmin, max_sep=tmax, nbins=nbins,
        sep_units='degrees')
    dr.process(gcat1, rcat)
    rr = treecorr.NNCorrelation(
        min_sep=tmin, max_sep=tmax, nbins=nbins,
        sep_units='degrees')
    rr.process(rcat)
    dd.calculateXi(rr=rr, dr=dr)
    outfile = 'test.fits'
    dd.write(outfile, rr=rr, dr=dr)
    plt.clf()
    plt.scatter(gcat1.ra, gcat1.dec, c=gcat1.patch, s=1)
    plt.show()

def desi_legacy_overlap(desi_ranfile='DESI/BGS_ANY_S_0_clustering.ran.fits',
                        legacy_ranfile='Legacy/10.1/legacy_S_ran-0.fits'):
    """Test cluster_z. overlap code."""

    t = Table.read(desi_ranfile)
    c1 = SkyCoord(t['RA'], t['DEC'], frame='icrs')
    t = Table.read(legacy_ranfile)
    c2 = SkyCoord(t['RA']*u.deg, t['DEC']*u.deg, frame='icrs')
    hpmask = cluster_z.healpixMask(c1, c2)
    
def BGS_Legacy_cz_counts(spec_gal_file='DESI/BGS_ANY_S_clustering.dat.fits',
                         spec_ran_file='DESI/BGS_ANY_S_0_clustering.ran.fits',
                         phot_gal_file='Legacy/10.1/legacy_S.fits',
                         phot_ran_file='Legacy/10.1/legacy_S_ran-0.fits',
                         out_dir=''):
    """DESI x Legacy pair counts for cluster_z."""

    cluster_z.pair_counts(spec_gal_file, spec_ran_file, phot_gal_file, phot_ran_file, out_dir)
    
    
def BGS_S_cz_counts(spec_gal_file='DESI/BGS_ANY_S_clustering.dat.fits',
                  spec_ran_file='DESI/BGS_ANY_S_0_clustering.ran.fits',
                  phot_gal_file='DESI/BGS_ANY_S_clustering.dat.fits',
                  phot_ran_file='DESI/BGS_ANY_S_0_clustering.ran.fits',
                  out_dir='DESI/BGS_cz', zbins=np.linspace(0.0, 0.6, 13),
                  magbins=np.linspace(16, 20, 5)):
    """DESI x DESI pair counts for cluster_z."""

    def mag_fn(t):
        """Return magnitudes from table."""
        return 22.5 - 2.5*np.log10(t['FLUX_Z_DERED'].value)
    
    cluster_z.pair_counts(spec_gal_file, spec_ran_file, phot_gal_file, phot_ran_file, out_dir, mag_fn=mag_fn, zbins=zbins, magbins=magbins, npatch=10)
    

def BGS_Legacy_S_cz_counts(spec_gal_file='DESI/BGS_ANY_S_clustering.dat.fits',
                  spec_ran_file='DESI/BGS_ANY_S_0_clustering.ran.fits',
                  phot_gal_file='Legacy/10.1/legacy_S.fits',
                  phot_ran_file='Legacy/10.1/legacy_S_ran-0.fits',
                  out_dir='DESI/BGS_Legacy_S_cz', zbins=np.linspace(0.0, 0.6, 13),
                  magbins=np.linspace(16, 22, 7)):
    """DESI x DESI pair counts for cluster_z."""

    def mag_fn(t):
        """Return magnitudes from table."""
        return t['Z_MAG']
    
    cluster_z.pair_counts(spec_gal_file, spec_ran_file, phot_gal_file, phot_ran_file, out_dir, mag_fn=mag_fn, zbins=zbins, magbins=magbins, npatch=10)
    

def BGS_N_cz_counts(spec_gal_file='DESI/BGS_ANY_N_clustering.dat.fits',
                  spec_ran_file='DESI/BGS_ANY_N_0_clustering.ran.fits',
                  phot_gal_file='DESI/BGS_ANY_N_clustering.dat.fits',
                  phot_ran_file='DESI/BGS_ANY_N_0_clustering.ran.fits',
                  out_dir='DESI/BGS_N_cz', zbins=np.linspace(0.0, 0.6, 13),
                  magbins=np.linspace(16, 20, 5)):
    """DESI x DESI pair counts for cluster_z."""

    def mag_fn(t):
        """Return magnitudes from table."""
        return 22.5 - 2.5*np.log10(t['FLUX_Z_DERED'].value)
    
    cluster_z.pair_counts(spec_gal_file, spec_ran_file, phot_gal_file, phot_ran_file, out_dir, mag_fn=mag_fn, zbins=zbins, magbins=magbins, npatch=10)
    
