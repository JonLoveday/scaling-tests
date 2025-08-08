from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
# from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table, join
from kcorrect.kcorrect import Kcorrect
import math
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import pymangle
import scipy.optimize
import util
import wcorr

def shark_hmf(shark_cat='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/wide/waves_wide_gals.parquet',
              kcorr_file='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/wide/waves_wide_kcorr.fits',
              mlim=21.1, zmax=0.2, area=1134, Nmin=5, mag_col='total_ap_dust_Z_VISTA', massbins=np.linspace(11, 15, 17)):
    '''Shark HMF.
    For Vmax calculation, We assume that Nth brightest galaxy in each halo would
    remains the Nth brightest at different redshift, i.e. k-corrections 
    don't affect rank-ordering of brightness within groups.  
    (Strictly, one should re-evaluate which galaxy is Nth brightest at each
    trial redshift.)'''

    area_sr = area*(math.pi/180)**2
    area_corr = area_sr/(4*math.pi)
    metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'
    cosmo = util.CosmoLookup(H0=67.4, omega_l=0.685, zlimits=[0.001, zmax])


    t = Table.read(shark_cat)
    ktab = Table.read(kcorr_file)
    kc = Kcorrect(responses=ktab.meta['RESPONSES'])
    z0 = ktab.meta['Z0']
    refband = ktab.meta['REFBAND']

    t = join(t, ktab, keys_left='id_galaxy_sky', keys_right='CATAID', metadata_conflicts=metadata_conflicts)
    print(len(t), 'total galaxies read')
    t = t[(t[mag_col] < mlim) * (t['zobs'] < zmax) * (t['id_group_sky'] > 0)]
    print(len(t), 'grouped galaxies within magnitude and redshift limits')
    t_group = t.group_by('id_group_sky')
    print(len(t_group.groups.keys), 'galaxy groups')
    group_size = np.diff(t_group.groups.indices)
    sel = group_size >= Nmin
    t_sel = t_group.groups[sel]
    ngroup = len(t_sel.groups.keys)
    print(ngroup, 'galaxy groups have >=', Nmin, 'members')
    V, Vmax, lgM = np.zeros(ngroup), np.zeros(ngroup), np.zeros(ngroup)
    for igroup in range(ngroup):
        tg = t_sel.groups[igroup]
        lgM[igroup] = np.log10(tg['mvir_hosthalo'][0])
        tg.add_index(mag_col)
        gal = tg.iloc[Nmin-1]  # Nth brightest galaxy in the group
        redshift = gal['zobs']

        # Limiting redshift corresponding to m = mlim
        # Calculate absolute magnitude using distance modulus and k-correction,
        # so that volume limit calculation is consistent
        kcoeffs = gal['kcoeffs']
        kcorr = kc.kcorrect(redshift=redshift, coeffs=kcoeffs, band_shift=z0)[refband]
        mapp = gal[mag_col]
        Mabs = mapp - cosmo.dist_mod(redshift) - kcorr
        dmod = mlim - Mabs

        if (cosmo.dist_mod(zmax) +
            kc.kcorrect(redshift=zmax, coeffs=kcoeffs, band_shift=z0)[refband] < dmod):
            zlim = zmax
        else:
            zlim = scipy.optimize.brentq(
                    lambda z: cosmo.dist_mod(z) +
                    kc.kcorrect(redshift=z, coeffs=kcoeffs, band_shift=z0)[refband] - dmod,
                    redshift, zmax, xtol=1e-5, rtol=1e-5)
        V[igroup] = (area_corr * cosmo.V(redshift))
        Vmax[igroup] = (area_corr * cosmo.V(zlim))
    N, edges = np.histogram(lgM, bins=massbins)
    smf, edges = np.histogram(lgM, bins=massbins, weights=1.0/Vmax)
    smf /= np.diff(massbins)
    err = smf/N**0.5

    plt.clf()
    plt.hist(V/Vmax, bins=np.linspace(0.0, 1.0, 21))
    plt.xlabel('V/Vmax')
    plt.ylabel('N')
    plt.show()

    plt.clf()
    plt.errorbar(massbins[:-1] + 0.5*np.diff(massbins), smf, err)
    plt.semilogy()
    plt.xlabel('lg Mh')
    plt.ylabel(r'$\phi(M)$')
    plt.show()


def shark_smf(shark_cat='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/wide/waves_wide_gals.parquet',
              kcorr_file='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/wide/waves_wide_kcorr.fits',
              mlim=21.1, zmax=0.2, area=1134, mag_col='total_ap_dust_Z_VISTA', massbins=np.linspace(5, 13, 17)):
    '''Shark SMF.'''

    area_sr = area*(math.pi/180)**2
    area_corr = (area_sr/(4*math.pi)).value
    metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'
    cosmo = util.CosmoLookup(H0=67.4, omega_l=0.685, zlimits=[0.001, zmax])

    t = Table.read(shark_cat)
    ktab = Table.read(kcorr_file)
    kc = Kcorrect(responses=ktab.meta['RESPONSES'])
    z0 = ktab.meta['Z0']
    refband = ktab.meta['REFBAND']

    t = join(t, ktab, keys_left='id_galaxy_sky', keys_right='CATAID', metadata_conflicts=metadata_conflicts)
    print(len(t), 'total galaxies read')
    t = t[(t[mag_col] < mlim) * (t['zobs'] < zmax)]
    redshift = t['zobs']
    lgM = np.log10(t['mstars_disk'] + t['mstars_bulge'])
    plt.clf()
    plt.scatter(redshift, lgM, s=0.1)
    plt.xlabel('Redshift')
    plt.ylabel(r'$\lg M_*$')
    plt.show()

    # Calculate absolute magnitude using distance modulus and k-correction,
    # so that volume limit calculation is consistent
    kcoeffs = t['kcoeffs']
    kcorr = kc.kcorrect(redshift=redshift, coeffs=kcoeffs, band_shift=z0)[:, refband]
    mapp = t[mag_col]
    Mabs = mapp - cosmo.dist_mod(redshift) - kcorr
    dmod_lim = mlim - Mabs
    ngal = len(t)
    print(ngal, 'galaxies within magnitude and redshift limits')
    V = (area_corr * cosmo.V(redshift))
    Vmax = np.zeros(ngal)
    for igal in range(ngal):
        if (cosmo.dist_mod(zmax) +
            kc.kcorrect(redshift=zmax, coeffs=kcoeffs[igal, :], band_shift=z0)[refband] < dmod_lim[igal]):
            zlim = zmax
        else:
            zlim = scipy.optimize.brentq(
                    lambda z: cosmo.dist_mod(z) +
                    kc.kcorrect(redshift=z, coeffs=kcoeffs[igal, :], band_shift=z0)[refband] - dmod_lim[igal],
                    redshift[igal], zmax, xtol=1e-5, rtol=1e-5)
        Vmax[igal] = (area_corr * cosmo.V(zlim))
    N, edges = np.histogram(lgM, bins=massbins)
    smf, edges = np.histogram(lgM, bins=massbins, weights=1.0/Vmax)
    smf /= np.diff(massbins)
    err = smf/N**0.5

    plt.clf()
    plt.hist(V/Vmax, bins=np.linspace(0.0, 1.0, 21))
    plt.xlabel('V/Vmax')
    plt.ylabel('N')
    plt.show()

    plt.clf()
    plt.errorbar(massbins[:-1] + 0.5*np.diff(massbins), smf, err)
    plt.semilogy()
    plt.xlabel(r'$\lg M_*$')
    plt.ylabel(r'$\phi(M)$')
    plt.show()


def waves_deep_mask(limits=[339, 351, -35, -30], rect_mask='waves_deep.ply'):
    wcorr.make_rect_mask(limits, rect_mask)


def gen_rand(nran=1000000, maskfile='waves_deep.ply', outfile='waves_deep_ran.fits'):
    pymask = pymangle.Mangle(maskfile)
    # genrand_range does not work if limits wrap zero
    # ra, dec = pymask.genrand_range(nran, *limits)
    ra, dec = pymask.genrand(nran)
    t = Table((ra.astype(np.float64), dec.astype(np.float64)), names=('RA', 'DEC'))
    t.write(outfile, overwrite=True)

def wcounts_mag(galfile='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/deep/waves_deep_gals.parquet',
                ranfile='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/deep/waves_deep_ran.fits',
                out_dir='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/deep/',
                tmin=0.01, tmax=10, nbins=20,
                magbins=np.linspace(18, 23, 6), ra_col='ra', dec_col='dec',
                mag_col='total_ap_dust_Z_VISTA', npatch=9):
    """Angular pair counts in magnitude bins using treecorr."""
    wcorr.wcounts_mag(galfile=galfile,
                ranfile=ranfile,
                out_dir=out_dir,
                tmin=tmin, tmax=tmax, nbins=nbins,
                magbins=magbins, ra_col=ra_col, dec_col=dec_col,
                mag_col=mag_col, npatch=npatch)

# shark_smf(Nmin=5)
shark_hmf(shark_cat='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/deep/waves_deep_gals.parquet',
        kcorr_file='/Users/loveday/Data/4MOST/WAVES/Shark/v0.3.0/deep/waves_deep_kcorr.fits',
        mlim=21.25, zmax=1.0, area=51, Nmin=3, mag_col='total_ap_dust_Z_VISTA',
        massbins=np.linspace(11, 15, 17))