# Routines for analysing GAMA group mocks

def gama_group_mock_masks():
    """Masks for gama mock groups."""
    make_rect_mask((129.0, 141.0, -1.0, 3.0), 'G09.ply')
    make_rect_mask((174.0, 186.0, -2.0, 2.0), 'G12.ply')
    make_rect_mask((211.5, 223.5, -2.0, 2.0), 'G15.ply')


def wcounts(infile='G3CMockGalv06.fits', out_pref='w_mag/',
                ranfac=1, tmin=0.01, tmax=10, nbins=20,
                magbins=np.linspace(15.8, 19.8, 5)):
    """Angular pair counts in mag bins for GAMA group mocks."""

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    # Make separate catalogues for GAMA regions and volumes
    w_samp = np.zeros((27, nbins))
    t = Table.read(infile)
    limits = {'09': (129.0, 141.0, -1.0, 3.0), '12': (174.0, 186.0, -2.0, 2.0),
              '15': (211.5, 223.5, -2.0, 2.0)}
    isamp = 0
    for region in ('09', '12', '15'):
        selr = ((limits[region][0] <= t['RA'].value) *
                (t['RA'].value < limits[region][1]))
        ngal = len(t[selr])/9
        nran = int(ranfac * ngal)
        mask = pymangle.Mangle(f'G{region}.ply')
        rar, decr = mask.genrand_range(nran, *limits[region])
        rar, decr = rar.astype(float), decr.astype(float)
        print(f'Region {region} nran = {len(rar)}')
        info = {'Region': region, 'Nran': len(rar), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_G{region}.pkl'
        pool.apply_async(wcounts, args=(rar, decr, bins, info, outfile))
        for ivol in range(1, 10):
            for imag in range(len(magbins) - 1):
                mlo, mhi = magbins[imag], magbins[imag+1]
                sel = (selr * (t['Volume'] == ivol) *
                       (mlo <= t['Rpetro']) * (t['Rpetro'] < mhi))
                ra, dec = t['RA'][sel].value, t['DEC'][sel].value
                print(ivol, imag, len(ra))
                info = {'Region': region, 'Vol': ivol, 'mlo': mlo, 'mhi': mhi,
                        'Ngal': len(ra), 'Nran': len(rar), 'bins': bins, 'tcen': tcen}
                outfile = f'{out_pref}GG_G{region}_V{ivol}_m{imag}.pkl'
                pool.apply_async(wcounts, args=(ra, dec, bins, info, outfile))
                outfile = f'{out_pref}GR_G{region}_V{ivol}_m{imag}.pkl'
                pool.apply_async(wcounts, args=(ra, dec, bins, info,
                                 outfile, rar, decr))
    pool.close()
    pool.join()


def xi_gama_mock(infile='G3CMockGalv06.fits', out_pref='w_mag/',
                ranfac=1, rmin=0.01, rmax=100, nbins=20,
                zbins=np.linspace(0.0, 0.5, 6)):
    """3d pair counts in refshift bins for GAMA group mocks."""

    bins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    tcen = 10**(0.5*np.diff(np.log10(bins)) + np.log10(bins[:-1]))
    ncpu = mp.cpu_count()
    pool = mp.Pool(ncpu)
    print('Using', ncpu, 'CPUs')
    
    # Make separate catalogues for GAMA regions and volumes
    w_samp = np.zeros((27, nbins))
    t = Table.read(infile)
    limits = {'09': (129.0, 141.0, -1.0, 3.0), '12': (174.0, 186.0, -2.0, 2.0),
              '15': (211.5, 223.5, -2.0, 2.0)}
    isamp = 0
    for region in ('09', '12', '15'):
        selr = ((limits[region][0] <= t['RA'].value) *
                (t['RA'].value < limits[region][1]))
        ngal = len(t[selr])/9
        nran = int(ranfac * ngal)
        mask = pymangle.Mangle(f'G{region}.ply')
        rar, decr = mask.genrand_range(nran, *limits[region])
        rar, decr = rar.astype(float), decr.astype(float)
        print(f'Region {region} nran = {len(rar)}')
        info = {'Region': region, 'Nran': len(rar), 'bins': bins, 'tcen': tcen}
        outfile = f'{out_pref}RR_G{region}.pkl'
        pool.apply_async(wcounts, args=(rar, decr, bins, info, outfile))
        for ivol in range(1, 10):
            for imag in range(len(magbins) - 1):
                mlo, mhi = magbins[imag], magbins[imag+1]
                sel = (selr * (t['Volume'] == ivol) *
                       (mlo <= t['Rpetro']) * (t['Rpetro'] < mhi))
                ra, dec = t['RA'][sel].value, t['DEC'][sel].value
                print(ivol, imag, len(ra))
                info = {'Region': region, 'Vol': ivol, 'mlo': mlo, 'mhi': mhi,
                        'Ngal': len(ra), 'Nran': len(rar), 'bins': bins, 'tcen': tcen}
                outfile = f'{out_pref}GG_G{region}_V{ivol}_m{imag}.pkl'
                pool.apply_async(wcounts, args=(ra, dec, bins, info, outfile))
                outfile = f'{out_pref}GR_G{region}_V{ivol}_m{imag}.pkl'
                pool.apply_async(wcounts, args=(ra, dec, bins, info,
                                 outfile, rar, decr))
    pool.close()
    pool.join()


def w_gama_mock(nmag=4):
    """w(theta) from angular pair counts in mag bins for GAMA group mocks."""

    w = np.zeros((27, 4, 20))
    mlims = np.zeros((4, 2))
    isamp = 0
    for region in ('09', '12', '15'):
        infile = f'RR_G{region}.pkl'
        (info, RR_counts) = pickle.load(open(infile, 'rb'))
        for ivol in range(1, 10):
            for imag in range(nmag):
                infile = f'GG_G{region}_V{ivol}_m{imag}.pkl'
                (info, DD_counts) = pickle.load(open(infile, 'rb'))
                infile = f'GR_G{region}_V{ivol}_m{imag}.pkl'
                (info, DR_counts) = pickle.load(open(infile, 'rb'))
                w[isamp, imag, :] = Corrfunc.utils.convert_3d_counts_to_cf(
                    info['Ngal'], info['Ngal'], info['Nran'], info['Nran'],
                    DD_counts, DR_counts, DR_counts, RR_counts)
                mlims[imag, :] = np.array((info['mlo'], info['mhi']))
            isamp += 1

    wmean = np.mean(w, axis=0)
    werr = np.std(w, axis=0)
    plt.clf()
    for imag in range(nmag):
        plt.errorbar(info['tcen'], wmean[imag, :], werr[imag, :],
                     label=f'm = [{mlims[imag, 0]}, {mlims[imag, 1]}]')
    plt.loglog()
    plt.legend()
    plt.xlabel(r'$\theta$ / degrees')
    plt.ylabel(r'$w(\theta)$')
    plt.show()


def xi_z_counts_gama_mock(infile='12244.fits', mask_file='mask.ply',
         out_file='xi_z_12244.pkl', Mr_lims=[-22, -20],
         zbins=np.linspace(0, 1, 6), limits=(180, 200, 0, 20),
         ranfac=1, nra=3, ndec=3, rbins=np.logspace(-1, 2, 16),
         randist='poly', nthreads=2):
    """xi(r) for Euclid flagship in redshift bins."""
    # rcen = 10**(0.5*np.diff(np.log10(rbins)) + np.log10(rbins[:-1]))
    # print(rcen)

    t = Table.read(infile)

    # Select L* galaxies, well-sampled in all z slices
    sel = (Mr_lims[0] < t['abs_mag_r01']) *(t['abs_mag_r01'] < Mr_lims[1])
    t = t[sel]
    ra, dec, mag = t['ra_gal'], t['dec_gal'], t['rmag']
    z = t['true_redshift_gal']
    r = cosmo.dc(z)
    xi_dict_list = []

    plt.clf()
    fig, axes = plt.subplots(5, 1, num=1)
    fig.set_size_inches(4, 8)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for iz in range(len(zbins) - 1):
        dmin, dmax = cosmo.dc(zbins[iz]), cosmo.dc(zbins[iz+1])
        dbins = np.linspace(dmin, dmax, 21)
        sel = (zbins[iz] <= z) * (z < zbins[iz+1])
        galcat = Cat(ra[sel], dec[sel], r=r[sel], nthreads=nthreads)
        galcat.assign_jk(limits, nra, ndec)
        galcat.gen_cart()
        ngal = len(ra[sel])

        # Quadratic fit to N(d), constrained to pass through origin
        dhist, edges = np.histogram(galcat.r, dbins)
        dcen = edges[:-1] + 0.5*np.diff(edges)
        p = Polynomial.fit(dcen, dhist, deg=[1,2], w=dhist,
                           domain=[0, dcen[-1]], window=[0, dcen[-1]])
        
        nran = int(ranfac*ngal)
        mask = pymangle.Mangle(mask_file)
        rar, decr = mask.genrand_range(nran, *limits)
        if randist == 'shuffle':
            rr = rng.choice(r[sel], nran, replace=True)
        if randist == 'poly':
            rr = st_util.ran_fun(p, dmin, dmax, nran)
        rancat = Cat(rar.astype('float32'), decr.astype('float32'), r=rr)
        rancat.assign_jk(limits, nra, ndec)
        rancat.gen_cart()

        print(iz, galcat.nobj, rancat.nobj, 'galaxies and randoms')
        ax = axes[iz]
        dfit = p(dcen)
        ax.step(dcen, dhist)
        dhist, _ = np.histogram(rancat.r, dbins)
        ax.step(dcen, dhist)
        ax.plot(dcen, dfit)
        ax.text(0.05, 0.8, rf'z = {zbins[iz]:3.1f}-{zbins[iz+1]:3.1f}',
            transform=ax.transAxes)
        
        xi_dict = xir_jack(galcat, rancat, rbins)
        lbl = f'z = {zbins[iz]:3.1f}-{zbins[iz+1]:3.1f}'
        xi_dict.update({'lbl': lbl})
        xi_dict_list.append(xi_dict)
    pickle.dump(xi_dict_list, open(out_file, 'wb'))
    plt.show()

