import numpy as np
from numpy.random import default_rng
import Corrfunc

def test():
    """Simple test of corrfunc pair counting."""

    print(Corrfunc.__version__)
    nthreads = 1
    ngal, nran = 1000, 1000
    boxsize = 1000
    bins=np.logspace(-1, 2, 16)
    rng = default_rng()
    x, y, z = boxsize*rng.random(ngal), boxsize*rng.random(ngal), boxsize*rng.random(ngal)
    counts = Corrfunc.theory.DD(1, nthreads, bins, x, y, z, periodic=False)
    print(counts)
