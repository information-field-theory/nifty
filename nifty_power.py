## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
    ..                     __   ____   __
    ..                   /__/ /   _/ /  /_
    ..         __ ___    __  /  /_  /   _/  __   __
    ..       /   _   | /  / /   _/ /  /   /  / /  /
    ..      /  / /  / /  / /  /   /  /_  /  /_/  /
    ..     /__/ /__/ /__/ /__/    \___/  \___   /  power
    ..                                  /______/

    NIFTy offers a number of automatized routines for handling
    power spectra. It is possible to draw a field from a random distribution
    with a certain autocorrelation or, equivalently, with a certain
    power spectrum in its conjugate space (see :py:func:`field.random`). In
    NIFTy, it is usually assumed that such a field follows statistical
    homogeneity and isotropy. Fields which are only statistically homogeneous
    can also be created using the diagonal operator routine.

    In the moment, NIFTy offers one additional routine for power spectrum
    manipulation, the smooth_power function to smooth a power spectrum with a
    Gaussian convolution kernel. This can be necessary in cases where power
    spectra are reconstructed and reused in an interative algorithm, where
    too much statistical variation might effect severely the results.

"""

from __future__ import division
import numpy as np
import smoothing as gs


##-----------------------------------------------------------------------------

def smooth_power(power,kindex,exclude=1,sigma=-1):
    """
    Smoothes a power spectrum via convolution with a Gaussian kernel.

    Parameters
    ----------
    power : ndarray
        The power spectrum to be smoothed.

    kindex : ndarray
        The array specifying the coordinate indices in conjugate space.

    exclude : scalar
        Excludes the first power spectrum entries from smoothing, indicated by
        the given integer scalar (default=1, the monopol is not smoothed).

    smooth_length : scalar
        FWHM of Gaussian convolution kernel.

    Returns
    -------
    smoothpower : ndarray
        The smoothed power spectrum.

    """
    return gs.smooth_power(power,kindex,exclude=exclude,smooth_length=sigma)

##-----------------------------------------------------------------------------

