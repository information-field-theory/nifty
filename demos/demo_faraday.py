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
    ..                  __   ____   __
    ..                /__/ /   _/ /  /_
    ..      __ ___    __  /  /_  /   _/  __   __
    ..    /   _   | /  / /   _/ /  /   /  / /  /
    ..   /  / /  / /  / /  /   /  /_  /  /_/  /
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  demo
    ..                               /______/

    NIFTY demo applying transformations to the "Faraday Map" [#]_.

    References
    ----------
    .. [#] N. Opermann et. al., "An improved map of the Galactic Faraday sky",
        Astronomy & Astrophysics, Volume 542, id.A93, 06/2012;
        `arXiv:1111.6186 <http://www.arxiv.org/abs/1111.6186>`_

"""
from __future__ import division
from nifty import *


about.warnings.off()
about.infos.off()


##-----------------------------------------------------------------------------

## spaces
h  = hp_space(128)
l  = lm_space(383)
g  = gl_space(384) ## nlon == 767
g_ = gl_space(384, nlon=768)
r  = rg_space([768, 384], dist=[1/360, 1/180])
r_ = rg_space([256, 128], dist=[1/120, 1/60])

## map
m = field(h, val=np.load("demo_faraday_map.npy"))

## projection operator
Sk = None

##-----------------------------------------------------------------------------



##=============================================================================

def run(projection=False, power=False):
    """
        Runs the demo.

        Parameters
        ----------
        projection : bool, *optional*
            Whether to additionaly include projections in the demo or not. If
            ``projection == True`` the projection operator `Sk` will be
            defined. (default: False)
        power : bool, *optional*
            Whether to additionaly show power spectra in the demo or not.
            (default: False)

    """
    global Sk
    ## start in hp_space
    m0 = m

    ## transform to lm_space
    m1 = m0.transform(l)
    if(projection):
        ## define projection operator
        Sk = projection_operator(l)
        ## project quadrupole
        m2 = Sk(m0, band=2)

    ## transform to gl_space
    m3 = m1.transform(g)

    ## transform to rg_space
    m4 = m1.transform(g_) ## auxiliary gl_space
    m4.cast_domain(r) ## rg_space cast
    m4.set_val(np.roll(m4.val[::-1, ::-1], g.nlon()//2, axis=1)) ## rearrange
    if(power):
        ## restrict to central window
        m5 = field(r_, val=m4[128:256, 256:512]).transform()

    ## plots
    m0.plot(title=r"$m$ on a HEALPix grid", vmin=-4, vmax=4, cmap=ncmap.fm())
    if(power):
        m1.plot(title=r"angular power spectrum of $m$", vmin=1E-2, vmax=1E+1, mono=False)
    if(projection):
        m2.plot(title=r"quadrupole of $m$ on a HEALPix grid", vmin=-4, vmax=4, cmap=ncmap.fm())
    m3.plot(title=r"$m$ on a spherical Gauss-Legendre grid", vmin=-4, vmax=4, cmap=ncmap.fm())
    m4.plot(title=r"$m$ on a regular 2D grid", vmin=-4, vmax=4, cmap=ncmap.fm())
    if(power):
        m5.plot(title=r"(restricted, binned) Fourier power spectrum of $m$", vmin=1E-3, vmax=1E+0, mono=False, log=True)

##=============================================================================

