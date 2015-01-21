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

import copy_reg as cr
from types import MethodType as mt


##-----------------------------------------------------------------------------

def _pickle_method(method):
    fct_name = method.im_func.__name__
    obj = method.im_self
    cl = method.im_class
    ## handle mangled function name
    if(fct_name.startswith("__"))and(not fct_name.endswith("__")):
        cl_name = cl.__name__.lstrip("_")
        fct_name = "_" + cl_name + fct_name
    return _unpickle_method, (fct_name, obj, cl)

##-----------------------------------------------------------------------------

def _unpickle_method(fct_name, obj, cl):
    for oo in cl.__mro__:
        try:
            fct = oo.__dict__[fct_name]
        except(KeyError):
            pass
        else:
            break
    return fct.__get__(obj, cl)

##-----------------------------------------------------------------------------

## enable instance methods pickling
cr.pickle(mt, _pickle_method, _unpickle_method)

