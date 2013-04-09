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
    ..  /__/ /__/ /__/ /__/    \___/  \___   /  core
    ..                               /______/

    .. The NIFTY project homepage is http://www.mpa-garching.mpg.de/ift/nifty/

    NIFTY [#]_, "Numerical Information Field Theory", is a versatile
    library designed to enable the development of signal inference algorithms
    that operate regardless of the underlying spatial grid and its resolution.
    Its object-oriented framework is written in Python, although it accesses
    libraries written in Cython, C++, and C for efficiency.

    NIFTY offers a toolkit that abstracts discretized representations of
    continuous spaces, fields in these spaces, and operators acting on fields
    into classes. Thereby, the correct normalization of operations on fields is
    taken care of automatically without concerning the user. This allows for an
    abstract formulation and programming of inference algorithms, including
    those derived within information field theory. Thus, NIFTY permits its user
    to rapidly prototype algorithms in 1D and then apply the developed code in
    higher-dimensional settings of real world problems. The set of spaces on
    which NIFTY operates comprises point sets, n-dimensional regular grids,
    spherical spaces, their harmonic counterparts, and product spaces
    constructed as combinations of those.

    Class & Feature Overview
    ------------------------
    The NIFTY library features three main classes: **spaces** that represent
    certain grids, **fields** that are defined on spaces, and **operators**
    that apply to fields.

    Overview of all (core) classes:

    .. - switch
    .. - notification
    .. - _about
    .. - random
    .. - space
    ..     - point_space
    ..     - rg_space
    ..     - lm_space
    ..     - gl_space
    ..     - hp_space
    ..     - nested_space
    .. - field
    .. - operator
    ..     - diagonal_operator
    ..         - power_operator
    ..     - projection_operator
    ..     - vecvec_operator
    ..     - response_operator
    .. - probing
    ..     - trace_probing
    ..     - diagonal_probing

    .. automodule:: nifty

    :py:class:`space`

    - :py:class:`point_space`
    - :py:class:`rg_space`
    - :py:class:`lm_space`
    - :py:class:`gl_space`
    - :py:class:`hp_space`
    - :py:class:`nested_space`

    :py:class:`field`

    :py:class:`operator`

    - :py:class:`diagonal_operator`
        - :py:class:`power_operator`
    - :py:class:`projection_operator`
    - :py:class:`vecvec_operator`
    - :py:class:`response_operator`

    :py:class:`probing`

    - :py:class:`trace_probing`
    - :py:class:`diagonal_probing`

    References
    ----------
    .. [#] Selig et al., "NIFTY -- Numerical Information Field Theory --
        a versatile Python library for signal inference", submitted to A&A,
        2013; `arXiv:1301.4499 <http://www.arxiv.org/abs/1301.4499>`_

"""
from __future__ import division
import os
#import sys
from sys import stdout as so
import numpy as np
import gfft as gf
import healpy as hp
import libsharp_wrapper_gl as gl
import smoothing as gs
import powerspectrum as gp
import pylab as pl
from matplotlib.colors import LogNorm as ln
from matplotlib.ticker import LogFormatter as lf
from multiprocessing import Pool as mp


pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679


##-----------------------------------------------------------------------------

class switch(object):
    """
        ..                            __   __               __
        ..                          /__/ /  /_            /  /
        ..     _______  __     __   __  /   _/  _______  /  /___
        ..   /  _____/ |  |/\/  / /  / /  /   /   ____/ /   _   |
        ..  /_____  /  |       / /  / /  /_  /  /____  /  / /  /
        .. /_______/   |__/\__/ /__/  \___/  \______/ /__/ /__/  class

        NIFTY support class for switches.

        Parameters
        ----------
        default : bool
            Default status of the switch (default: False).

        See Also
        --------
        notification : A derived class for displaying notifications.

        Examples
        --------
        >>> option = switch()
        >>> option.status
        False
        >>> option
        OFF
        >>> print(option)
        OFF
        >>> option.on()
        >>> print(option)
        ON

        Attributes
        ----------
        status : bool
            Status of the switch.

    """
    def __init__(self,default=False):
        """
            Initilizes the switch and sets the `status`

            Parameters
            ----------
            default : bool
                Default status of the switch (default: False).

        """
        self.status = bool(default)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on(self):
        """
            Switches the `status` to True.

        """
        self.status = True

    def off(self):
        """
            Switches the `status` to False.

        """
        self.status = False


    def toggle(self):
        """
            Switches the `status`.

        """
        self.status = not self.status

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        if(self.status):
            return "ON"
        else:
            return "OFF"

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class notification(switch):
    """
        ..                           __     __   ____   __                       __     __
        ..                         /  /_  /__/ /   _/ /__/                     /  /_  /__/
        ..     __ ___    ______   /   _/  __  /  /_   __   _______   ____ __  /   _/  __   ______    __ ___
        ..   /   _   | /   _   | /  /   /  / /   _/ /  / /   ____/ /   _   / /  /   /  / /   _   | /   _   |
        ..  /  / /  / /  /_/  / /  /_  /  / /  /   /  / /  /____  /  /_/  / /  /_  /  / /  /_/  / /  / /  /
        .. /__/ /__/  \______/  \___/ /__/ /__/   /__/  \______/  \______|  \___/ /__/  \______/ /__/ /__/  class

        NIFTY support class for notifications.

        Parameters
        ----------
        default : bool
            Default status of the switch (default: False).
        ccode : string
            Color code as string (default: "\033[0m"). The surrounding special
            characters are added if missing.

        Notes
        -----
        The color code is a special ANSI escape code, for a list of valid codes
        see [#]_. Multiple codes can be combined by seperating them with a
        semicolon ';'.

        References
        ----------
        .. [#] Wikipedia, `ANSI escape code <http://en.wikipedia.org/wiki/ANSI_escape_code#graphics>`_.

        Examples
        --------
        >>> note = notification()
        >>> note.status
        True
        >>> note.cprint("This is noteworthy.")
        This is noteworthy.
        >>> note.cflush("12"); note.cflush('3')
        123
        >>> note.off()
        >>> note.cprint("This is noteworthy.")
        >>>

        Raises
        ------
        TypeError
            If `ccode` is no string.

        Attributes
        ----------
        status : bool
            Status of the switch.
        ccode : string
            Color code as string.

    """
    _code = "\033[0m" ## "\033[39;49m"

    def __init__(self,default=True,ccode="\033[0m"):
        """
            Initializes the notification and sets `status` and `ccode`

            Parameters
            ----------
            default : bool
                Default status of the switch (default: False).
            ccode : string
                Color code as string (default: "\033[0m"). The surrounding
                special characters are added if missing.

            Raises
            ------
            TypeError
                If `ccode` is no string.

        """
        self.status = bool(default)

        ## check colour code
        if(not isinstance(ccode,str)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        if(ccode[0]!="\033"):
            ccode = "\033"+ccode
        if(ccode[1]!='['):
            ccode = ccode[0]+'['+ccode[1:]
        if(ccode[-1]!='m'):
            ccode = ccode+'m'
        self.ccode = ccode

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_ccode(self,newccode=None):
        """
            Resets the the `ccode` string.

            Parameters
            ----------
            newccode : string
                Color code as string (default: "\033[0m"). The surrounding
                characters "\033", '[', and 'm' are added if missing.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                If `ccode` is no string.

            Examples
            --------
            >>> note = notification()
            >>> note.set_ccode("31;1") ## "31;1" corresponds to red and bright

        """
        if(newccode is None):
            newccode = self._code
        else:
            ## check colour code
            if(not isinstance(newccode,str)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            if(newccode[0]!="\033"):
                newccode = "\033"+newccode
            if(newccode[1]!='['):
                newccode = newccode[0]+'['+newccode[1:]
            if(newccode[-1]!='m'):
                newccode = newccode+'m'
        self.ccode = newccode

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def cstring(self,subject):
        """
            Casts an object to a string and augments that with a colour code.

            Parameters
            ----------
            subject : {string, object}
                String to be augmented with a color code. A given object is
                cast to its string representation by :py:func:`str`.

            Returns
            -------
            cstring : string
                String augmented with a color code.

        """
        return self.ccode+str(subject)+self._code

    def cflush(self,subject):
        """
            Flushes an object in its colour coded sting representation to the
            standard output (*without* line break).

            Parameters
            ----------
            subject : {string, object}
                String to be flushed. A given object is
                cast to a string by :py:func:`str`.

            Returns
            -------
            None

        """
        if(self.status):
            so.write(self.cstring(subject))
            so.flush()

    def cprint(self,subject):
        """
            Flushes an object in its colour coded sting representation to the
            standard output (*with* line break).

            Parameters
            ----------
            subject : {string, object}
                String to be flushed. A given object is
                cast to a string by :py:func:`str`.

            Returns
            -------
            None

        """
        if(self.status):
            so.write(self.cstring(subject)+"\n")
            so.flush()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        if(self.status):
            return self.cstring("ON")
        else:
            return self.cstring("OFF")

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class _about(object): ## nifty support class for global settings
    """
        NIFTY support class for global settings.

        .. warning::
            Turning off the `_error` notification will suppress all NIFTY error
            strings (not recommended).

        Examples
        --------
        >>> from nifty import *
        >>> about
        nifty version 0.2.0
        >>> print(about)
        nifty version 0.2.0
        - errors          = ON (immutable)
        - warnings        = ON
        - infos           = OFF
        - multiprocessing = ON
        - hermitianize    = ON
        - lm2gl           = ON
        >>> about.infos.on()
        >>> about.about.save_config()

        >>> from nifty import *
        INFO: nifty version 0.2.0
        >>> print(about)
        nifty version 0.2.0
        - errors          = ON (immutable)
        - warnings        = ON
        - infos           = ON
        - multiprocessing = ON
        - hermitianize    = ON
        - lm2gl           = ON

        Attributes
        ----------
        warnings : notification
            Notification instance controlling whether warings shall be printed.
        infos : notification
            Notification instance controlling whether information shall be
            printed.
        multiprocessing : switch
            Switch instance controlling whether multiprocessing might be
            performed.
        hermitianize : switch
            Switch instance controlling whether hermitian symmetry for certain
            :py:class:`rg_space` instances is inforced.
        lm2gl : switch
            Switch instance controlling whether default target of a
            :py:class:`lm_space` instance is a :py:class:`gl_space` or a
            :py:class:`hp_space` instance.

    """
    def __init__(self):
        """
            Initializes the _about and sets the attributes.

        """
        ## version
        self._version = "0.3.1"

        ## switches and notifications
        self._errors = notification(default=True,ccode=notification._code)
        self.warnings = notification(default=True,ccode=notification._code)
        self.infos =  notification(default=False,ccode=notification._code)
        self.multiprocessing = switch(default=True)
        self.hermitianize = switch(default=True)
        self.lm2gl = switch(default=True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def load_config(self,force=True):
        """
            Reads the configuration file "~/.nifty/nifty_config".

            Parameters
            ----------
            force : bool
                Whether to cause an error if the file does not exsist or not.

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If the configuration file is malformed.
            OSError
                If the configuration file does not exist.

        """
        nconfig = os.path.expanduser('~')+"/.nifty/nifty_config"
        if(os.path.isfile(nconfig)):
            rawconfig = []
            with open(nconfig,'r') as configfile:
                for ll in configfile:
                    if(not ll.startswith('#')):
                        rawconfig += ll.split()
            try:
                self._errors = notification(default=True,ccode=rawconfig[0])
                self.warnings = notification(default=int(rawconfig[1]),ccode=rawconfig[2])
                self.infos =  notification(default=int(rawconfig[3]),ccode=rawconfig[4])
                self.multiprocessing = switch(default=int(rawconfig[5]))
                self.hermitianize = switch(default=int(rawconfig[6]))
                self.lm2gl = switch(default=int(rawconfig[7]))
            except(IndexError):
                raise ValueError(about._errors.cstring("ERROR: '"+nconfig+"' damaged."))
        elif(force):
            raise OSError(about._errors.cstring("ERROR: '"+nconfig+"' nonexisting."))

    def save_config(self):
        """
            Writes to the configuration file "~/.nifty/nifty_config".

            Returns
            -------
            None

        """
        rawconfig = [self._errors.ccode[2:-1],str(int(self.warnings.status)),self.warnings.ccode[2:-1],str(int(self.infos.status)),self.infos.ccode[2:-1],str(int(self.multiprocessing.status)),str(int(self.hermitianize.status)),str(int(self.lm2gl.status))]

        nconfig = os.path.expanduser('~')+"/.nifty/nifty_config"
        if(os.path.isfile(nconfig)):
            rawconfig = [self._errors.ccode[2:-1],str(int(self.warnings.status)),self.warnings.ccode[2:-1],str(int(self.infos.status)),self.infos.ccode[2:-1],str(int(self.multiprocessing.status)),str(int(self.hermitianize.status)),str(int(self.lm2gl.status))]
            nconfig = os.path.expanduser('~')+"/.nifty/nifty_config"

            with open(nconfig,'r') as sourcefile:
                with open(nconfig+"_",'w') as targetfile:
                    for ll in sourcefile:
                        if(ll.startswith('#')):
                            targetfile.write(ll)
                        else:
                            ll = ll.replace(ll.split()[0],rawconfig[0]) ## one(!) per line
                            rawconfig = rawconfig[1:]
                            targetfile.write(ll)
            os.rename(nconfig+"_",nconfig) ## overwrite old congiguration
        else:
            if(not os.path.exists(os.path.expanduser('~')+"/.nifty")):
                os.makedirs(os.path.expanduser('~')+"/.nifty")
            with open(nconfig,'w') as targetfile:
                for rr in rawconfig:
                    targetfile.write(rr+"\n") ## one(!) per line

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "nifty version "+self._version

    def __str__(self):
        return "nifty version "+self._version+"\n- errors          = "+self._errors.cstring("ON")+" (immutable)\n- warnings        = "+str(self.warnings)+"\n- infos           = "+str(self.infos)+"\n- multiprocessing = "+str(self.multiprocessing)+"\n- hermitianize    = "+str(self.hermitianize)+"\n- lm2gl           = "+str(self.lm2gl)

##-----------------------------------------------------------------------------

## set global instance
about = _about()
about.load_config(force=False)
about.infos.cprint("INFO: "+about.__repr__())





##-----------------------------------------------------------------------------

class random(object):
    """
        ..                                          __
        ..                                        /  /
        ..       _____   ____ __   __ ___    ____/  /  ______    __ ____ ___
        ..     /   __/ /   _   / /   _   | /   _   / /   _   | /   _    _   |
        ..    /  /    /  /_/  / /  / /  / /  /_/  / /  /_/  / /  / /  / /  /
        ..   /__/     \______| /__/ /__/  \______|  \______/ /__/ /__/ /__/  class

        NIFTY (static) class for pseudo random number generators.

    """
    __init__ = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def arguments(domain,**kwargs):
        """
            Analyses the keyword arguments for supported or necessary ones.

            Parameters
            ----------
            domain : space
                Space wherein the random field values live.
            random : string, *optional*
                Specifies a certain distribution to be drwan from using a
                pseudo random number generator. Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given
                    standard deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

            dev : {scalar, list, ndarray, field}, *optional*
                Standard deviation of the normal distribution if
                ``random == "gau"`` (default: None).
            var : {scalar, list, ndarray, field}, *optional*
                Variance of the normal distribution (outranks the standard
                deviation) if ``random == "gau"`` (default: None).
            spec : {scalar, list, ndarray, field}, *optional*
                Power spectrum for ``random == "syn"`` (default: 1).
            size : integer, *optional*
                Number of irreducible bands for ``random == "syn"``
                (default: None).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each irreducible band (default: None).
            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution if ``random == "uni"``
                (default: 0).
            vmax : {scalar, list, ndarray, field}, *optional*
                Upper limit of the uniform distribution if ``random == "uni"``
                (default: 1).

            Returns
            -------
            arg : list
                Ordered list of arguments (to be processed in
                ``get_random_values`` of the domain).

            Raises
            ------
            KeyError
                If the `random` key is not supporrted.

        """
        if(kwargs.has_key("random")):
            key = kwargs.get("random")
        else:
            return None

        if(key=="pm1"):
            return [key]

        elif(key=="gau"):
            if(kwargs.has_key("mean")):
                mean = domain.enforce_values(kwargs.get("mean"),extend=False)
            else:
                mean = None
            if(kwargs.has_key("dev")):
                dev = domain.enforce_values(kwargs.get("dev"),extend=False)
            else:
                dev = None
            if(kwargs.has_key("var")):
                var = domain.enforce_values(kwargs.get("var"),extend=False)
            else:
                var = None
            return [key,mean,dev,var]

        elif(key=="syn"):
            if(kwargs.has_key("spec")):
                spec = kwargs.get("spec")
            else:
                spec = 1
            size = None
            kpack = None
            if(kwargs.has_key("size")):
                size = kwargs.get("size")
            if(kwargs.has_key("pindex"))and(kwargs.has_key("kindex")):
                kpack = [kwargs.get("pindex"),kwargs.get("kindex")]
                size = len(kpack[1])
            return [key,domain.enforce_power(spec,size=size),kpack]

        elif(key=="uni"):
            if(kwargs.has_key("vmin")):
                vmin = domain.enforce_values(kwargs.get("vmin"),extend=False)
            else:
                vmin = 0
            if(kwargs.has_key("vmax")):
                vmax = domain.enforce_values(kwargs.get("vmax"),extend=False)
            else:
                vmax = 1
            return [key,vmin,vmax]

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(key)+"'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def pm1(datatype=np.int,shape=1):
        """
            Generates random field values according to an uniform distribution
            over {+1,-1} or {+1,+i,-1,-i}, respectively.

            Parameters
            ----------
            datatype : type, *optional*
                Data type of the field values (default: np.int).
            shape : {integer, tuple, list, ndarray}, *optional*
                Split up dimension of the space (default: 1).

            Returns
            -------
            x : ndarray
                Random field values (with correct dtype and shape).

        """
        size = np.prod(shape,axis=0,dtype=np.int,out=None)

        if(datatype in [np.complex64,np.complex128]):
            x = np.array([1+0j,0+1j,-1+0j,0-1j],dtype=datatype)[np.random.randint(4,high=None,size=size)]
        else:
            x = 2*np.random.randint(2,high=None,size=size)-1

        return x.astype(datatype).reshape(shape,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def gau(datatype=np.float64,shape=1,mean=None,dev=None,var=None):
        """
            Generates random field values according to a normal distribution.

            Parameters
            ----------
            datatype : type, *optional*
                Data type of the field values (default: np.float64).
            shape : {integer, tuple, list, ndarray}, *optional*
                Split up dimension of the space (default: 1).
            mean : {scalar, ndarray}, *optional*
                Mean of the normal distribution (default: 0).
            dev : {scalar, ndarray}, *optional*
                Standard deviation of the normal distribution (default: 1).
            var : {scalar, ndarray}, *optional*
                Variance of the normal distribution (outranks the standard
                deviation) (default: None).

            Returns
            -------
            x : ndarray
                Random field values (with correct dtype and shape).

            Raises
            ------
            ValueError
                If the array dimension of `mean`, `dev` or `var` mismatch with
                `shape`.

        """
        size = np.prod(shape,axis=0,dtype=np.int,out=None)

        if(datatype in [np.complex64,np.complex128]):
            x = np.empty(size,dtype=datatype,order='C')
            x.real = np.random.normal(loc=0,scale=np.sqrt(0.5),size=size)
            x.imag = np.random.normal(loc=0,scale=np.sqrt(0.5),size=size)
        else:
            x = np.random.normal(loc=0,scale=1,size=size)

        if(var is not None):
            if(np.size(var)==1):
                x *= np.sqrt(np.abs(var))
            elif(np.size(var)==size):
                x *= np.sqrt(np.absolute(var).flatten(order='C'))
            else:
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(var))+" <> "+str(size)+" )."))
        elif(dev is not None):
            if(np.size(dev)==1):
                x *= np.abs(dev)
            elif(np.size(dev)==size):
                x *= np.absolute(dev).flatten(order='C')
            else:
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(dev))+" <> "+str(size)+" )."))
        if(mean is not None):
            if(np.size(mean)==1):
                x += mean
            elif(np.size(mean)==size):
                x += np.array(mean).flatten(order='C')
            else:
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(mean))+" <> "+str(size)+" )."))

        return x.astype(datatype).reshape(shape,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def uni(datatype=np.float64,shape=1,vmin=0,vmax=1):
        """
            Generates random field values according to an uniform distribution
            over [vmin,vmax[.

            Parameters
            ----------
            datatype : type, *optional*
                Data type of the field values (default: np.float64).
            shape : {integer, tuple, list, ndarray}, *optional*
                Split up dimension of the space (default: 1).

            vmin : {scalar, list, ndarray, field}, *optional*
                Lower limit of the uniform distribution (default: 0).
            vmax : {scalar, list, ndarray, field}, *optional*
                Upper limit of the uniform distribution (default: 1).

            Returns
            -------
            x : ndarray
                Random field values (with correct dtype and shape).

        """
        size = np.prod(shape,axis=0,dtype=np.int,out=None)
        if(np.size(vmin)>1):
            vmin = np.array(vmin).flatten(order='C')
        if(np.size(vmax)>1):
            vmax = np.array(vmax).flatten(order='C')

        if(datatype in [np.complex64,np.complex128]):
            x = np.empty(size,dtype=datatype,order='C')
            x.real = (vmax-vmin)*np.random.random(size=size)+vmin
            x.imag = (vmax-vmin)*np.random.random(size=size)+vmin
        elif(datatype in [np.int8,np.int16,np.int32,np.int64]):
            x = np.random.randint(min(vmin,vmax),high=max(vmin,vmax),size=size)
        else:
            x = (vmax-vmin)*np.random.random(size=size)+vmin

        return x.astype(datatype).reshape(shape,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.random>"

##-----------------------------------------------------------------------------





##=============================================================================

class space(object):
    """
        ..     _______   ______    ____ __   _______   _______
        ..   /  _____/ /   _   | /   _   / /   ____/ /   __  /
        ..  /_____  / /  /_/  / /  /_/  / /  /____  /  /____/
        .. /_______/ /   ____/  \______|  \______/  \______/  class
        ..          /__/

        NIFTY base class for spaces and their discretizations.

        The base NIFTY space class is an abstract class from which other
        specific space subclasses, including those preimplemented in NIFTY
        (e.g. the regular grid class) must be derived.

        Parameters
        ----------
        para : {single object, list of objects}, *optional*
            This is a freeform list of parameters that derivatives of the space
            class can use (default: 0).
        datatype : numpy.dtype, *optional*
            Data type of the field values for a field defined on this space
            (default: numpy.float64).

        See Also
        --------
        point_space :  A class for unstructured lists of numbers.
        rg_space : A class for regular cartesian grids in arbitrary dimensions.
        hp_space : A class for the HEALPix discretization of the sphere
            [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the sphere
            [#]_.
        lm_space : A class for spherical harmonic components.
        nested_space : A class for product spaces.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        Attributes
        ----------
        para : {single object, list of objects}
            This is a freeform list of parameters that derivatives of the space class can use.
        datatype : numpy.dtype
            Data type of the field values for a field defined on this space.
        discrete : bool
            Whether the space is inherently discrete (true) or a discretization
            of a continuous space (false).
        vol : numpy.ndarray
            An array of pixel volumes, only one component if the pixels all
            have the same volume.
    """
    def __init__(self,para=0,datatype=None):
        """
            Sets the attributes for a space class instance.

            Parameters
            ----------
            para : {single object, list of objects}, *optional*
                This is a freeform list of parameters that derivatives of the
                space class can use (default: 0).
            datatype : numpy.dtype, *optional*
                Data type of the field values for a field defined on this space
                (default: numpy.float64).

            Returns
            -------
            None
        """
        if(np.isscalar(para)):
            para = np.array([para],dtype=np.int)
        else:
            para = np.array(para,dtype=np.int)
        self.para = para

        ## check data type
        if(datatype is None):
            datatype = np.float64
        elif(datatype not in [np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.complex64,np.complex128]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.float64
        self.datatype = datatype

        self.discrete = True
        self.vol = np.real(np.array([1],dtype=self.datatype))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension split up, i.e. the numbers of
                pixels in each direction, or not (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'dim'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'dof'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {float, numpy.ndarray, nifty.field}
                Fiducial power spectrum from which a valid power spectrum is to
                be calculated. Scalars are interpreted as constant power
                spectra.

            Returns
            -------
            spec : numpy.ndarray
                Valid power spectrum.

            Other parameters
            ----------------
            size : int, *optional*
                Number of bands the power spectrum shall have (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band.

        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'enforce_power'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Provides the indexing array of the power spectrum.

            Provides either an array giving for each component of a field the
            corresponding index of a power spectrum (if ``irreducible==False``)
            or two arrays containing the scales of the modes and the numbers of
            modes with this scale (if ``irreducible==True``).

            Parameters
            ----------
            irreducible : bool, *optional*
                Whether to return two arrays containing the scales and
                corresponding number of represented modes (if True) or the
                indexing array (if False) (default: False).

            Returns
            -------
            kindex : numpy.ndarray
                Scale of each band, returned only if ``irreducible==True``.
            rho : numpy.ndarray
                Number of modes per scale represented in the discretization,
                returned only if ``irreducible==True``.
            pindex : numpy.ndarray
                Indexing array giving the power spectrum index for each
                represented mode, returned only if ``irreducible==False``.

            Notes
            -----
            The indexing array is of the same shape as a field living in this
            space and contains the indices of the associated bands.
            kindex and rho are each one-dimensional arrays.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_power_index'."))

    def get_power_undex(self,pindex=None):
        """
            Provides the unindexing list for an indexed power spectrum.

            Parameters
            ----------
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index for each
                represented mode.

            Returns
            -------
            pundex : list
                Unindexing list undoing power indexing.

            Notes
            -----
            Indexing with the unindexing list undoes the indexing with the
            indexing array; i.e., ``x == x[pindex][pundex]``.

            See also
            --------
            get_power_index

        """
        if(pindex is None):
            pindex = self.get_power_index(irreducible=False)
        return list(np.unravel_index(np.unique(pindex,return_index=True,return_inverse=False)[1],pindex.shape,order='C'))

    ## TODO: *new* `space` attribute `power` from *new* `_power` class carrying power functions and if-needed-computed index arrays
    def get_power_indices(self):
        """

        """
        pindex = self.get_power_index(irreducible=False)
        kindex,rho = self.get_power_index(irreducible=True)

#        pindex,kindex,rho = gp.reset_power_indices(pindex,kindex,rho)

        pundex = list(np.unravel_index(np.unique(pindex,return_index=True,return_inverse=False)[1],pindex.shape,order='C'))
        return pindex,pundex,kindex,rho ## FIXME: order

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_shape(self,x):
        """
            Shapes an array of valid field values correctly, according to the
            specifications of the space instance.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values to be put into shape.

            Returns
            -------
            y : numpy.ndarray
                Correctly shaped array.
        """
        x = np.array(x)

        if(np.size(x)!=self.dim(split=False)):
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(x))+" <> "+str(self.dim(split=False))+" )."))
#        elif(not np.all(np.array(np.shape(x))==self.dim(split=True))):
#            about.warnings.cprint("WARNING: reshaping forced.")

        return x.reshape(self.dim(split=True),order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, according to the
            constraints from the space instance.

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray
                Array containing the valid field values.

            Other parameters
            ----------------
            extend : bool, *optional*
                Whether a scalar is extented to a constant array or not
                (default: True).
        """
        if(isinstance(x,field)):
            if(self==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    x = x.val
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            if(np.size(x)==1):
                if(extend):
                    x = self.datatype(x)*np.ones(self.dim(split=True),dtype=self.datatype,order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x],dtype=self.datatype)
                    else:
                        x = np.array(x,dtype=self.datatype)
            else:
                x = self.enforce_shape(np.array(x,dtype=self.datatype))

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {float, numpy.ndarray}, *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            x = np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the space or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, usually either the position basis or the basis of
            harmonic eigenmodes.

            Parameters
            ----------
            coname : string, *optional*
                String specifying a desired codomain (default: None).
            cozerocenter : {bool, numpy.ndarray}, *optional*
                Whether or not the grid is zerocentered for each axis or not
                (default: None).
            conest : list, *optional*
                List of nested spaces of the codomain (default: None).
            coorder : list, *optional*
                Permutation of the list of nested spaces (default: None).

            Returns
            -------
            codomain : nifty.space
                A compatible codomain.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_codomain'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_meta_volume'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array of field values with the pixel volumes (not
            the meta volumes) to a given power.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be weighted.
            power : float, *optional*
                Power of the pixel volumes to be used (default: 1).

            Returns
            -------
            y : numpy.ndarray
                Weighted array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight
        return x*self.vol**power

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_dot(self,x,y):
        """
            Computes the discrete inner product of two given arrays of field
            values.

            Parameters
            ----------
            x : numpy.ndarray
                First array
            y : numpy.ndarray
                Second array

            Returns
            -------
            dot : float
                Inner product of the two arrays.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        y = self.enforce_shape(np.array(y,dtype=self.datatype))
        ## inner product
        return np.dot(np.conjugate(x),y,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations performed in specific transformations.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel.

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations (default: 0).
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'calc_smooth'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Computes the power of an array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values of which the power is to be
                calculated.

            Returns
            -------
            spec : numpy.ndarray
                Power contained in the input array.

            Other parameters
            ----------------
            pindex : numpy.ndarray, *optional*
                Indexing array assigning the input array components to
                components of the power spectrum (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            rho : numpy.ndarray, *optional*
                Number of degrees of freedom per band (default: None).
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'calc_power'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            power : bool, *optional*
                Whether to plot the power contained in the field or the field
                values themselves (default: False).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            mono : bool, *optional*
                Whether to plot the monopole or not (default: True).
            error : {float, numpy.ndarray, nifty.field}, *optional*
                Object indicating some confidence interval to be plotted
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            iter : int, *optional*
                Number of iterations (default: 0).
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'get_plot'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.space>"

    def __str__(self):
        return "nifty.space instance\n- para     = "+str(self.para)+"\n- datatype = numpy."+str(np.result_type(self.datatype))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __len__(self):
        return int(self.dim(split=False))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _meta_vars(self): ## > captures all nonstandard properties
        mars = np.array([ii[1] for ii in vars(self).iteritems() if ii[0] not in ["para","datatype","discrete","vol"]],dtype=np.object)
        if(np.size(mars)==0):
            return None
        else:
            return mars

    def __eq__(self,x): ## __eq__ : self == x
        if(isinstance(x,space)):
            if(isinstance(x,type(self)))and(np.all(self.para==x.para))and(self.discrete==x.discrete)and(np.all(self.vol==x.vol))and(np.all(self._meta_vars()==x._meta_vars())): ## data types are ignored
                return True
        return False

    def __ne__(self,x): ## __ne__ : self <> x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.any(self.para!=x.para))or(self.discrete!=x.discrete)or(np.any(self.vol!=x.vol))or(np.any(self._meta_vars()!=x._meta_vars())): ## data types are ignored
                return True
        return False

    def __lt__(self,x): ## __lt__ : self < x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in range(np.size(self.para)):
                    if(self.para[ii]<x.para[ii]):
                        return True
                    elif(self.para[ii]>x.para[ii]):
                        return False
                for ii in range(np.size(self.vol)):
                    if(self.vol[ii]<x.vol[ii]):
                        return True
                    elif(self.vol[ii]>x.vol[ii]):
                        return False
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in range(np.size(s_mars)):
                    if(np.all(s_mars[ii]<x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]>x_mars[ii])):
                        break
        return False

    def __le__(self,x): ## __le__ : self <= x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in range(np.size(self.para)):
                    if(self.para[ii]<x.para[ii]):
                        return True
                    if(self.para[ii]>x.para[ii]):
                        return False
                for ii in range(np.size(self.vol)):
                    if(self.vol[ii]<x.vol[ii]):
                        return True
                    if(self.vol[ii]>x.vol[ii]):
                        return False
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in range(np.size(s_mars)):
                    if(np.all(s_mars[ii]<x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]>x_mars[ii])):
                        return False
                return True
        return False

    def __gt__(self,x): ## __gt__ : self > x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in range(np.size(self.para)):
                    if(self.para[ii]>x.para[ii]):
                        return True
                    elif(self.para[ii]<x.para[ii]):
                        break
                for ii in range(np.size(self.vol)):
                    if(self.vol[ii]>x.vol[ii]):
                        return True
                    elif(self.vol[ii]<x.vol[ii]):
                        break
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in range(np.size(s_mars)):
                    if(np.all(s_mars[ii]>x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]<x_mars[ii])):
                        break
        return False

    def __ge__(self,x): ## __ge__ : self >= x
        if(isinstance(x,space)):
            if(not isinstance(x,type(self)))or(np.size(self.para)!=np.size(x.para))or(np.size(self.vol)!=np.size(x.vol)):
                raise ValueError(about._errors.cstring("ERROR: incomparable spaces."))
            elif(self.discrete==x.discrete): ## data types are ignored
                for ii in range(np.size(self.para)):
                    if(self.para[ii]>x.para[ii]):
                        return True
                    if(self.para[ii]<x.para[ii]):
                        return False
                for ii in range(np.size(self.vol)):
                    if(self.vol[ii]>x.vol[ii]):
                        return True
                    if(self.vol[ii]<x.vol[ii]):
                        return False
                s_mars = self._meta_vars()
                x_mars = x._meta_vars()
                for ii in range(np.size(s_mars)):
                    if(np.all(s_mars[ii]>x_mars[ii])):
                        return True
                    elif(np.any(s_mars[ii]<x_mars[ii])):
                        return False
                return True
        return False

##=============================================================================



##-----------------------------------------------------------------------------

class point_space(space):
    """
        ..                            __             __
        ..                          /__/           /  /_
        ..      ______    ______    __   __ ___   /   _/
        ..    /   _   | /   _   | /  / /   _   | /  /
        ..   /  /_/  / /  /_/  / /  / /  / /  / /  /_
        ..  /   ____/  \______/ /__/ /__/ /__/  \___/  space class
        .. /__/

        NIFTY subclass for unstructured spaces.

        Unstructured spaces are lists of values without any geometrical
        information.

        Parameters
        ----------
        num : int
            Number of points.
        datatype : numpy.dtype, *optional*
            Data type of the field values (default: None).

        Attributes
        ----------
        para : numpy.ndarray
            Array containing the number of points.
        datatype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that a :py:class:`point_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`point_space`, which is always 1.
    """
    def __init__(self,num,datatype=None):
        """
            Sets the attributes for a point_space class instance.

            Parameters
            ----------
            num : int
                Number of points.
            datatype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None.
        """
        ## check parameter
        if(num<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        self.para = np.array([num],dtype=np.int)

        ## check datatype
        if(datatype is None):
            datatype = np.float64
        elif(datatype not in [np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.complex64,np.complex128]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.float64
        self.datatype = datatype

        self.discrete = True
        self.vol = np.real(np.array([1],dtype=self.datatype))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def num(self):
        """
            Returns the number of points.

            Returns
            -------
            num : int
                Number of points.
        """
        return self.para[0]

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of points.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space.
        """
        ## dim = num
        if(split):
            return np.array([self.para[0]],dtype=np.int)
        else:
            return self.para[0]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space, i.e./  the
            number of points for real-valued fields and twice that number for
            complex-valued fields.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        ## dof ~ dim
        if(self.datatype in [np.complex64,np.complex128]):
            return 2*self.para[0]
        else:
            return self.para[0]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined for (unstructured) point spaces."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined for (unstructured) point spaces."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, in this case another instance of
            :py:class:`point_space` with the same properties.

            Returns
            -------
            codomain : nifty.point_space
                A compatible codomain.
        """
        return point_space(self.para[0],datatype=self.datatype) ## == self

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            Since point spaces are unstructured, the meta volume of each
            component is one, the total meta volume of the space is the number
            of points.
        """
        if(total):
            return self.dim(split=False)
        else:
            return np.ones(self.dim(split=True),dtype=self.vol.dtype,order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,**kwargs):
        """
            Raises an error since smoothing is ill-defined on an unstructured
            space.
        """
        raise AttributeError(about._errors.cstring("ERROR: smoothing ill-defined for (unstructured) point space."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Raises an error since the power spectrum is ill-defined for point
            spaces.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined for (unstructured) point space."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,unit="",norm=None,other=None,legend=False,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
        """
        if(not pl.isinteractive()):
            about.warnings.cprint("WARNING: interactive mode off.")

        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

        xaxes = np.arange(self.para[0],dtype=np.int)
        if(vmin is None):
            if(np.iscomplexobj(x)):
                vmin = min(np.min(np.absolute(x),axis=None,out=None),np.min(np.real(x),axis=None,out=None),np.min(np.imag(x),axis=None,out=None))
            else:
                vmin = np.min(x,axis=None,out=None)
        if(vmax is None):
            if(np.iscomplexobj(x)):
                vmax = max(np.max(np.absolute(x),axis=None,out=None),np.max(np.real(x),axis=None,out=None),np.max(np.imag(x),axis=None,out=None))
            else:
                vmax = np.max(x,axis=None,out=None)

        if(norm=="log")and(vmin<=0):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        if(np.iscomplexobj(x)):
            ax0.scatter(xaxes,np.absolute(x),s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph (absolute)",linewidths=None,verts=None,zorder=1)
            ax0.scatter(xaxes,np.real(x),s=20,color=[0.0,0.5,0.0],marker='s',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph (real part)",linewidths=None,verts=None,facecolor="none",zorder=1)
            ax0.scatter(xaxes,np.imag(x),s=20,color=[0.0,0.5,0.0],marker='D',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph (imaginary part)",linewidths=None,verts=None,facecolor="none",zorder=1)
            if(legend):
                ax0.legend()
        elif(other is not None):
            ax0.scatter(xaxes,x,s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph 0",linewidths=None,verts=None,zorder=1)
            if(isinstance(other,tuple)):
                other = [self.enforce_values(xx,extend=True) for xx in other]
            else:
                other = [self.enforce_values(other,extend=True)]
            imax = max(1,len(other)-1)
            for ii in range(len(other)):
                ax0.scatter(xaxes,other[ii],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph "+str(ii),linewidths=None,verts=None,zorder=-ii)
            if(legend):
                ax0.legend()
        else:
            ax0.scatter(xaxes,x,s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,label="graph 0",linewidths=None,verts=None,zorder=1)

        ax0.set_xlim(xaxes[0],xaxes[-1])
        ax0.set_xlabel("index")
        ax0.set_ylim(vmin,vmax)
        if(norm=="log"):
            ax0.set_yscale('log')

        if(unit):
            unit = " ["+unit+"]"
        ax0.set_ylabel("values"+unit)
        ax0.set_title(title)


    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.point_space>"

    def __str__(self):
        return "nifty.point_space instance\n- num      = "+str(self.para[0])+"\n- datatype = numpy."+str(np.result_type(self.datatype))

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class rg_space(space):
    """
        ..      _____   _______
        ..    /   __/ /   _   /
        ..   /  /    /  /_/  /
        ..  /__/     \____  /  space class
        ..           /_____/

        NIFTY subclass for spaces of regular Cartesian grids.

        Parameters
        ----------
        num : {int, numpy.ndarray}
            Number of gridpoints or numbers of gridpoints along each axis.
        naxes : int, *optional*
            Number of axes (default: None).
        zerocenter : {bool, numpy.ndarray}, *optional*
            Whether the Fourier zero-mode is located in the center of the grid
            (or the center of each axis speparately) or not (default: True).
        hermitian : bool, *optional*
            Whether the fields living in the space follow hermitian symmetry or
            not (default: True).
        purelyreal : bool, *optional*
            Whether the field values are purely real (default: True).
        dist : {float, numpy.ndarray}, *optional*
            Distance between two grid points along each axis (default: None).
        fourier : bool, *optional*
            Whether the space represents a Fourier or a position grid
            (default: False).

        Notes
        -----
        Only even numbers of grid points per axis are supported.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing information on the axes of the
            space in the following form: The first entries give the grid-points
            along each axis in reverse order; the next entry is 0 if the
            fields defined on the space are purely real-valued, 1 if they are
            hermitian and complex, and 2 if they are not hermitian, but
            complex-valued; the last entries hold the information on whether
            the axes are centered on zero or not, containing a one for each
            zero-centered axis and a zero for each other one, in reverse order.
        datatype : numpy.dtype
            Data type of the field values for a field defined on this space,
            either ``numpy.float64`` or ``numpy.complex128``.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for regular grids.
        vol : numpy.ndarray
            One-dimensional array containing the distances between two grid
            points along each axis, in reverse order. By default, the total
            length of each axis is assumed to be one.
        fourier : bool
            Whether or not the grid represents a Fourier basis.
    """
    epsilon = 0.0001 ## relative precision for comparisons

    def __init__(self,num,naxes=None,zerocenter=True,hermitian=True,purelyreal=True,dist=None,fourier=False):
        """
            Sets the attributes for an rg_space class instance.

            Parameters
            ----------
            num : {int, numpy.ndarray}
                Number of gridpoints or numbers of gridpoints along each axis.
            naxes : int, *optional*
                Number of axes (default: None).
            zerocenter : {bool, numpy.ndarray}, *optional*
                Whether the Fourier zero-mode is located in the center of the
                grid (or the center of each axis speparately) or not
                (default: True).
            hermitian : bool, *optional*
                Whether the fields living in the space follow hermitian
                symmetry or not (default: True).
            purelyreal : bool, *optional*
                Whether the field values are purely real (default: True).
            dist : {float, numpy.ndarray}, *optional*
                Distance between two grid points along each axis
                (default: None).
            fourier : bool, *optional*
                Whether the space represents a Fourier or a position grid
                (default: False).

            Returns
            -------
            None
        """
        ## check parameters
        para = np.array([],dtype=np.int)
        if(np.isscalar(num)):
            num = np.array([num],dtype=np.int)
        else:
            num = np.array(num,dtype=np.int)
        if(np.any(num%2)): ## module restriction
            raise ValueError(about._errors.cstring("ERROR: unsupported odd number of grid points."))
        if(naxes is None):
            naxes = np.size(num)
        elif(np.size(num)==1):
            num = num*np.ones(naxes,dtype=np.int,order='C')
        elif(np.size(num)!=naxes):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(num))+" <> "+str(naxes)+" )."))
        para = np.append(para,num[::-1],axis=None)
        para = np.append(para,2-(bool(hermitian) or bool(purelyreal))-bool(purelyreal),axis=None) ## {0,1,2}
        if(np.isscalar(zerocenter)):
            zerocenter = bool(zerocenter)*np.ones(naxes,dtype=np.int,order='C')
        else:
            zerocenter = np.array(zerocenter,dtype=np.bool)
            if(np.size(zerocenter)==1):
                zerocenter = zerocenter*np.ones(naxes,dtype=np.int,order='C')
            elif(np.size(zerocenter)!=naxes):
                raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(zerocenter))+" <> "+str(naxes)+" )."))
        para = np.append(para,zerocenter[::-1]*-1,axis=None) ## -1 XOR 0 (centered XOR not)

        self.para = para

        ## set data type
        if(not self.para[naxes]):
            self.datatype = np.float64
        else:
            self.datatype = np.complex128

        self.discrete = False

        ## set volume
        if(dist is None):
            dist = 1/num.astype(self.datatype)
        elif(np.isscalar(dist)):
            dist = self.datatype(dist)*np.ones(naxes,dtype=self.datatype,order='C')
        else:
            dist = np.array(dist,dtype=self.datatype)
            if(np.size(dist)==1):
                dist = dist*np.ones(naxes,dtype=self.datatype,order='C')
            if(np.size(dist)!=naxes):
                raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(dist))+" <> "+str(naxes)+" )."))
        if(np.any(dist<=0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive distance(s)."))
        self.vol = np.real(dist)[::-1]

        self.fourier = bool(fourier)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def naxes(self):
        """
            Returns the number of axes of the grid.

            Returns
            -------
            naxes : int
                Number of axes of the regular grid.
        """
        return (np.size(self.para)-1)//2

    def zerocenter(self):
        """
            Returns information on the centering of the axes.

            Returns
            -------
            zerocenter : numpy.ndarray
                Whether the grid is centered on zero for each axis or not.
        """
        return self.para[-(np.size(self.para)-1)//2:][::-1].astype(np.bool)

    def dist(self):
        """
            Returns the distances between grid points along each axis.

            Returns
            -------
            dist : np.ndarray
                Distances between two grid points on each axis.
        """
        return self.vol[::-1]

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension split up, i.e. the numbers of
                pixels along each axis, or their product (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space. If ``split==True``, a
                one-dimensional array with an entry for each axis is returned.
        """
        ## dim = product(n)
        if(split):
            return self.para[:(np.size(self.para)-1)//2]
        else:
            return np.prod(self.para[:(np.size(self.para)-1)//2],axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space, i.e.\  the
            number of grid points multiplied with one or two, depending on
            complex-valuedness and hermitian symmetry of the fields.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        ## dof ~ dim
        if(self.para[(np.size(self.para)-1)//2]<2):
            return np.prod(self.para[:(np.size(self.para)-1)//2],axis=0,dtype=None,out=None)
        else:
            return 2*np.prod(self.para[:(np.size(self.para)-1)//2],axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,size=None,**kwargs):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {float, numpy.ndarray, nifty.field}
                Fiducial power spectrum from which a valid power spectrum is to
                be calculated. Scalars are interpreted as constant power
                spectra.

            Returns
            -------
            spec : numpy.ndarray
                Valid power spectrum.

            Other parameters
            ----------------
            size : int, *optional*
                Number of bands the power spectrum shall have (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band.

        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(self.datatype)
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)
        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        if(size is None):
            if(kwargs.has_key("kindex")):
                size = len(kwargs.get("kindex"))
            else:
                size = np.size(gp.get_power_index(self.para[:(np.size(self.para)-1)//2],self.vol,self.para[-(np.size(self.para)-1)//2:].astype(np.bool),irred=True,fourier=self.fourier)[0]) ## nontrivial
        ## drop imaginary part
        spec = np.real(spec)

        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Provides the indexing array of the power spectrum.

            Provides either an array giving for each component of a field the
            corresponding index of a power spectrum (if ``irreducible==False``)
            or two arrays containing the scales of the modes and the numbers of
            modes with this scale (if ``irreducible==True``).

            Parameters
            ----------
            irreducible : bool, *optional*
                Whether to return two arrays containing the scales and
                corresponding number of represented modes (if True) or the
                indexing array (if False) (default: False).

            Returns
            -------
            kindex : numpy.ndarray
                Scale of each band, returned only if ``irreducible==True``.
            rho : numpy.ndarray
                Number of modes per scale represented in the discretization,
                returned only if ``irreducible==True``.
            pindex : numpy.ndarray
                Indexing array giving the power spectrum index for each
                represented mode, returned only if ``irreducible==False``.

            Notes
            -----
            The indexing array is of the same shape as a field living in this
            space and contains the indices of the associated bands.
            kindex and rho are each one-dimensional arrays.
        """
        if(self.fourier):
            return gp.get_power_index(self.para[:(np.size(self.para)-1)//2],self.vol,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),irred=irreducible,fourier=self.fourier) ## nontrivial
        else:
            raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, taking care of
            data types, shape, and symmetry.

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray
                Array containing the valid field values.

            Other parameters
            ----------------
            extend : bool, *optional*
                Whether a scalar is extented to a constant array or not
                (default: True).
        """
        if(isinstance(x,field)):
            if(self==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    x = x.val
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            if(np.size(x)==1):
                if(extend):
                    x = self.datatype(x)*np.ones(self.dim(split=True),dtype=self.datatype,order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x],dtype=self.datatype)
                    else:
                        x = np.array(x,dtype=self.datatype)
            else:
                x = self.enforce_shape(np.array(x,dtype=self.datatype))

        ## hermitianize if ...
        if(about.hermitianize.status)and(np.size(x)!=1)and(self.para[(np.size(self.para)-1)//2]==1):
            x = gp.nhermitianize_fast(x,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),special=False)

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters, taking into account possible complex-valuedness
            and hermitian symmetry.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {float, numpy.ndarray}, *optional*
                Power spectrum (default: 1).
            pindex : numpy.ndarray, *optional*
                Indexing array giving the power spectrum index of each band
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale of each band (default: None).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            return np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            if(about.hermitianize.status)and(self.para[(np.size(self.para)-1)//2]==1):
                return gp.random_hermitian_pm1(self.datatype,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),self.dim(split=True)) ## special case
            else:
                x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            naxes = (np.size(self.para)-1)//2
            x = gp.draw_vector_nd(self.para[:naxes],self.vol,arg[1],symtype=self.para[naxes],fourier=self.fourier,zerocentered=self.para[-naxes:].astype(np.bool),kpack=arg[2])
            ## correct for 'ifft'
            if(not self.fourier):
                x = self.calc_weight(x,power=-1)
            return x

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        ## hermitianize if ...
        if(about.hermitianize.status)and(self.para[(np.size(self.para)-1)//2]==1):
            x = gp.nhermitianize_fast(x,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),special=(arg[0] in ["gau","pm1"]))

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the space or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        elif(isinstance(codomain,rg_space)):
            ##                       naxes==naxes
            if((np.size(codomain.para)-1)//2==(np.size(self.para)-1)//2):
                naxes = (np.size(self.para)-1)//2
                ##                            num'==num
                if(np.all(codomain.para[:naxes]==self.para[:naxes])):
                    ##                 typ'==typ             ==2
                    if(codomain.para[naxes]==self.para[naxes]==2):
                        ##                                         dist'~=1/(num*dist)
                        if(np.all(np.absolute(self.para[:naxes]*self.vol*codomain.vol-1)<self.epsilon)):
                            return True
                        ##           fourier'==fourier
                        elif(codomain.fourier==self.fourier):
                            ##                           dist'~=dist
                            if(np.all(np.absolute(codomain.vol/self.vol-1)<self.epsilon)):
                                return True
                            else:
                                about.warnings.cprint("WARNING: unrecommended codomain.")
                    ##   2!=                typ'!=typ             !=2                                             dist'~=1/(num*dist)
                    elif(2!=codomain.para[naxes]!=self.para[naxes]!=2)and(np.all(np.absolute(self.para[:naxes]*self.vol*codomain.vol-1)<self.epsilon)):
                        return True
                    ##                   typ'==typ             !=2
                    elif(codomain.para[naxes]==self.para[naxes]!=2)and(codomain.fourier==self.fourier):
                        ##                           dist'~=dist
                        if(np.all(np.absolute(codomain.vol/self.vol-1)<self.epsilon)):
                            return True
                        else:
                            about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,coname=None,cozerocenter=None,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  either a shifted grid or a Fourier conjugate
            grid.

            Parameters
            ----------
            coname : string, *optional*
                String specifying a desired codomain (default: None).
            cozerocenter : {bool, numpy.ndarray}, *optional*
                Whether or not the grid is zerocentered for each axis or not
                (default: None).

            Returns
            -------
            codomain : nifty.rg_space
                A compatible codomain.

            Notes
            -----
            Possible arguments for `coname` are ``'f'`` in which case the
            codomain arises from a Fourier transformation, ``'i'`` in which case
            it arises from an inverse Fourier transformation, and ``'?'`` in
            which case it arises from a simple shift. If no `coname` is given,
            the Fourier conjugate grid is produced.
        """
        naxes = (np.size(self.para)-1)//2
        if(cozerocenter is None):
            cozerocenter = self.para[-naxes:][::-1]
        elif(np.isscalar(cozerocenter)):
            cozerocenter = bool(cozerocenter)
        else:
            cozerocenter = np.array(cozerocenter,dtype=np.bool)
            if(np.size(cozerocenter)==1):
                cozerocenter = np.asscalar(cozerocenter)
            elif(np.size(cozerocenter)!=naxes):
                raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(cozerocenter))+" <> "+str(naxes)+" )."))

        if(coname is None):
            return rg_space(self.para[:naxes][::-1],naxes=naxes,zerocenter=cozerocenter,hermitian=bool(self.para[naxes]<2),purelyreal=bool(self.para[naxes]==1),dist=1/(self.para[:naxes]*self.vol)[::-1],fourier=bool(not self.fourier)) ## dist',fourier' = 1/(num*dist),NOT fourier

        elif(coname[0]=='f'):
            return rg_space(self.para[:naxes][::-1],naxes=naxes,zerocenter=cozerocenter,hermitian=bool(self.para[naxes]<2),purelyreal=bool(self.para[naxes]==1),dist=1/(self.para[:naxes]*self.vol)[::-1],fourier=True) ## dist',fourier' = 1/(num*dist),True

        elif(coname[0]=='i'):
            return rg_space(self.para[:naxes][::-1],naxes=naxes,zerocenter=cozerocenter,hermitian=bool(self.para[naxes]<2),purelyreal=bool(self.para[naxes]==1),dist=1/(self.para[:naxes]*self.vol)[::-1],fourier=False) ## dist',fourier' = 1/(num*dist),False

        else:
            return rg_space(self.para[:naxes][::-1],naxes=naxes,zerocenter=cozerocenter,hermitian=bool(self.para[naxes]<2),purelyreal=bool(not self.para[naxes]),dist=self.vol[::-1],fourier=self.fourier) ## dist',fourier' = dist,fourier

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions. In the case of an :py:class:`rg_space`, the
            meta volumes are simply the pixel volumes.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each pixel (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the pixels or the complete space.
        """
        if(total):
            return self.dim(split=False)*np.prod(self.vol,axis=0,dtype=None,out=None)
        else:
            mol = np.ones(self.dim(split=True),dtype=self.vol.dtype,order='C')
            return self.calc_weight(mol,power=1)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array with the pixel volumes to a given power.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be weighted.
            power : float, *optional*
                Power of the pixel volumes to be used (default: 1).

            Returns
            -------
            y : numpy.ndarray
                Weighted array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight
        return x*np.prod(self.vol,axis=0,dtype=None,out=None)**power

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_dot(self,x,y):
        """
            Computes the discrete inner product of two given arrays.

            Parameters
            ----------
            x : numpy.ndarray
                First array
            y : numpy.ndarray
                Second array

            Returns
            -------
            dot : float
                Inner product of the two arrays.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        y = self.enforce_shape(np.array(y,dtype=self.datatype))
        ## inner product
        return np.dot(np.conjugate(x.flatten(order='C')),y.flatten(order='C'),out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.rg_space, *optional*
                Target space to which the transformation shall map
                (default: None).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## mandatory(!) codomain check
        if(isinstance(codomain,rg_space))and(self.check_codomain(codomain)):
            naxes = (np.size(self.para)-1)//2
            ## select machine
            if(np.all(np.absolute(self.para[:naxes]*self.vol*codomain.vol-1)<self.epsilon)):
                if(codomain.fourier):
                    ftmachine = "fft"
                    ## correct for 'fft'
                    x = self.calc_weight(x,power=1)
                else:
                    ftmachine = "ifft"
                    ## correct for 'ifft'
                    x = self.calc_weight(x,power=1)
                    x *= self.dim(split=False)
            else:
                ftmachine = "none"
            ## transform
            if(self.datatype==np.float64):
                Tx = gf.gfft(x.astype(np.complex128),in_ax=[],out_ax=[],ftmachine=ftmachine,in_zero_center=self.para[-naxes:].astype(np.bool).tolist(),out_zero_center=codomain.para[-naxes:].astype(np.bool).tolist(),enforce_hermitian_symmetry=bool(codomain.para[naxes]==1),W=-1,alpha=-1,verbose=False)
            else:
                Tx = gf.gfft(x,in_ax=[],out_ax=[],ftmachine=ftmachine,in_zero_center=self.para[-naxes:].astype(np.bool).tolist(),out_zero_center=codomain.para[-naxes:].astype(np.bool).tolist(),enforce_hermitian_symmetry=bool(codomain.para[naxes]==1),W=-1,alpha=-1,verbose=False)
            ## check complexity
            if(not codomain.para[naxes]): ## purely real
                ## check imaginary part
                if(np.any(Tx.imag!=0))and(np.dot(Tx.imag.flatten(order='C'),Tx.imag.flatten(order='C'),out=None)>self.epsilon**2*np.dot(Tx.real.flatten(order='C'),Tx.real.flatten(order='C'),out=None)):
                    about.warnings.cprint("WARNING: discarding considerable imaginary part.")
                Tx = np.real(Tx)

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel.

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        elif(sigma==0):
            return x
        else:
            ## smooth
            naxes = (np.size(self.para)-1)//2
            Gx = gs.smooth_field(x,self.fourier,self.para[-naxes:].astype(np.bool).tolist(),bool(self.para[naxes]==1),self.vol,smooth_length=sigma)
            ## check complexity
            if(not self.para[naxes]): ## purely real
                ## check imaginary part
                if(np.any(Gx.imag!=0))and(np.dot(Gx.imag.flatten(order='C'),Gx.imag.flatten(order='C'),out=None)>self.epsilon**2*np.dot(Gx.real.flatten(order='C'),Gx.real.flatten(order='C'),out=None)):
                    about.warnings.cprint("WARNING: discarding considerable imaginary part.")
                Gx = np.real(Gx)
            return Gx

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Computes the power of an array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values of which the power is to be
                calculated.

            Returns
            -------
            spec : numpy.ndarray
                Power contained in the input array.

            Other parameters
            ----------------
            pindex : numpy.ndarray, *optional*
                Indexing array assigning the input array components to
                components of the power spectrum (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            rho : numpy.ndarray, *optional*
                Number of degrees of freedom per band (default: None).
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## correct for 'fft'
        if(not self.fourier):
            x = self.calc_weight(x,power=1)
        ## power spectrum
        return gp.calc_ps_fast(x,self.para[:(np.size(self.para)-1)//2],self.vol,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),fourier=self.fourier,**kwargs)
#        if(kwargs.has_key("pindex"))and(kwargs.has_key("rho")):
#            return gp.calc_ps_fast(x,self.para[:(np.size(self.para)-1)//2],self.vol,[kwargs.get("pindex"),kwargs.get("rho")],self.para[-((np.size(self.para)-1)//2):].astype(np.bool),fourier=self.fourier)
#        else:
#            return gp.calc_ps(x,self.para[:(np.size(self.para)-1)//2],self.vol,self.para[-((np.size(self.para)-1)//2):].astype(np.bool),fourier=self.fourier)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=None,unit="",norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            power : bool, *optional*
                Whether to plot the power contained in the field or the field
                values themselves (default: False).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            mono : bool, *optional*
                Whether to plot the monopole or not (default: True).
            error : {float, numpy.ndarray, nifty.field}, *optional*
                Object indicating some confidence interval to be plotted
                (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).

        """
        if(not pl.isinteractive()):
            about.warnings.cprint("WARNING: interactive mode off.")

        naxes = (np.size(self.para)-1)//2
        if(power is None):
            power = bool(self.para[naxes])

        if(power):
            x = self.calc_power(x,**kwargs)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            if(kwargs.has_key("kindex")):
                xaxes = kwargs.get("kindex")
            else:
                xaxes = gp.get_power_index(self.para[:naxes],self.vol,self.para[-naxes:].astype(np.bool),irred=True,fourier=self.fourier)[0] ## nontrivial
            if(norm is None)or(not isinstance(norm,int)):
                norm = naxes
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes**norm*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes**norm*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes**norm*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in range(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii],size=np.size(xaxes))
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other,size=np.size(xaxes))]
                imax = max(1,len(other)-1)
                for ii in range(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes**norm*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$|k|$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$|k|^{%i} P_k$"%norm)
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x))

            if(naxes==1):
                fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
                ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

                xaxes = (np.arange(self.para[0],dtype=np.int)+self.para[2]*(self.para[0]//2))*self.vol
                if(vmin is None):
                    if(np.iscomplexobj(x)):
                        vmin = min(np.min(np.absolute(x),axis=None,out=None),np.min(np.real(x),axis=None,out=None),np.min(np.imag(x),axis=None,out=None))
                    else:
                        vmin = np.min(x,axis=None,out=None)
                if(vmax is None):
                    if(np.iscomplexobj(x)):
                        vmax = max(np.max(np.absolute(x),axis=None,out=None),np.max(np.real(x),axis=None,out=None),np.max(np.imag(x),axis=None,out=None))
                    else:
                        vmax = np.max(x,axis=None,out=None)
                if(norm=="log"):
                    ax0graph = ax0.semilogy
                    if(vmin<=0):
                        raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
                else:
                    ax0graph = ax0.plot

                if(np.iscomplexobj(x)):
                    ax0graph(xaxes,np.absolute(x),color=[0.0,0.5,0.0],label="graph (absolute)",linestyle='-',linewidth=2.0,zorder=1)
                    ax0graph(xaxes,np.real(x),color=[0.0,0.5,0.0],label="graph (real part)",linestyle="--",linewidth=1.0,zorder=0)
                    ax0graph(xaxes,np.imag(x),color=[0.0,0.5,0.0],label="graph (imaginary part)",linestyle=':',linewidth=1.0,zorder=0)
                    if(legend):
                        ax0.legend()
                elif(other is not None):
                    ax0graph(xaxes,x,color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
                    if(isinstance(other,tuple)):
                        other = [self.enforce_values(xx,extend=True) for xx in other]
                    else:
                        other = [self.enforce_values(other,extend=True)]
                    imax = max(1,len(other)-1)
                    for ii in range(len(other)):
                        ax0graph(xaxes,other[ii],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(kwargs.has_key("error")):
                        error = self.enforce_values(np.absolute(kwargs.get("error")),extend=True)
                        ax0.fill_between(xaxes,x-error,x+error,color=[0.8,0.8,0.8],label="error 0",zorder=-len(other))
                    if(legend):
                        ax0.legend()
                else:
                    ax0graph(xaxes,x,color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
                    if(kwargs.has_key("error")):
                        error = self.enforce_values(np.absolute(kwargs.get("error")),extend=True)
                        ax0.fill_between(xaxes,x-error,x+error,color=[0.8,0.8,0.8],label="error 0",zorder=0)

                ax0.set_xlim(xaxes[0],xaxes[-1])
                ax0.set_xlabel("coordinate")
                ax0.set_ylim(vmin,vmax)
                if(unit):
                    unit = " ["+unit+"]"
                ax0.set_ylabel("values"+unit)
                ax0.set_title(title)

            elif(naxes==2):
                if(np.iscomplexobj(x)):
                    about.infos.cprint("INFO: absolute values, real and imaginary part plotted.")
                    if(title):
                        title += " "
                    self.get_plot(np.absolute(x),title=title+"(absolute)",vmin=vmin,vmax=vmax,power=False,unit=unit,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                    self.get_plot(np.real(x),title=title+"(real part)",vmin=vmin,vmax=vmax,power=False,unit=unit,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                    self.get_plot(np.imag(x),title=title+"(imaginary part)",vmin=vmin,vmax=vmax,power=False,unit=unit,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                else:
                    if(vmin is None):
                        vmin = np.min(x,axis=None,out=None)
                    if(vmax is None):
                        vmax = np.max(x,axis=None,out=None)
                    if(norm=="log")and(vmin<=0):
                        raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

                    s_ = np.array([self.para[1]*self.vol[1]/np.max(self.para[:naxes]*self.vol,axis=None,out=None),self.para[0]*self.vol[0]/np.max(self.para[:naxes]*self.vol,axis=None,out=None)*(1.0+0.159*bool(cbar))])
                    fig = pl.figure(num=None,figsize=(6.4*s_[0],6.4*s_[1]),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
                    ax0 = fig.add_axes([0.06/s_[0],0.06/s_[1],1.0-0.12/s_[0],1.0-0.12/s_[1]])

                    xaxes = (np.arange(self.para[1]+1,dtype=np.int)-0.5+self.para[4]*(self.para[1]//2))*self.vol[1]
                    yaxes = (np.arange(self.para[0]+1,dtype=np.int)-0.5+self.para[3]*(self.para[0]//2))*self.vol[0]
                    if(norm=="log"):
                        n_ = ln(vmin=vmin,vmax=vmax)
                    else:
                        n_ = None
                    sub = ax0.pcolormesh(xaxes,yaxes,x,cmap=cmap,norm=n_,vmin=vmin,vmax=vmax)
                    ax0.set_xlim(xaxes[0],xaxes[-1])
                    ax0.set_xticks([0],minor=False)
                    ax0.set_ylim(yaxes[0],yaxes[-1])
                    ax0.set_yticks([0],minor=False)
                    ax0.set_aspect("equal")
                    if(cbar):
                        if(norm=="log"):
                            f_ = lf(10,labelOnlyBase=False)
                            b_ = sub.norm.inverse(np.linspace(0,1,sub.cmap.N+1))
                            v_ = np.linspace(sub.norm.vmin,sub.norm.vmax,sub.cmap.N)
                        else:
                            f_ = None
                            b_ = None
                            v_ = None
                        cb0 = fig.colorbar(sub,ax=ax0,orientation="horizontal",fraction=0.1,pad=0.05,shrink=0.75,aspect=20,ticks=[vmin,vmax],format=f_,drawedges=False,boundaries=b_,values=v_)
                        cb0.ax.text(0.5,-1.0,unit,fontdict=None,withdash=False,transform=cb0.ax.transAxes,horizontalalignment="center",verticalalignment="center")
                    ax0.set_title(title)

            else:
                raise ValueError(about._errors.cstring("ERROR: unsupported number of axes ( "+str(naxes)+" > 2 )."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.rg_space>"

    def __str__(self):
        naxes = (np.size(self.para)-1)//2
        num = self.para[:naxes][::-1].tolist()
        zerocenter = self.para[-naxes:][::-1].astype(np.bool).tolist()
        dist = self.vol[::-1].tolist()
        return "nifty.rg_space instance\n- num        = "+str(num)+"\n- naxes      = "+str(naxes)+"\n- hermitian  = "+str(bool(self.para[naxes]<2))+"\n- purelyreal = "+str(bool(not self.para[naxes]))+"\n- zerocenter = "+str(zerocenter)+"\n- dist       = "+str(dist)+"\n- fourier    = "+str(self.fourier)

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class lm_space(space):
    """
        ..       __
        ..     /  /
        ..    /  /    __ ____ ___
        ..   /  /   /   _    _   |
        ..  /  /_  /  / /  / /  /
        ..  \___/ /__/ /__/ /__/  space class

        NIFTY subclass for spherical harmonics components, for representations
        of fields on the two-sphere.

        Parameters
        ----------
        lmax : int
            Maximum :math:`\ell`-value up to which the spherical harmonics
            coefficients are to be used.
        mmax : int, *optional*
            Maximum :math:`m`-value up to which the spherical harmonics
            coefficients are to be used (default: `lmax`).
        datatype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.complex128).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.

        Notes
        -----
        Hermitian symmetry, i.e. :math:`a_{\ell -m} = \overline{a}_{\ell m}` is
        always assumed for the spherical harmonics components, i.e. only fields
        on the two-sphere with real-valued representations in position space
        can be handled.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `lmax` and
            `mmax`.
        datatype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Parameter captioning the fact that an :py:class:`lm_space` is
            always discrete.
        vol : numpy.ndarray
            Pixel volume of the :py:class:`lm_space`, which is always 1.
    """
    def __init__(self,lmax,mmax=None,datatype=None):
        """
            Sets the attributes for an lm_space class instance.

            Parameters
            ----------
            lmax : int
                Maximum :math:`\ell`-value up to which the spherical harmonics
                coefficients are to be used.
            mmax : int, *optional*
                Maximum :math:`m`-value up to which the spherical harmonics
                coefficients are to be used (default: `lmax`).
            datatype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.complex128).

            Returns
            -------
            None.
        """
        ## check parameters
        if(lmax<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        if(lmax%2==0)and(lmax>2): ## exception lmax == 2 (nside == 1)
            about.warnings.cprint("WARNING: unrecommended parameter ( lmax <> 2*n+1 ).")
        if(mmax is None):
            mmax = lmax
        elif(mmax<1)or(mmax>lmax):
            about.warnings.cprint("WARNING: parameter set to default.")
            mmax = lmax
        if(mmax!=lmax):
            about.warnings.cprint("WARNING: unrecommended parameter ( mmax <> lmax ).")
        self.para = np.array([lmax,mmax],dtype=np.int)

        ## check data type
        if(datatype is None):
            datatype = np.complex128
        elif(datatype not in [np.complex64,np.complex128]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.complex128
        self.datatype = datatype

        self.discrete = True
        self.vol = np.real(np.array([1],dtype=self.datatype))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def lmax(self):
        """
            Returns the maximum quantum number :math:`\ell`.

            Returns
            -------
            lmax : int
                Maximum quantum number :math:`\ell`.
        """
        return self.para[0]

    def mmax(self):
        """
            Returns the maximum quantum number :math:`m`.

            Returns
            -------
            mmax : int
                Maximum quantum number :math:`m`.

        """
        return self.para[1]

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of spherical
            harmonics components that are stored.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Number of spherical harmonics components.

            Notes
            -----
            Due to the symmetry assumption, only the components with
            non-negative :math:`m` are stored and only these components are
            counted here.
        """
        ## dim = (mmax+1)*(lmax-mmax/2+1)
        if(split):
            return np.array([(self.para[0]+1)*(self.para[1]+1)-(self.para[1]+1)*self.para[1]//2],dtype=np.int)
        else:
            return (self.para[0]+1)*(self.para[1]+1)-(self.para[1]+1)*self.para[1]//2

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space, taking into
            account symmetry constraints and complex-valuedness.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.

            Notes
            -----
            The number of degrees of freedom is reduced due to the hermitian
            symmetry, which is assumed for the spherical harmonics components.
        """
        ## dof = 2*dim-(lmax+1) = (lmax+1)*(2*mmax+1)*(mmax+1)*mmax
        return (self.para[0]+1)*(2*self.para[1]+1)-(self.para[1]+1)*self.para[1]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {float, numpy.ndarray, nifty.field}
                Fiducial power spectrum from which a valid power spectrum is to
                be calculated. Scalars are interpreted as constant power
                spectra.

            Returns
            -------
            spec : numpy.ndarray
                Valid power spectrum.
        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(self.datatype)
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)
        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        size = self.para[0]+1 ## lmax+1
        ## drop imaginary part
        spec = np.real(spec)

        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Provides the indexing array of the power spectrum.

            Provides either an array giving for each component of a field the
            corresponding index of a power spectrum (if ``irreducible==False``)
            or two arrays containing the scales of the modes and the numbers of
            modes with this scale (if ``irreducible==True``).

            Parameters
            ----------
            irreducible : bool, *optional*
                Whether to return two arrays containing the scales and
                corresponding number of represented modes (if True) or the
                indexing array (if False) (default: False).

            Returns
            -------
            kindex : numpy.ndarray
                Scale of each band, returned only if ``irreducible==True``.
            rho : numpy.ndarray
                Number of modes per scale represented in the discretization,
                returned only if ``irreducible==True``.
            pindex : numpy.ndarray
                Indexing array giving the power spectrum index for each
                represented mode, returned only if ``irreducible==False``.

            Notes
            -----
            The indexing array is of the same shape as a field living in this
            space and contains the indices of the associated bands.
            kindex and rho are each one-dimensional arrays.
        """
        if(irreducible):
            ind = np.sort(list(set(hp.Alm.getlm(self.para[0],i=None)[0])),axis=0,kind="quicksort",order=None)
            return ind,2*ind+1
        else:
            return hp.Alm.getlm(self.para[0],i=None)[0] ## l of (l,m)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, taking into
            account data types, size, and hermitian symmetry.

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray
                Array containing the valid field values.

            Other parameters
            ----------------
            extend : bool, *optional*
                Whether a scalar is extented to a constant array or not
                (default: True).
        """
        if(isinstance(x,field)):
            if(self==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    x = x.val
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            if(np.size(x)==1):
                if(extend):
                    x = self.datatype(x)*np.ones(self.dim(split=True),dtype=self.datatype,order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x],dtype=self.datatype)
                    else:
                        x = np.array(x,dtype=self.datatype)
            else:
                x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(np.size(x)!=1)and(np.any(x.imag[:self.para[0]+1]!=0)):
            about.warnings.cprint("WARNING: forbidden values reset.")
            x.real[:self.para[0]+1] = np.absolute(x[:self.para[0]+1])*(np.sign(x.real[:self.para[0]+1])+(np.sign(x.real[:self.para[0]+1])==0)*np.sign(x.imag[:self.para[0]+1])).astype(np.int)
            x.imag[:self.para[0]+1] = 0 ## x.imag[l,m==0] = 0

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters, taking into account complex-valuedness and
            hermitian symmetry.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {float, numpy.ndarray}, *optional*
                Power spectrum (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            return np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            lmax = self.para[0] ## lmax
            if(self.datatype==np.complex64):
                x = gl.synalm_f(arg[1],lmax=lmax,mmax=lmax)
            else:
                #x = gl.synalm(spec,lmax=lmax,mmax=lmax)
                x = hp.synalm(arg[1],lmax=lmax,mmax=lmax)
            return x

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        if(np.any(x.imag[:self.para[0]+1]!=0)):
            x.real[:self.para[0]+1] = np.absolute(x[:self.para[0]+1])*(np.sign(x.real[:self.para[0]+1])+(np.sign(x.real[:self.para[0]+1])==0)*np.sign(x.imag[:self.para[0]+1])).astype(np.int)
            x.imag[:self.para[0]+1] = 0 ## x.imag[l,m==0] = 0

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the
            :py:class:`lm_space` or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.

            Notes
            -----
            Compatible codomains are instances of :py:class:`lm_space`,
            :py:class:`gl_space`, and :py:class:`hp_space`.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        elif(isinstance(codomain,gl_space)):
            ##         lmax==mmax                         nlat==lmax+1                         nlon==2*lmax+1
            if(self.para[0]==self.para[1])and(codomain.para[0]==self.para[0]+1)and(codomain.para[1]==2*self.para[0]+1):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        elif(isinstance(codomain,hp_space)):
            ##         lmax==mmax                        3*nside-1==lmax
            if(self.para[0]==self.para[1])and(3*codomain.para[0]-1==self.para[0]):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,coname=None,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  a pixelization of the two-sphere.

            Parameters
            ----------
            coname : string, *optional*
                String specifying a desired codomain (default: None).

            Returns
            -------
            codomain : nifty.space
                A compatible codomain.

            Notes
            -----
            Possible arguments for `coname` are ``'gl'`` in which case a Gauss-
            Legendre pixelization [#]_ of the sphere is generated, and ``'hp'``
            in which case a HEALPix pixelization [#]_ is generated.

            References
            ----------
            .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
                   High-Resolution Discretization and Fast Analysis of Data
                   Distributed on the Sphere", *ApJ* 622..759G.
            .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
                   harmonic transforms revisited";
                   `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        """
        if(coname=="gl")or(coname is None)and(about.lm2gl.status): ## order matters
            if(self.datatype==np.complex64):
                return gl_space(self.para[0]+1,nlon=2*self.para[0]+1,datatype=np.float32) ## nlat,nlon = lmax+1,2*lmax+1
            else:
                return gl_space(self.para[0]+1,nlon=2*self.para[0]+1,datatype=np.float64) ## nlat,nlon = lmax+1,2*lmax+1

        elif(coname=="hp")or(coname is None)and(not about.lm2gl.status):
            return hp_space((self.para[0]+1)//3) ## nside = (lmax+1)/3

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported or incompatible space '"+coname+"'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            The spherical harmonics components with :math:`m=0` have meta
            volume 1, the ones with :math:`m>0` have meta volume 2, sinnce they
            each determine another component with negative :math:`m`.
        """
        if(total):
            return self.dof()
        else:
            mol = np.ones(self.dim(split=True),dtype=self.vol.dtype,order='C')
            mol[self.para[0]+1:] = 2 ## redundant in (l,m) and (l,-m)
            return mol

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_dot(self,x,y):
        """
            Computes the discrete inner product of two given arrays of field
            values.

            Parameters
            ----------
            x : numpy.ndarray
                First array
            y : numpy.ndarray
                Second array

            Returns
            -------
            dot : float
                Inner product of the two arrays.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        y = self.enforce_shape(np.array(y,dtype=self.datatype))

        ## inner product
        if(self.datatype==np.complex64):
            return gl.dotlm_f(x,y,lmax=self.para[0],mmax=self.para[1])
        else:
            return gl.dotlm(x,y,lmax=self.para[0],mmax=self.para[1])

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        elif(isinstance(codomain,gl_space)):
            ## transform
            if(self.datatype==np.complex64):
                Tx = gl.alm2map_f(x,nlat=codomain.para[0],nlon=codomain.para[1],lmax=self.para[0],mmax=self.para[1],cl=False)
            else:
                Tx = gl.alm2map(x,nlat=codomain.para[0],nlon=codomain.para[1],lmax=self.para[0],mmax=self.para[1],cl=False)
            ## weight if discrete
            if(codomain.discrete):
                Tx = codomain.calc_weight(Tx,power=0.5)

        elif(isinstance(codomain,hp_space)):
            ## transform
            Tx =  hp.alm2map(x.astype(np.complex128),codomain.para[0],lmax=self.para[0],mmax=self.para[1],pixwin=False,fwhm=0.0,sigma=None,invert=False,pol=True,inplace=False)
            ## weight if discrete
            if(codomain.discrete):
                Tx = codomain.calc_weight(Tx,power=0.5)

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel in position space.

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        elif(sigma==0):
            return x
        else:
            ## smooth
            #return gl.smoothalm(x,lmax=self.para[0],mmax=self.para[1],fwhm=0.0,sigma=sigma,overwrite=False) ## no overwrite
            return hp.smoothalm(x,fwhm=0.0,sigma=sigma,invert=False,pol=True,mmax=self.para[1],verbose=False,inplace=False) ## no overwrite

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Computes the power of an array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values of which the power is to be
                calculated.

            Returns
            -------
            spec : numpy.ndarray
                Power contained in the input array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## power spectrum
        if(self.datatype==np.complex64):
            return gl.anaalm_f(x,lmax=self.para[0],mmax=self.para[1])
        else:
            #return gl.anaalm(x,lmax=self.para[0],mmax=self.para[1])
            return hp.alm2cl(x,alms2=None,lmax=self.para[0],mmax=self.para[1],lmax_out=self.para[0],nspec=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=True,norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            power : bool, *optional*
                Whether to plot the power contained in the field or the field
                values themselves (default: True).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            mono : bool, *optional*
                Whether to plot the monopole or not (default: True).
        """
        if(not pl.isinteractive()):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x,**kwargs)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            xaxes = np.arange(self.para[0]+1,dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in range(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii],**kwargs)
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other,**kwargs)]
                imax = max(1,len(other)-1)
                for ii in range(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$l$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$l(2l+1) C_l$")
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x))
            if(np.iscomplexobj(x)):
                if(title):
                    title += " "
                self.get_plot(np.absolute(x),title=title+"(absolute)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                self.get_plot(np.real(x),title=title+"(real part)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
                self.get_plot(np.imag(x),title=title+"(imaginary part)",vmin=vmin,vmax=vmax,power=False,norm=norm,cmap=cmap,cbar=cbar,other=None,legend=False,**kwargs)
            else:
                if(vmin is None):
                    vmin = np.min(x,axis=None,out=None)
                if(vmax is None):
                    vmax = np.max(x,axis=None,out=None)
                if(norm=="log")and(vmin<=0):
                    raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

                xmesh = np.nan*np.empty(self.para[::-1]+1,dtype=np.float16,order='C') ## not a number
                xmesh[4,1] = None
                xmesh[1,4] = None
                lm = 0
                for mm in range(self.para[1]+1):
                    xmesh[mm][mm:] = x[lm:lm+self.para[0]+1-mm]
                    lm += self.para[0]+1-mm

                s_ = np.array([1,self.para[1]/self.para[0]*(1.0+0.159*bool(cbar))])
                fig = pl.figure(num=None,figsize=(6.4*s_[0],6.4*s_[1]),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
                ax0 = fig.add_axes([0.06/s_[0],0.06/s_[1],1.0-0.12/s_[0],1.0-0.12/s_[1]])
                ax0.set_axis_bgcolor([0.0,0.0,0.0,0.0])

                xaxes = np.arange(self.para[0]+2,dtype=np.int)-0.5
                yaxes = np.arange(self.para[1]+2,dtype=np.int)-0.5
                if(norm=="log"):
                    n_ = ln(vmin=vmin,vmax=vmax)
                else:
                    n_ = None
                sub = ax0.pcolormesh(xaxes,yaxes,np.ma.masked_where(np.isnan(xmesh),xmesh),cmap=cmap,norm=n_,vmin=vmin,vmax=vmax,clim=(vmin,vmax))
                ax0.set_xlim(xaxes[0],xaxes[-1])
                ax0.set_xticks([0],minor=False)
                ax0.set_xlabel(r"$l$")
                ax0.set_ylim(yaxes[0],yaxes[-1])
                ax0.set_yticks([0],minor=False)
                ax0.set_ylabel(r"$m$")
                ax0.set_aspect("equal")
                if(cbar):
                    if(norm=="log"):
                        f_ = lf(10,labelOnlyBase=False)
                        b_ = sub.norm.inverse(np.linspace(0,1,sub.cmap.N+1))
                        v_ = np.linspace(sub.norm.vmin,sub.norm.vmax,sub.cmap.N)
                    else:
                        f_ = None
                        b_ = None
                        v_ = None
                    fig.colorbar(sub,ax=ax0,orientation="horizontal",fraction=0.1,pad=0.05,shrink=0.75,aspect=20,ticks=[vmin,vmax],format=f_,drawedges=False,boundaries=b_,values=v_)
                ax0.set_title(title)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.lm_space>"

    def __str__(self):
        return "nifty.lm_space instance\n- lmax     = "+str(self.para[0])+"\n- mmax     = "+str(self.para[1])+"\n- datatype = numpy."+str(np.result_type(self.datatype))

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class gl_space(space):
    """
        ..                 __
        ..               /  /
        ..     ____ __  /  /
        ..   /   _   / /  /
        ..  /  /_/  / /  /_
        ..  \___   /  \___/  space class
        .. /______/

        NIFTY subclass for Gauss-Legendre pixelizations [#]_ of the two-sphere.

        Parameters
        ----------
        nlat : int
            Number of latitudinal bins, or rings.
        nlon : int, *optional*
            Number of longitudinal bins (default: ``2*nlat - 1``).
        datatype : numpy.dtype, *optional*
            Data type of the field values (default: numpy.float64).

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only real-valued fields on the two-sphere are supported, i.e.
        `datatype` has to be either numpy.float64 or numpy.float32.

        References
        ----------
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.

        Attributes
        ----------
        para : numpy.ndarray
            One-dimensional array containing the two numbers `nlat` and `nlon`.
        datatype : numpy.dtype
            Data type of the field values.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array containing the pixel sizes.
    """
    def __init__(self,nlat,nlon=None,datatype=None):
        """
            Sets the attributes for a gl_space class instance.

            Parameters
            ----------
            nlat : int
                Number of latitudinal bins, or rings.
            nlon : int, *optional*
                Number of longitudinal bins (default: ``2*nlat - 1``).
            datatype : numpy.dtype, *optional*
                Data type of the field values (default: numpy.float64).

            Returns
            -------
            None
        """
        ## check parameters
        if(nlat<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        if(nlat%2!=0):
            raise ValueError(about._errors.cstring("ERROR: invalid parameter ( nlat <> 2*n )."))
        if(nlon is None):
            nlon = 2*nlat-1
        elif(nlon<1):
            about.warnings.cprint("WARNING: parameter set to default.")
            nlon = 2*nlat-1
        if(nlon!=2*nlat-1):
            about.warnings.cprint("WARNING: unrecommended parameter ( nlon <> 2*nlat-1 ).")
        self.para = np.array([nlat,nlon],dtype=np.int)

        ## check data type
        if(datatype is None):
            datatype = np.float64
        elif(datatype not in [np.float32,np.float64]):
            about.warnings.cprint("WARNING: data type set to default.")
            datatype = np.float64
        self.datatype = datatype

        self.discrete = False
        self.vol = gl.vol(self.para[0],nlon=self.para[1]).astype(self.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def nlat(self):
        """
            Returns the number of latitudinal bins.

            Returns
            -------
            nlat : int
                Number of latitudinal bins, or rings.
        """
        return self.para[0]

    def nlon(self):
        """
            Returns the number of longitudinal bins.

            Returns
            -------
            nlon : int
                Number of longitudinal bins.
        """
        return self.para[1]

    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension of the space.
        """
        ## dim = nlat*nlon
        if(split):
            return np.array([self.para[0]*self.para[1]],dtype=np.int)
        else:
            return self.para[0]*self.para[1]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.

            Notes
            -----
            Since the :py:class:`gl_space` class only supports real-valued
            fields, the number of degrees of freedom is the number of pixels.
        """
        ## dof = dim
        return self.para[0]*self.para[1]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {float, numpy.ndarray, nifty.field}
                Fiducial power spectrum from which a valid power spectrum is to
                be calculated. Scalars are interpreted as constant power
                spectra.

            Returns
            -------
            spec : numpy.ndarray
                Valid power spectrum.
        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(self.datatype)
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)
        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        size = self.para[0] ## nlat

        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Raises an error since the power spectrum for a field on the sphere
            is defined via the spherical harmonics components and not its
            position-space representation.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {float, numpy.ndarray}, *optional*
                Power spectrum (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            x = np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            lmax = self.para[0]-1 ## nlat-1
            if(self.datatype==np.float32):
                x = gl.synfast_f(arg[1],nlat=self.para[0],nlon=self.para[1],lmax=lmax,mmax=lmax,alm=False)
            else:
                x = gl.synfast(arg[1],nlat=self.para[0],nlon=self.para[1],lmax=lmax,mmax=lmax,alm=False)
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=0.5)

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the space or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.

            Notes
            -----
            Compatible codomains are instances of :py:class:`gl_space` and
            :py:class:`lm_space`.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        if(isinstance(codomain,lm_space)):
            ##         nlon==2*lat-1                          lmax==nlat-1                         lmax==mmax
            if(self.para[1]==2*self.para[0]-1)and(codomain.para[0]==self.para[0]-1)and(codomain.para[0]==codomain.para[1]):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Returns
            -------
            codomain : nifty.lm_space
                A compatible codomain.
        """
        if(self.datatype==np.float32):
            return lm_space(self.para[0]-1,mmax=self.para[0]-1,datatype=np.complex64) ## lmax,mmax = nlat-1,nlat-1
        else:
            return lm_space(self.para[0]-1,mmax=self.para[0]-1,datatype=np.complex128) ## lmax,mmax = nlat-1,nlat-1

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            For Gauss-Legendre pixelizations, the meta volumes are the pixel
            sizes.
        """
        if(total):
            return self.datatype(4*pi)
        else:
            mol = np.ones(self.dim(split=True),dtype=self.datatype,order='C')
            return self.calc_weight(mol,power=1)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array with the pixel volumes to a given power.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be weighted.
            power : float, *optional*
                Power of the pixel volumes to be used (default: 1).

            Returns
            -------
            y : numpy.ndarray
                Weighted array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight
        if(self.datatype==np.float32):
            return gl.weight_f(x,self.vol,p=np.float32(power),nlat=self.para[0],nlon=self.para[1],overwrite=False)
        else:
            return gl.weight(x,self.vol,p=np.float64(power),nlat=self.para[0],nlon=self.para[1],overwrite=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array

            Notes
            -----
            Only instances of the :py:class:`lm_space` or :py:class:`gl_space`
            classes are allowed as `codomain`.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        if(isinstance(codomain,lm_space)):
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=-0.5)
            ## transform
            if(self.datatype==np.float32):
                Tx = gl.map2alm_f(x,nlat=self.para[0],nlon=self.para[1],lmax=codomain.para[0],mmax=codomain.para[1])
            else:
                Tx = gl.map2alm(x,nlat=self.para[0],nlon=self.para[1],lmax=codomain.para[0],mmax=codomain.para[1])

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel.

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        elif(sigma==0):
            return x
        else:
            ## smooth
            return gl.smoothmap(x,nlat=self.para[0],nlon=self.para[1],lmax=self.para[0]-1,mmax=self.para[0]-1,fwhm=0.0,sigma=sigma)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Computes the power of an array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values of which the power is to be
                calculated.

            Returns
            -------
            spec : numpy.ndarray
                Power contained in the input array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight if discrete
        if(self.discrete):
            x = self.calc_weight(x,power=-0.5)
        ## power spectrum
        if(self.datatype==np.float32):
            return gl.anafast_f(x,nlat=self.para[0],nlon=self.para[1],lmax=self.para[0]-1,mmax=self.para[0]-1,alm=False)
        else:
            return gl.anafast(x,nlat=self.para[0],nlon=self.para[1],lmax=self.para[0]-1,mmax=self.para[0]-1,alm=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=False,unit="",norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            power : bool, *optional*
                Whether to plot the power contained in the field or the field
                values themselves (default: False).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            mono : bool, *optional*
                Whether to plot the monopole or not (default: True).
        """
        if(not pl.isinteractive()):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x,**kwargs)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            xaxes = np.arange(self.para[0],dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in range(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii],**kwargs)
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other,**kwargs)]
                imax = max(1,len(other)-1)
                for ii in range(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$l$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$l(2l+1) C_l$")
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x,dtype=self.datatype))
            if(vmin is None):
                vmin = np.min(x,axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x,axis=None,out=None)
            if(norm=="log")and(vmin<=0):
                raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

            fig = pl.figure(num=None,figsize=(8.5,5.4),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.02,0.05,0.96,0.9])

            lon,lat = gl.bounds(self.para[0],nlon=self.para[1])
            lon = (lon-pi)*180/pi
            lat = (lat-pi/2)*180/pi
            if(norm=="log"):
                n_ = ln(vmin=vmin,vmax=vmax)
            else:
                n_ = None
            sub = ax0.pcolormesh(lon,lat,np.roll(x.reshape((self.para[0],self.para[1]),order='C'),self.para[1]//2,axis=1)[::-1,::-1],cmap=cmap,norm=n_,vmin=vmin,vmax=vmax)
            ax0.set_xlim(-180,180)
            ax0.set_ylim(-90,90)
            ax0.set_aspect("equal")
            ax0.axis("off")
            if(cbar):
                if(norm=="log"):
                    f_ = lf(10,labelOnlyBase=False)
                    b_ = sub.norm.inverse(np.linspace(0,1,sub.cmap.N+1))
                    v_ = np.linspace(sub.norm.vmin,sub.norm.vmax,sub.cmap.N)
                else:
                    f_ = None
                    b_ = None
                    v_ = None
                cb0 = fig.colorbar(sub,ax=ax0,orientation="horizontal",fraction=0.1,pad=0.05,shrink=0.5,aspect=25,ticks=[vmin,vmax],format=f_,drawedges=False,boundaries=b_,values=v_)
                cb0.ax.text(0.5,-1.0,unit,fontdict=None,withdash=False,transform=cb0.ax.transAxes,horizontalalignment="center",verticalalignment="center")
            ax0.set_title(title)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.gl_space>"

    def __str__(self):
        return "nifty.gl_space instance\n- nlat     = "+str(self.para[0])+"\n- nlon     = "+str(self.para[1])+"\n- datatype = numpy."+str(np.result_type(self.datatype))

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class hp_space(space):
    """
        ..        __
        ..      /  /
        ..     /  /___    ______
        ..    /   _   | /   _   |
        ..   /  / /  / /  /_/  /
        ..  /__/ /__/ /   ____/  space class
        ..           /__/

        NIFTY subclass for HEALPix discretizations of the two-sphere [#]_.

        Parameters
        ----------
        nside : int
            Resolution parameter for the HEALPix discretization, resulting in
            ``12*nside**2`` pixels.

        See Also
        --------
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.
        lm_space : A class for spherical harmonic components.

        Notes
        -----
        Only powers of two are allowed for `nside`.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

        Attributes
        ----------
        para : numpy.ndarray
            Array containing the number `nside`.
        datatype : numpy.dtype
            Data type of the field values, which is always numpy.float64.
        discrete : bool
            Whether or not the underlying space is discrete, always ``False``
            for spherical spaces.
        vol : numpy.ndarray
            An array with one element containing the pixel size.
    """
    niter = 0 ## default number of iterations used for transformations

    def __init__(self,nside):
        """
            Sets the attributes for a hp_space class instance.

            Parameters
            ----------
            nside : int
                Resolution parameter for the HEALPix discretization, resulting
                in ``12*nside**2`` pixels.

            Returns
            -------
            None
        """
        ## check parameters
        if(nside<1):
            raise ValueError(about._errors.cstring("ERROR: nonpositive number."))
        if(not hp.isnsideok(nside)):
            raise ValueError(about._errors.cstring("ERROR: invalid parameter ( nside <> 2**n )."))
        self.para = np.array([nside],dtype=np.int)

        self.datatype = np.float64
        self.discrete = False
        self.vol = np.array([4*pi/(12*self.para[0]**2)],dtype=self.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def nside(self):
        """
            Returns the resolution parameter.

            Returns
            -------
            nside : int
                HEALPix resolution parameter.
        """
        return self.para[0]


    def dim(self,split=False):
        """
            Computes the dimension of the space, i.e.\  the number of pixels.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension as an array with one component
                or as a scalar (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension of the space.
        """
        ## dim = 12*nside**2
        if(split):
            return np.array([12*self.para[0]**2],dtype=np.int)
        else:
            return 12*self.para[0]**2

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the space.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.

            Notes
            -----
            Since the :py:class:`hp_space` class only supports real-valued
            fields, the number of degrees of freedom is the number of pixels.
        """
        ## dof = dim
        return 12*self.para[0]**2

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Provides a valid power spectrum array from a given object.

            Parameters
            ----------
            spec : {float, numpy.ndarray, nifty.field}
                Fiducial power spectrum from which a valid power spectrum is to
                be calculated. Scalars are interpreted as constant power
                spectra.

            Returns
            -------
            spec : numpy.ndarray
                Valid power spectrum.
        """
        if(isinstance(spec,field)):
            spec = spec.val.astype(self.datatype)
        elif(np.isscalar(spec)):
            spec = np.array([spec],dtype=self.datatype)
        else:
            spec = np.array(spec,dtype=self.datatype)
        ## check finiteness
        if(not np.all(np.isfinite(spec))):
            about.warnings.cprint("WARNING: infinite value(s).")
        ## check positivity (excluding null)
        if(np.any(spec==0)):
            about.warnings.cprint("WARNING: nonpositive value(s).")
        if(np.any(spec<0)):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        size = 3*self.para[0] ## 3*nside

        ## extend
        if(np.size(spec)==1):
            spec = spec*np.ones(size,dtype=spec.dtype,order='C')
        ## size check
        elif(np.size(spec)<size):
            raise ValueError(about._errors.cstring("ERROR: size mismatch ( "+str(np.size(spec))+" < "+str(size)+" )."))
        elif(np.size(spec)>size):
            about.warnings.cprint("WARNING: power spectrum cut to size ( == "+str(size)+" ).")
            spec = spec[:size]

        return spec

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Raises an error since the power spectrum for a field on the sphere
            is defined via the spherical harmonics components and not its
            position-space representation.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "syn" (synthesizes from a given power spectrum)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            spec : {float, numpy.ndarray}, *optional*
                Power spectrum (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            x = np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="syn"):
            lmax = 3*self.para[0]-1 ## 3*nside-1
            x = hp.synfast(arg[1],self.para[0],lmax=lmax,mmax=lmax,alm=False,pol=True,pixwin=False,fwhm=0.0,sigma=None)
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=0.5)

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the space or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.

            Notes
            -----
            Compatible codomains are instances of :py:class:`hp_space` and
            :py:class:`lm_space`.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        if(isinstance(codomain,lm_space)):
            ##        3*nside-1==lmax                             lmax==mmax
            if(3*self.para[0]-1==codomain.para[0])and(codomain.para[0]==codomain.para[1]):
                return True
            else:
                about.warnings.cprint("WARNING: unrecommended codomain.")

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Returns
            -------
            codomain : nifty.lm_space
                A compatible codomain.
        """
        return lm_space(3*self.para[0]-1,mmax=3*self.para[0]-1,datatype=np.complex128) ## lmax,mmax = 3*nside-1,3*nside-1

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.

            Notes
            -----
            For HEALpix discretizations, the meta volumes are the pixel sizes.
        """
        if(total):
            return self.datatype(4*pi)
        else:
            mol = np.ones(self.dim(split=True),dtype=self.datatype,order='C')
            return self.calc_weight(mol,power=1)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.

            Notes
            -----
            Only instances of the :py:class:`lm_space` or :py:class:`hp_space`
            classes are allowed as `codomain`.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain):
            return x ## T == id

        if(isinstance(codomain,lm_space)):
            ## weight if discrete
            if(self.discrete):
                x = self.calc_weight(x,power=-0.5)
            ## check keyword arguments
            if(kwargs.has_key("iter")):
                iterations = kwargs.get("iter")
            else:
                iterations = self.niter
            ## transform
            Tx = hp.map2alm(x.astype(np.float64),lmax=codomain.para[0],mmax=codomain.para[1],iter=iterations,pol=True,use_weights=False,regression=True,datapath=None)

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel.

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        elif(sigma==0):
            return x
        else:
            ## check keyword arguments
            if(kwargs.has_key("iter")):
                iterations = kwargs.get("iter")
            else:
                iterations = self.niter
            ## smooth
            return hp.smoothing(x,fwhm=0.0,sigma=sigma,invert=False,pol=True,iter=iterations,lmax=3*self.para[0]-1,mmax=3*self.para[0]-1,use_weights=False,regression=True,datapath=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Computes the power of an array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values of which the power is to be
                calculated.

            Returns
            -------
            spec : numpy.ndarray
                Power contained in the input array.

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight if discrete
        if(self.discrete):
            x = self.calc_weight(x,power=-0.5)
        ## power spectrum
        if(kwargs.has_key("iter")):
            iterations = kwargs.get("iter")
        else:
            iterations = self.niter
        return hp.anafast(x,map2=None,nspec=None,lmax=3*self.para[0]-1,mmax=3*self.para[0]-1,iter=iterations,alm=False,pol=True,use_weights=False,regression=True,datapath=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,x,title="",vmin=None,vmax=None,power=False,unit="",norm=None,cmap=None,cbar=True,other=None,legend=False,mono=True,**kwargs):
        """
            Creates a plot of field values according to the specifications
            given by the parameters.

            Parameters
            ----------
            x : numpy.ndarray
                Array containing the field values.

            Returns
            -------
            None

            Other parameters
            ----------------
            title : string, *optional*
                Title of the plot (default: "").
            vmin : float, *optional*
                Minimum value to be displayed (default: ``min(x)``).
            vmax : float, *optional*
                Maximum value to be displayed (default: ``max(x)``).
            power : bool, *optional*
                Whether to plot the power contained in the field or the field
                values themselves (default: False).
            unit : string, *optional*
                Unit of the field values (default: "").
            norm : string, *optional*
                Scaling of the field values before plotting (default: None).
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            other : {single object, tuple of objects}, *optional*
                Object or tuple of objects to be added, where objects can be
                scalars, arrays, or fields (default: None).
            legend : bool, *optional*
                Whether to show the legend or not (default: False).
            mono : bool, *optional*
                Whether to plot the monopole or not (default: True).
            iter : int, *optional*
                Number of iterations performed in the HEALPix basis
                transformation.
        """
        if(not pl.isinteractive()):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(power):
            x = self.calc_power(x,**kwargs)

            fig = pl.figure(num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
            ax0 = fig.add_axes([0.12,0.12,0.82,0.76])

            xaxes = np.arange(3*self.para[0],dtype=np.int)
            if(vmin is None):
                vmin = np.min(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            if(vmax is None):
                vmax = np.max(x[:mono].tolist()+(xaxes*(2*xaxes+1)*x)[1:].tolist(),axis=None,out=None)
            ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*x)[1:],color=[0.0,0.5,0.0],label="graph 0",linestyle='-',linewidth=2.0,zorder=1)
            if(mono):
                ax0.scatter(0.5*(xaxes[1]+xaxes[2]),x[0],s=20,color=[0.0,0.5,0.0],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=1)

            if(other is not None):
                if(isinstance(other,tuple)):
                    other = list(other)
                    for ii in range(len(other)):
                        if(isinstance(other[ii],field)):
                            other[ii] = other[ii].power(**kwargs)
                        else:
                            other[ii] = self.enforce_power(other[ii],**kwargs)
                elif(isinstance(other,field)):
                    other = [other.power(**kwargs)]
                else:
                    other = [self.enforce_power(other,**kwargs)]
                imax = max(1,len(other)-1)
                for ii in range(len(other)):
                    ax0.loglog(xaxes[1:],(xaxes*(2*xaxes+1)*other[ii])[1:],color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],label="graph "+str(ii+1),linestyle='-',linewidth=1.0,zorder=-ii)
                    if(mono):
                        ax0.scatter(0.5*(xaxes[1]+xaxes[2]),other[ii][0],s=20,color=[max(0.0,1.0-(2*ii/imax)**2),0.5*((2*ii-imax)/imax)**2,max(0.0,1.0-(2*(ii-imax)/imax)**2)],marker='o',cmap=None,norm=None,vmin=None,vmax=None,alpha=None,linewidths=None,verts=None,zorder=-ii)
                if(legend):
                    ax0.legend()

            ax0.set_xlim(xaxes[1],xaxes[-1])
            ax0.set_xlabel(r"$l$")
            ax0.set_ylim(vmin,vmax)
            ax0.set_ylabel(r"$l(2l+1) C_l$")
            ax0.set_title(title)

        else:
            x = self.enforce_shape(np.array(x,dtype=self.datatype))
            if(norm=="log")and(np.min(x,axis=None,out=None)<=0):
                raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))
            if(cmap is None):
                cmap = pl.cm.jet ## default
            cmap.set_under(color='k',alpha=0.0) ## transparent box
            hp.mollview(x,fig=None,rot=None,coord=None,unit=unit,xsize=800,title=title,nest=False,min=vmin,max=vmax,flip="astro",remove_dip=False,remove_mono=False,gal_cut=0,format="%g",format2="%g",cbar=cbar,cmap=cmap,notext=False,norm=norm,hold=False,margins=None,sub=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.hp_space>"

    def __str__(self):
        return "nifty.hp_space instance\n- nside = "+str(self.para[0])

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------
class nested_space(space):
    """
        ..                                      __                    __
        ..                                    /  /_                 /  /
        ..      __ ___    _______   _______  /   _/  _______   ____/  /
        ..    /   _   | /   __  / /  _____/ /  /   /   __  / /   _   /
        ..   /  / /  / /  /____/ /_____  / /  /_  /  /____/ /  /_/  /
        ..  /__/ /__/  \______/ /_______/  \___/  \______/  \______|  space class

        NIFTY subclass for product spaces

        Parameters
        ----------
        nest : list
            A list of space instances that are to be combined into a product
            space.

        Notes
        -----
        Note that the order of the spaces is important for some of the methods.

        Attributes
        ----------
        nest : list
            List of the space instances that are combined into the product space, any instances of the :py:class:`nested_space` class itself are further unraveled.
        para : numpy.ndarray
            One-dimensional array containing the dimensions of all the space instances (split up into their axes when applicable) that are contained in the nested space.
        datatype : numpy.dtype
            Data type of the field values, inherited from the innermost space, i.e. that last entry in the `nest` list.
        discrete : bool
            Whether or not the product space is discrete, ``True`` only if all subspaces are discrete.
    """
    def __init__(self,nest):
        """
            Sets the attributes for a nested_space class instance.

            Parameters
            ----------
            nest : list
                A list of space instances that are to be combined into a product
                space.

            Returns
            -------
            None
        """
        if(not isinstance(nest,list)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        ## check nest
        purenest = list([])
        para = np.array([],dtype=np.int)
        for nn in nest:
            if(not isinstance(nn,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            elif(isinstance(nn,nested_space)): ## no 2nd level nesting
                for nn_ in nn.nest:
                    purenest.append(nn_)
                    para = np.append(para,nn_.dim(split=True),axis=None)
            else:
                purenest.append(nn)
                para = np.append(para,nn.dim(split=True),axis=None)
        if(len(purenest)<2):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        self.nest = purenest
        self.para = para

        ## check data type
        for nn in self.nest[:-1]:
            if(nn.datatype!=self.nest[-1].datatype): ## may conflict permutability
                about.infos.cprint("INFO: ambiguous data type.")
                break
        self.datatype = self.nest[-1].datatype

        self.discrete = np.prod([nn.discrete for nn in self.nest],axis=0,dtype=np.bool,out=None)
        self.vol = None ## not needed

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dim(self,split=False):
        """
            Computes the dimension of the product space.

            Parameters
            ----------
            split : bool, *optional*
                Whether to return the dimension split up into the dimensions of
                each subspace, each one of these split up into the number of
                pixels along each axis when applicable, or not
                (default: False).

            Returns
            -------
            dim : {int, numpy.ndarray}
                Dimension(s) of the space.
        """
        if(split):
            return self.para
        else:
            return np.prod(self.para,axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dof(self):
        """
            Computes the number of degrees of freedom of the product space, as
            the product of the degrees of freedom of each subspace.

            Returns
            -------
            dof : int
                Number of degrees of freedom of the space.
        """
        return np.prod([nn.dof() for nn in self.nest],axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_power(self,spec,**kwargs):
        """
            Raises an error since there is no canonical definition for the
            power spectrum on a generic product space.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power_index(self,irreducible=False):
        """
            Raises an error since there is no canonical definition for the
            power spectrum on a generic product space.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra indexing ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enforce_values(self,x,extend=True):
        """
            Computes valid field values from a given object, according to the
            constraints from the space instances that make up the product
            space.

            Parameters
            ----------
            x : {float, numpy.ndarray, nifty.field}
                Object to be transformed into an array of valid field values.

            Returns
            -------
            x : numpy.ndarray
                Array containing the valid field values.

            Other parameters
            ----------------
            extend : bool, *optional*
                Whether a scalar is extented to a constant array or not
                (default: True).
        """
        if(isinstance(x,field)):
            if(self==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    x = x.val
            elif(self.nest[-1]==x.domain):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    subshape = self.para[:-np.size(self.nest[-1].dim(split=True))]
                    x = np.tensordot(np.ones(subshape,dtype=self.datatype,order='C'),x.val,axes=0)
            elif(isinstance(x.domain,nested_space)):
                if(self.datatype is not x.domain.datatype):
                    raise TypeError(about._errors.cstring("ERROR: inequal data types ( '"+str(np.result_type(self.datatype))+"' <> '"+str(np.result_type(x.domain.datatype))+"' )."))
                else:
                    if(np.all(self.nest[-len(x.domain.nest):]==x.domain.nest)):
                        subshape = self.para[:np.sum([np.size(nn.dim(split=True)) for nn in self.nest[:-len(x.domain.nest)]],axis=0,dtype=np.int,out=None)]
                        x = np.tensordot(np.ones(subshape,dtype=self.datatype,order='C'),x.val,axes=0)
                    else:
                        raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            if(np.size(x)==1):
                if(extend):
                    x = self.datatype(x)*np.ones(self.para,dtype=self.datatype,order='C')
                else:
                    if(np.isscalar(x)):
                        x = np.array([x],dtype=self.datatype)
                    else:
                        x = np.array(x,dtype=self.datatype)
            else:
                x = np.array(x,dtype=self.datatype)
                if(np.ndim(x)<np.size(self.para)):
                    subshape = np.array([],dtype=np.int)
                    for ii in range(len(self.nest))[::-1]:
                        subshape = np.append(self.nest[ii].dim(split=True),subshape,axis=None)
                        if(np.all(np.array(np.shape(x))==subshape)):
                            subshape = self.para[:np.sum([np.size(nn.dim(split=True)) for nn in self.nest[:ii]],axis=0,dtype=np.int,out=None)]
                            x = np.tensordot(np.ones(subshape,dtype=self.datatype,order='C'),x,axes=0)
                            break
                else:
                    x = self.enforce_shape(x)

        if(np.size(x)!=1):
            subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
            ## enforce special properties
            x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
            x = np.array([self.nest[-1].enforce_values(xx,extend=True) for xx in x],dtype=self.datatype).reshape(self.para,order='C')

        ## check finiteness
        if(not np.all(np.isfinite(x))):
            about.warnings.cprint("WARNING: infinite value(s).")

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_values(self,**kwargs):
        """
            Generates random field values according to the specifications given
            by the parameters.

            Returns
            -------
            x : numpy.ndarray
                Valid field values.

            Other parameters
            ----------------
            random : string, *optional*
                Specifies the probability distribution from which the random
                numbers are to be drawn.
                Supported distributions are:

                - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
                - "gau" (normal distribution with zero-mean and a given standard
                    deviation or variance)
                - "uni" (uniform distribution over [vmin,vmax[)

                (default: None).
            dev : float, *optional*
                Standard deviation (default: 1).
            var : float, *optional*
                Variance, overriding `dev` if both are specified
                (default: 1).
            vmin : float, *optional*
                Lower limit for a uniform distribution (default: 0).
            vmax : float, *optional*
                Upper limit for a uniform distribution (default: 1).
        """
        arg = random.arguments(self,**kwargs)

        if(arg is None):
            return np.zeros(self.dim(split=True),dtype=self.datatype,order='C')

        elif(arg[0]=="pm1"):
            x = random.pm1(datatype=self.datatype,shape=self.dim(split=True))

        elif(arg[0]=="gau"):
            x = random.gau(datatype=self.datatype,shape=self.dim(split=True),mean=None,dev=arg[2],var=arg[3])

        elif(arg[0]=="uni"):
            x = random.uni(datatype=self.datatype,shape=self.dim(split=True),vmin=arg[1],vmax=arg[2])

        else:
            raise KeyError(about._errors.cstring("ERROR: unsupported random key '"+str(arg[0])+"'."))

        subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
        ## enforce special properties
        x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
        x = np.array([self.nest[-1].enforce_values(xx,extend=True) for xx in x],dtype=self.datatype).reshape(self.para,order='C')

        return x

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_codomain(self,conest=None,coorder=None,**kwargs):
        """
            Generates a compatible codomain to which transformations are
            reasonable.

            Parameters
            ----------
            conest : list, *optional*
                List of nested spaces of the codomain (default: None).
            coorder : list, *optional*
                Permutation of the list of nested spaces (default: None).

            Returns
            -------
            codomain : nifty.nested_space
                A compatible codomain.

            Notes
            -----
            By default, the codomain of the innermost subspace (i.e. the last
            entry of the `nest` list) is generated and the outer subspaces are
            left unchanged. If `conest` is given, this nested space is checked
            for compatibility and returned as codomain. If `conest` is not
            given but `coorder` is, the codomain is a reordered version of the
            original :py:class:`nested_space` instance.
        """
        if(conest is None):
            if(coorder is None):
                return nested_space(self.nest[:-1]+[self.nest[-1].get_codomain(**kwargs)])
            else:
                ## check coorder
                coorder = np.array(coorder,dtype=np.int).reshape(len(self.nest),order='C')
                if(np.any(np.sort(coorder,axis=0,kind="quicksort",order=None)!=np.arange(len(self.nest)))):
                    raise ValueError(about._errors.cstring("ERROR: invalid input."))
                ## check data type
                if(self.nest[np.argmax(coorder,axis=0)]!=self.datatype):
                    about.warnings.cprint("WARNING: ambiguous data type.")
                return nested_space(np.array(self.nest)[coorder].tolist()) ## list

        else:
            codomain = nested_space(conest)
            if(self.check_codomain(codomain)):
                return codomain
            else:
                raise ValueError(about._errors.cstring("ERROR: unsupported or incompatible input."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def check_codomain(self,codomain):
        """
            Checks whether a given codomain is compatible to the space or not.

            Parameters
            ----------
            codomain : nifty.space
                Space to be checked for compatibility.

            Returns
            -------
            check : bool
                Whether or not the given codomain is compatible to the space.

            Notes
            -----
            Only instances of the :py:class:`nested_space` class can be valid
            codomains.
        """
        if(not isinstance(codomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        if(self==codomain):
            return True

        elif(isinstance(codomain,nested_space)):
            ##                nest'[:-1]==nest[:-1]
            if(np.all(codomain.nest[:-1]==self.nest[:-1])):
                return self.nest[-1].check_codomain(codomain.nest[-1])
            ##   len(nest')==len(nest)
            elif(len(codomain.nest)==len(self.nest)):
                ## check permutability
                unpaired = range(len(self.nest))
                ambiguous = False
                for ii in range(len(self.nest)):
                    for jj in range(len(self.nest)):
                        if(codomain.nest[ii]==self.nest[jj]):
                            if(jj in unpaired):
                                unpaired.remove(jj)
                                break
                            else:
                                ambiguous = True
                if(len(unpaired)!=0):
                    return False
                elif(ambiguous):
                    about.infos.cprint("INFO: ambiguous permutation.")
                return True

        return False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_meta_volume(self,total=False):
        """
            Calculates the meta volumes.

            The meta volumes are the volumes associated with each component of
            a field, taking into account field components that are not
            explicitly included in the array of field values but are determined
            by symmetry conditions.

            Parameters
            ----------
            total : bool, *optional*
                Whether to return the total meta volume of the space or the
                individual ones of each field component (default: False).

            Returns
            -------
            mol : {numpy.ndarray, float}
                Meta volume of the field components or the complete space.
        """
        if(total):
            ## product
            return np.prod([nn.get_meta_volume(total=True) for nn in self.nest],axis=0,dtype=None,out=None)
        else:
            mol = self.nest[0].get_meta_volume(total=False)
            ## tensor product
            for nn in self.nest[1:]:
                mol = np.tensordot(mol,nn.get_meta_volume(total=False),axes=0)
            return mol

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_weight(self,x,power=1):
        """
            Weights a given array of field values with the pixel volumes (not
            the meta volumes) to a given power.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be weighted.
            power : float, *optional*
                Power of the pixel volumes to be used (default: 1).

            Returns
            -------
            y : numpy.ndarray
                Weighted array.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## weight
        return x*self.get_meta_volume(total=False)**power

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_dot(self,x,y):
        """
            Computes the discrete inner product of two given arrays of field
            values.

            Parameters
            ----------
            x : numpy.ndarray
                First array
            y : numpy.ndarray
                Second array

            Returns
            -------
            dot : float
                Inner product of the two arrays.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        y = self.enforce_shape(np.array(y,dtype=self.datatype))
        ## inner product
        return np.sum(np.conjugate(x)*y,axis=None,dtype=None,out=None)

    def calc_pseudo_dot(self,x,y,**kwargs):
        """
            Computes the (correctly weighted) inner product in the innermost
            subspace (i.e.\  the last one in the `nest` list).

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values for a field on the product space.
            y : numpy.ndarray
                Array of field values for a field on the innermost subspace.

            Returns
            -------
            pot : numpy.ndarray
                Array containing the field values of the outcome of the pseudo
                inner product.

            Other parameters
            ----------------
            target : nifty.space, *optional*
                Space in which the transform of the output field lives
                (default: None).

            Notes
            -----
            The outcome of the pseudo inner product calculation is a field
            defined on a nested space that misses the innermost subspace.
            Instead of a field on the innermost subspace, a field on the
            complete nested space can be provided as `y`, in which case the
            regular inner product is calculated and the output is a scalar.
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        ## analyse (sub)array
        dotspace = None
        subspace = None
        if(np.size(y)==1)or(np.all(np.array(np.shape(y))==self.nest[-1].dim(split=True))):
            dotspace = self.nest[-1]
            if(len(self.nest)==2):
                subspace = self.nest[0]
            else:
                subspace = nested_space(self.nest[:-1])
        elif(np.all(np.array(np.shape(y))==self.para)):
            about.warnings.cprint("WARNING: computing (normal) inner product.")
            return self.calc_dot(x,self.enforce_values(y,extend=True))
        else:
            dotshape = self.nest[-1].dim(split=True)
            for ii in range(len(self.nest)-1)[::-1]:
                dotshape = np.append(self.nest[ii].dim(split=True),dotshape,axis=None)
                if(np.all(np.array(np.shape(y))==dotshape)):
                    dotspace = nested_space(self.nest[ii:])
                    if(ii<2):
                        subspace = self.nest[0]
                    else:
                        subspace = nested_space(self.nest[:ii])
                    break

        if(dotspace is None):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        y = dotspace.enforce_values(y,extend=True)

        ## weight if ...
        if(not dotspace.discrete):
            y = dotspace.calc_weight(y,power=1)
        ## pseudo inner product(s)
        x = x.reshape([subspace.dim(split=False)]+dotspace.dim(split=True).tolist(),order='C')
        pot = np.array([dotspace.calc_dot(xx,y) for xx in x],dtype=subspace.datatype).reshape(subspace.dim(split=True),order='C')
        return field(subspace,val=pot,**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_transform(self,x,codomain=None,coorder=None,**kwargs):
        """
            Computes the transform of a given array of field values.

            Parameters
            ----------
            x : numpy.ndarray
                Array to be transformed.
            codomain : nifty.space, *optional*
                Target space to which the transformation shall map
                (default: self).

            Returns
            -------
            Tx : numpy.ndarray
                Transformed array

            Other parameters
            ----------------
            coorder : list, *optional*
                Permutation of the subspaces.

            Notes
            -----
            Possible transformations are reorderings of the subspaces or any
            transformations acting on a single subspace.2
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))

        if(codomain is None):
            return x ## T == id

        ## check codomain
        self.check_codomain(codomain) ## a bit pointless

        if(self==codomain)and(coorder is None):
            return x ## T == id

        elif(isinstance(codomain,nested_space)):
            if(np.all(codomain.nest[:-1]==self.nest[:-1]))and(coorder is None):
                ## reshape
                subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
                x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
                ## transform
                Tx = np.array([self.nest[-1].calc_transform(xx,codomain=codomain.nest[-1],**kwargs) for xx in x],dtype=codomain.datatype).reshape(codomain.dim(split=True),order='C')
            elif(len(codomain.nest)==len(self.nest)):#and(np.all([nn in self.nest for nn in codomain.nest]))and(np.all([nn in codomain.nest for nn in self.nest])):
                ## check coorder
                if(coorder is None):
                    coorder = -np.ones(len(self.nest),dtype=np.int,order='C')
                    for ii in range(len(self.nest)):
                        for jj in range(len(self.nest)):
                            if(codomain.nest[ii]==self.nest[jj]):
                                if(ii not in coorder):
                                    coorder[jj] = ii
                                    break
                    if(np.any(coorder==-1)):
                        raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))
                else:
                    coorder = np.array(coorder,dtype=np.int).reshape(len(self.nest),order='C')
                    if(np.any(np.sort(coorder,axis=0,kind="quicksort",order=None)!=np.arange(len(self.nest)))):
                        raise ValueError(about._errors.cstring("ERROR: invalid input."))
                    for ii in range(len(self.nest)):
                        if(codomain.nest[coorder[ii]]!=self.nest[ii]):
                            raise ValueError(about._errors.cstring("ERROR: invalid input."))
                ## compute axes permutation
                lim = np.zeros((len(self.nest),2),dtype=np.int)
                for ii in range(len(self.nest)):
                    lim[ii] = np.array([lim[ii-1][1],lim[ii-1][1]+np.size(self.nest[coorder[ii]].dim(split=True))])
                lim = lim[coorder]
                reorder = []
                for ii in range(len(self.nest)):
                    reorder += range(lim[ii][0],lim[ii][1])
                ## permute
                Tx = np.copy(x)
                for ii in range(len(reorder)):
                    while(reorder[ii]!=ii):
                        Tx = np.swapaxes(Tx,ii,reorder[ii])
                        ii_ = reorder[reorder[ii]]
                        reorder[reorder[ii]] = reorder[ii]
                        reorder[ii] = ii_
                ## check data type
                if(codomain.datatype!=self.datatype):
                    about.warnings.cprint("WARNING: ambiguous data type.")
            else:
                raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        else:
            raise ValueError(about._errors.cstring("ERROR: unsupported transformation."))

        return Tx.astype(codomain.datatype)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_smooth(self,x,sigma=0,**kwargs):
        """
            Smoothes an array of field values by convolution with a Gaussian
            kernel, acting on the innermost subspace only (i.e.\  on the last
            entry of the `nest` list).

            Parameters
            ----------
            x : numpy.ndarray
                Array of field values to be smoothed.
            sigma : float, *optional*
                Standard deviation of the Gaussian kernel, specified in units
                of length in position space of the innermost subspace
                (default: 0).

            Returns
            -------
            Gx : numpy.ndarray
                Smoothed array.

            Other parameters
            ----------------
            iter : int, *optional*
                Number of iterations (default: 0).
        """
        x = self.enforce_shape(np.array(x,dtype=self.datatype))
        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        elif(sigma==0):
            return x
        else:
            ## reshape
            subdim = np.prod(self.para[:-np.size(self.nest[-1].dim(split=True))],axis=0,dtype=np.int,out=None)
            x = x.reshape([subdim]+self.nest[-1].dim(split=True).tolist(),order='C')
            ## smooth
            return np.array([self.nest[-1].calc_smooth(xx,sigma=sigma,**kwargs) for xx in x],dtype=self.datatype).reshape(self.dim(split=True),order='C')

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def calc_power(self,x,**kwargs):
        """
            Raises an error since there is no canonical definition for the
            power spectrum on a generic product space.
        """
        raise AttributeError(about._errors.cstring("ERROR: power spectra ill-defined."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.nested_space>"

    def __str__(self):
        return "nifty.nested_space instance\n- nest = "+str(self.nest)

##-----------------------------------------------------------------------------





##=============================================================================

class field(object):
    """
        ..         ____   __             __          __
        ..       /   _/ /__/           /  /        /  /
        ..      /  /_   __   _______  /  /    ____/  /
        ..     /   _/ /  / /   __  / /  /   /   _   /
        ..    /  /   /  / /  /____/ /  /_  /  /_/  /
        ..   /__/   /__/  \______/  \___/  \______|  class

        Basic NIFTy class for fields.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar, ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by kwargs.

        target : space, *optional*
            The space wherein the operator output lives (default: domain).


        Other Parameters
        ----------------
        random : string
            Indicates that the field values should be drawn from a certain
            distribution using a pseudo-random number generator.
            Supported distributions are:

            - "pm1" (uniform distribution over {+1,-1} or {+1,+i,-1,-i}
            - "gau" (normal distribution with zero-mean and a given standard
                deviation or variance)
            - "syn" (synthesizes from a given power spectrum)
            - "uni" (uniform distribution over [vmin,vmax[)

        dev : scalar
            Sets the standard deviation of the Gaussian distribution
            (default=1).

        var : scalar
            Sets the variance of the Gaussian distribution, outranking the dev
            parameter (default=1).

        spec : {scalar, ndarray}
            Specifies a power spectrum from which the field values should be
            synthesized (default=1). Can be given as a constant, or as an
            array with indvidual entries per mode.

        vmin : scalar
            Sets the lower limit for the uniform distribution.
        vmax : scalar
            Sets the upper limit for the uniform distribution.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar, ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by the keyword arguments.

        target : space, *optional*
            The space wherein the operator output lives (default: domain).

    """
    def __init__(self,domain,val=None,target=None,**kwargs):
        """
            Sets the attributes for a field class instance.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        val : {scalar,ndarray}, *optional*
            Defines field values, either to be given by a number interpreted
            as a constant array, or as an arbitrary array consistent with the
            space defined in domain or to be drawn from a random distribution
            controlled by the keyword arguments.

        target : space, *optional*
            The space wherein the operator output lives (default: domain).

        Returns
        -------
        Nothing

        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        if(val is None):
            self.val = self.domain.get_random_values(**kwargs)
        else:
            self.val = self.domain.enforce_values(val,extend=True)

        if(target is None):
            target = domain.get_codomain()
        ## check codomain
        self.domain.check_codomain(target) ## a bit pointless
        self.target = target

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dim(self,split=False):
        """
        Computes the (array) dimension of the underlying space.

        Parameters
        ----------
        split : bool
            Sets the output to be either split up per axis or
            in form of total number of field entries in all
            dimensions (default=False)

        Returns
        -------
        dim : {scalar, ndarray}
            Dimension of space.

        """

        return self.domain.dim(split=split)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def cast_domain(self,newdomain,newtarget=None,force=True):
        """
            Casts the domain of the field.

            Parameters
            ----------
            newdomain : space
                New space wherein the field should live.

            newtarget : space, *optional*
                Space wherein the transform of the field should live.
                When not given, target will automatically be the codomain
                of the newly casted domain (default=None).

            force : bool, *optional*
                Whether to force reshaping of the field if necessary or not
                (default=True)

            Returns
            -------
            Nothing

        """
        if(not isinstance(newdomain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(newdomain.datatype is not self.domain.datatype):
            raise TypeError(about._errors.cstring("ERROR: inequal data types '"+str(np.result_type(newdomain.datatype))+"' and '"+str(np.result_type(self.domain.datatype))+"'."))
        elif(newdomain.dim(split=False)!=self.domain.dim(split=False)):
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(newdomain.dim(split=False))+" <> "+str(self.domain.dim(split=False))+" )."))

        if(force):
            newshape = newdomain.dim(split=True)
            if(not np.all(newshape==self.domain.dim(split=True))):
                about.infos.cprint("INFO: reshaping forced.")
                self.val.shape = newshape
        else:
            if(not np.all(newdomain.dim(split=True)==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(newdomain.dim(split=True))+" <> "+str(self.domain.dim(split=True))+" )."))

        self.domain = newdomain

        ## check target
        if(newtarget is None):
            if(not self.domain.check_codomain(self.target)):
                if(force):
                    about.infos.cprint("INFO: codomain set to default.")
                else:
                    about.warnings.cprint("WARNING: codomain set to default.")
                self.set_target(newtarget=self.domain.get_codomain())
        else:
            self.set_target(newtarget=newtarget)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_val(self,newval):
        """
            Resets the field values.

            Parameters
            ----------
            newval : {scalar, ndarray}
                New field values either as a constant or an arbitrary array.

        """
        self.val = self.domain.enforce_values(newval,extend=True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_target(self,newtarget=None):
        """
            Resets the codomain of the field.

            Parameters
            ----------
            newtarget : space
                 The new space wherein the transform of the field should live.
                 (default=None).

        """
        if(newtarget is None):
            newtarget = self.domain.get_codomain()
        ## check codomain
        self.domain.check_codomain(newtarget) ## a bit pointless
        self.target = newtarget

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def weight(self,power=1,overwrite=False):
        """
            Returns the field values, weighted with the volume factors to a
            given power. The field values will optionally be overwritten.

            Parameters
            ----------
            power : scalar, *optional*
                Specifies the optional power coefficient to which the field
                values are taken (default=1).

            overwrite : bool, *optional*
                Whether to overwrite the field values or not (default: False).

            Returns
            -------
            field   : field, *optional*
                If overwrite is False, the weighted field is returned.
                Otherwise, nothing is returned.

        """
        if(overwrite):
            self.val = self.domain.calc_weight(self.val,power=power)
        else:
            return field(self.domain,val=self.domain.calc_weight(self.val,power=power),target=self.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def dot(self,x=None):
        """
            Computes the inner product of the field with a given object
            implying the correct volume factor needed to reflect the
            discretization of the continuous fields.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Returns
            -------
            dot : scalar
                The result of the inner product.

        """
        if(x is None):
            x = self.val
        if(isinstance(x,field)):
            if(self.domain!=x.domain):
                try: ## to transform field
                    x = x.transform(target=self.domain,overwrite=False)
                except(ValueError):
                    if(np.size(x.dim(split=True))>np.size(self.dim(split=True))): ## switch
                        return x.dot(x=self)
                    else:
                        try: ## to complete subfield
                            x = field(self.domain,val=x,target=self.target)
                        except(TypeError,ValueError):
                            try: ## to complete transformed subfield
                                x = field(self.domain,val=x.transform(target=x.target,overwrite=False),target=self.target)
                            except(TypeError,ValueError):
                                raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
            if(x.domain.datatype>self.domain.datatype):
                if(not self.domain.discrete):
                    return x.domain.calc_dot(self.val.astype(x.domain.datatype),x.weight(power=1,overwrite=False))
                else:
                    return x.domain.calc_dot(self.val.astype(x.domain.datatype),x.val)
            else:
                if(not self.domain.discrete):
                    return self.domain.calc_dot(self.val,self.domain.calc_weight(x.val.astype(self.domain.datatype),power=1))
                else:
                    return self.domain.calc_dot(self.val,x.val.astype(self.domain.datatype))
        else:
            x = self.domain.enforce_values(x,extend=True)
            if(not self.domain.discrete):
                x = self.domain.calc_weight(x,power=1)
            return self.domain.calc_dot(self.val,x)

    def norm(self,q=None):
        """
            Computes the Lq-norm of the field values.

            Parameters
            ----------
            q : scalar
                Parameter q of the Lq-norm (default: 2).

            Returns
            -------
            norm : scalar
                The Lq-norm of the field values.

        """
        if(q is None):
            return np.sqrt(self.dot(x=self.val))
        else:
            return self.dot(x=self.val**(q-1))**(1/q)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def pseudo_dot(self,x=1,**kwargs):
        """
            Computes the pseudo inner product of the field with a given object
            implying the correct volume factor needed to reflect the
            discretization of the continuous fields. This method specifically
            handles the inner products of fields defined over a
            :py:class:`nested_space`.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Other Parameters
            ----------------
            target : space, *optional*
                space wherein the transform of the output field should live
                (default: None).

            Returns
            -------
            pot : ndarray
                The result of the pseudo inner product.

            Examples
            --------
            Pseudo inner product of a field defined over a nested space with
            a simple field defined over a rg_space.

            >>> from nifty import *
            >>> space = rg_space(2)
            >>> nspace = nested_space([space,space])
            >>> nval = array([[1,2],[3,4]])
            >>> nfield = nifty.field(domain = nspace, val = nval)
            >>> val = array([1,1])
            >>> nfield.pseudo_dot(x=val).val
            array([ 1.5,  3.5])

        """
        ## check attribute
        if(not hasattr(self.domain,"calc_pseudo_dot")):
            if(isinstance(x,field)):
                if(hasattr(x.domain,"calc_pseudo_dot")):
                    return x.pseudo_dot(x=self,**kwargs)
            about.warnings.cprint("WARNING: computing (normal) inner product.")
            return self.dot(x=x)
        ## strip field (calc_pseudo_dot handles subspace)
        if(isinstance(x,field)):
            if(np.size(x.dim(split=True))>np.size(self.dim(split=True))): ## switch
                return x.pseudo_dot(x=self,**kwargs)
            else:
                try:
                    return self.pseudo_dot(x=x.val,**kwargs)
                except(TypeError,ValueError):
                    try:
                        return self.pseudo_dot(x=x.transform(target=x.target,overwrite=False).val,**kwargs)
                    except(TypeError,ValueError):
                        raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        ## pseudo inner product (calc_pseudo_dot handles weights)
        else:
            if(np.isscalar(x)):
                x = np.array([x],dtype=self.domain.datatype)
            else:
                x = np.array(x,dtype=self.domain.datatype)

            if(np.size(x)>self.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(x))+" <> "+str(self.dim(split=False))+" )."))
            elif(np.size(x)==self.dim(split=False)):
                about.warnings.cprint("WARNING: computing (normal) inner product.")
                return self.dot(x=x)
            else:
                return self.domain.calc_pseudo_dot(self.val,x,**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tensor_dot(self,x=None,**kwargs):
        """
            Computes the tensor product of a field defined on a arbitrary domain
            with a given object defined on another arbitrary domain.

            Parameters
            ----------
            x : {scalar, ndarray, field}, *optional*
                The object with which the inner product is computed
                (default=None).

            Other Parameters
            ----------------
            target : space, *optional*
                space wherein the transform of the output field should live
                (default: None).

            Returns
            -------
            tot : field
                The result of the tensor product, a field defined over a nested
                space.

        """
        if(x is None):
            return self
        elif(isinstance(x,field)):
            return field(nested_space([self.domain,x.domain]),val=np.tensordot(self.val,x.val,axes=0),**kwargs)
        else:
            return field(nested_space([self.domain,self.domain]),val=np.tensordot(self.val,self.domain.enforce_values(x,extend=True),axes=0),**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def conjugate(self):
        """
            Computes the complex conjugate of the field.

            Returns
            -------
            cc : field
                The complex conjugated field.

        """
        return field(self.domain,val=np.conjugate(self.val),target=self.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def transform(self,target=None,overwrite=False,**kwargs):
        """
            Computes the transform of the field using the appropriate conjugate
            transformation.

            Parameters
            ----------
            target : space, *optional*
                Domain of the transform of the field (default:self.target)

            overwrite : bool, *optional*
                Whether to overwrite the field or not (default: False).

            Other Parameters
            ----------------
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            field : field, *optional*
                If overwrite is False, the transformed field is returned.
                Otherwise, nothing is returned.

        """
        if(target is None):
            target = self.target
        else:
            self.domain.check_codomain(target) ## a bit pointless
        if(overwrite):
            self.val = self.domain.calc_transform(self.val,codomain=target,**kwargs)
            self.target = self.domain
            self.domain = target
        else:
            return field(target,val=self.domain.calc_transform(self.val,codomain=target,**kwargs),target=self.domain)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def smooth(self,sigma=0,overwrite=False,**kwargs):
        """
            Smoothes the field by convolution with a Gaussian kernel.

            Parameters
            ----------
            sigma : scalar, *optional*
                standard deviation of the Gaussian kernel specified in units of
                length in position space (default=1)

            overwrite : bool, *optional*
                Whether to overwrite the field or not (default: False).

            Other Parameters
            ----------------
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            field : field, *optional*
                If overwrite is False, the transformed field is returned.
                Otherwise, nothing is returned.

        """
        if(overwrite):
            self.val = self.domain.calc_smooth(self.val,sigma=sigma,**kwargs)
        else:
            return field(self.domain,val=self.domain.calc_smooth(self.val,sigma=sigma,**kwargs),target=self.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def power(self,**kwargs):
        """
            Computes the power spectrum of the field values.

            Other Parameters
            ----------------
            pindex : ndarray
                Specifies the indexing array for the distribution of
                indices in conjugate space (default: None).
            kindex : numpy.ndarray, *optional*
                Scale corresponding to each band in the power spectrum
                (default: None).
            rho : scalar
                Number of degrees of freedom per irreducible band
                (default=None).
            iter : scalar
                Number of iterations (default: 0)

            Returns
            -------
            spec : ndarray
                Returns the power spectrum.

        """
        return self.domain.calc_power(self.val,**kwargs)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def hat(self):
        """
            Translates the field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        return diagonal_operator(domain=self.domain,diag=self.val,bare=False)

    def inverse_hat(self):
        """
            Translates the inverted field into a diagonal operator.

            Returns
            -------
            D : operator
                The new diagonal operator instance.

        """
        if(np.any(self.val==0)):
            raise AttributeError(about._errors.cstring("ERROR: singular operator."))
        else:
            return diagonal_operator(domain=self.domain,diag=1/self.val,bare=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def plot(self,**kwargs):
        """
            Plots the field values using matplotlib routines.

            Other Parameters
            ----------------
            title : string
                Title of the plot (default= "").
            vmin : scalar
                Minimum value displayed (default=min(x)).
            vmax : scalar
                Maximum value displayed (default=max(x)).
            power : bool
                Whether to plot the power spectrum or the array (default=None).
            unit : string
                The unit of the field values (default="").
            norm : scalar
                A normalization (default=None).
            cmap : cmap
                A color map (default=None).
            cbar : bool
                Whether to show the color bar or not (default=True).
            other : {scalar, ndarray, fiels}
                Object or tuple of objects to be added (default=None).
            legend : bool
                Whether to show the legend or not (default=False).
            mono : bool
                Whether to plot the monopol of the power spectrum or not
                (default=True).
            error : {scalar, ndarray, fiels}
                object indicating some confidence intervall (default=None).
            iter : scalar
                Number of iterations (default: 0).
            kindex : scalar
                The spectral index per irreducible band (default=None).

            Notes
            -----
            The applicability of the keyword arguments depends on the
            respective space on which the field is defined. Confer to the
            corresponding :py:meth:`get_plot` method.

        """
        interactive = pl.isinteractive()
        if(not interactive):
            pl.ion()

        self.domain.get_plot(self.val,**kwargs)

        if(not interactive):
            pl.ioff()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.field>"

    def __str__(self):
        minmax = [np.min(self.val,axis=None,out=None),np.max(self.val,axis=None,out=None)]
        medmean = [np.median(self.val,axis=None,out=None,overwrite_input=False),np.mean(self.val,axis=None,dtype=self.domain.datatype,out=None)]
        return "nifty.field instance\n- domain      = "+repr(self.domain)+"\n- val         = [...]"+"\n  - min.,max. = "+str(minmax)+"\n  - med.,mean = "+str(medmean)+"\n- target      = "+repr(self.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __len__(self):
        return int(self.dim(split=True)[0])

    def __getitem__(self,key):
        return self.val[key]

    def __setitem__(self,key,value):
        self.val[key] = self.domain.datatype(value)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __pos__(self):
        return field(self.domain,val=+self.val,target=self.target)

    def __neg__(self):
        return field(self.domain,val=-self.val,target=self.target)

    def __abs__(self):
        if(np.iscomplexobj(self.val)):
            return np.absolute(self.val)
        else:
            return field(self.domain,val=np.absolute(self.val),target=self.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __add__(self,x): ## __add__ : self + x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=self.val.astype(x.domain.datatype)+x.val,target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=self.val+x.val.astype(self.domain.datatype),target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=self.val+x,target=self.target)

    __radd__ = __add__  ## __add__ : x + self

    def __iadd__(self,x): ## __iadd__ : self += x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val += x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val += x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val += x
        return self

    def __sub__(self,x): ## __sub__ : self - x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=self.val.astype(x.domain.datatype)-x.val,target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=self.val-x.val.astype(self.domain.datatype),target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=self.val-x,target=self.target)

    def __rsub__(self,x): ## __rsub__ : x - self
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=x.val-self.val.astype(x.domain.datatype),target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=x.val.astype(self.domain.datatype)-self.val,target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=x-self.val,target=self.target)

    def __isub__(self,x): ## __isub__ : self -= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):

                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val -= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val -= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val -= x
        return self

    def __mul__(self,x): ## __mul__ : self * x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=self.val.astype(x.domain.datatype)*x.val,target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=self.val*x.val.astype(self.domain.datatype),target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=self.val*x,target=self.target)

    __rmul__ = __mul__  ## __rmul__ : x * self

    def __imul__(self,x): ## __imul__ : self *= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val *= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val *= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val *= x
        return self

    def __div__(self,x): ## __div__ : self / x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=self.val.astype(x.domain.datatype)/x.val,target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=self.val/x.val.astype(self.domain.datatype),target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=self.val/x,target=self.target)

    __truediv__ = __div__

    def __rdiv__(self,x): ## __rdiv__ : x / self
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=x.val/self.val.astype(x.domain.datatype),target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=x.val.astype(self.domain.datatype)/self.val,target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=x/self.val,target=self.target)

    __rtruediv__ = __rdiv__

    def __idiv__(self,x): ## __idiv__ : self /= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val /= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val /= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val /= x
        return self

    __itruediv__ = __idiv__

    def __pow__(self,x): ## __pow__(): self ** x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=np.power(self.val.astype(x.domain.datatype),x.val),target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=np.power(self.val,x.val.astype(self.domain.datatype)),target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=np.power(self.val,x),target=self.target)

    def __rpow__(self,x): ## __pow__(): x ** self
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    about.warnings.cprint("WARNING: codomain set to default.")
                    return field(x.domain,val=np.power(x.val,self.val.astype(x.domain.datatype)),target=x.domain.get_codomain())
                else:
                    return field(self.domain,val=np.power(x.val.astype(self.domain.datatype),self.val),target=self.target)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            return field(self.domain,val=np.power(x,self.val),target=self.target)

    def __ipow__(self,x): ## __pow__(): self **= x
        if(isinstance(x,field)):
            if(self.domain==x.domain):
                if(x.domain.datatype>self.domain.datatype):
                    self.domain = x.domain
                    self.val **= x.val
                    about.warnings.cprint("WARNING: codomain set to default.")
                    self.target = x.domain.get_codomain()
                else:
                    self.val **= x.val.astype(self.domain.datatype)
            else:
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
        else:
            x = self.domain.enforce_values(x,extend=False)
            self.val **= x
        return self

##=============================================================================



##-----------------------------------------------------------------------------

def cos(x):
    """
        Returns the cos of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        cosx : {scalar, array, field}
            Cosine of `x` to the specified base.

        See also
        --------
        sin
        tan

        Examples
        --------
        >>> cos([-1,1])
        array([ 0.54030231,  0.54030231])
        >>> cos(field(point_space(2), val=[10, 100])).val
        array([ 0.54030231,  0.54030231])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.cos(x.val),target=x.target)
    else:
        return np.cos(np.array(x))

def sin(x):
    """
        Returns the sine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        sinx : {scalar, array, field}
            Sine of `x` to the specified base.

        See also
        --------
        cos
        tan

        Examples
        --------
        >>> sin([-1,1])
        array([-0.84147098,  0.84147098])
        >>> sin(field(point_space(2), val=[-1, 1])).val
        array([-0.84147098,  0.84147098])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.sin(x.val),target=x.target)
    else:
        return np.sin(np.array(x))

def cosh(x):
    """
        Returns the hyperbolic cosine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        coshx : {scalar, array, field}
            cosh of `x` to the specified base.

        See also
        --------
        sinh
        tanh

        Examples
        --------
        >>> cosh([-1,1])
        array([ 1.54308063,  1.54308063])
        >>> cosh(field(point_space(2), val=[-1, 1])).val
        array([ 1.54308063,  1.54308063])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.cosh(x.val),target=x.target)
    else:
        return np.cosh(np.array(x))

def sinh(x):
    """
        Returns the hyperbolic sine  of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        sinhx : {scalar, array, field}
            sinh of `x` to the specified base.

        See also
        --------
        cosh
        tanh

        Examples
        --------
        >>> sinh([-1,1])
        array([-1.17520119,  1.17520119])
        >>> sinh(field(point_space(2), val=[-1, 1])).val
        array([-1.17520119,  1.17520119])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.sinh(x.val),target=x.target)
    else:
        return np.sinh(np.array(x))

def tan(x):
    """
        Returns the tangent of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        tanx : {scalar, array, field}
            Tangent of `x` to the specified base.

        See also
        --------
        cos
        sin

        Examples
        --------
        >>> tan([10,100])
        array([ 0.64836083, -0.58721392])
        >>> tan(field(point_space(2), val=[10, 100])).val
        array([ 0.64836083, -0.58721392])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.tan(x.val),target=x.target)
    else:
        return np.tan(np.array(x))

def tanh(x):
    """
        Returns the hyperbolic tangent of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        tanhx : {scalar, array, field}
            tanh of `x` to the specified base.

        See also
        --------
        cosh
        sinh

        Examples
        --------
        >>> tanh([-1,1])
        array([-0.76159416,  0.76159416])
        >>> tanh(field(point_space(2), val=[-1, 1])).val
        array([-0.76159416,  0.76159416])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.tanh(x.val),target=x.target)
    else:
        return np.tanh(np.array(x))

def arccos(x):
    """
        Returns the arccosine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arccosx : {scalar, array, field}
            arccos of `x` to the specified base.

        See also
        --------
        arcsin
        arctan

        Examples
        --------
        >>> arccos([-1,1])
        array([ 3.14159265,  0.        ])
        >>> arccos(field(point_space(2), val=[-1, 1])).val
        array([ 3.14159265,  0.        ])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.arccos(x.val),target=x.target)
    else:
        return np.arccos(np.array(x))

def arcsin(x):
    """
        Returns the arcsine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arcsinx : {scalar, array, field}
            Logarithm of `x` to the specified base.

        See also
        --------
        arccos
        arctan

        Examples
        --------
        >>> arcsin([-1,1])
        array([-1.57079633,  1.57079633])
        >>> arcsin(field(point_space(2), val=[-1, 1])).val
        array([-1.57079633,  1.57079633])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.arcsin(x.val),target=x.target)
    else:
        return np.arcsin(np.array(x))

def arccosh(x):
    """
        Returns the hyperbolic arccos of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arccoshx : {scalar, array, field}
            arccos of `x` to the specified base.

        See also
        --------
        arcsinh
        arctanh

        Examples
        --------
        >>> arcosh([1,10])
        array([ 0.        ,  2.99322285])
        >>> arccosh(field(point_space(2), val=[1, 10])).val
        array([ 0.        ,  2.99322285])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.arccosh(x.val),target=x.target)
    else:
        return np.arccosh(np.array(x))

def arcsinh(x):
    """
        Returns the hypberbolic sin of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arcsinhx : {scalar, array, field}
            arcsinh of `x` to the specified base.

        See also
        --------
        arccosh
        arctanh

        Examples
        --------
        >>> arcsinh([1,10])
        array([ 0.88137359,  2.99822295])
        >>> arcsinh(field(point_space(2), val=[1, 10])).val
        array([ 0.88137359,  2.99822295])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.arcsinh(x.val),target=x.target)
    else:
        return np.arcsinh(np.array(x))

def arctan(x):
    """
        Returns the arctan of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arctanx : {scalar, array, field}
            arctan of `x` to the specified base.

        See also
        --------
        arccos
        arcsin

        Examples
        --------
        >>> arctan([1,10])
        array([ 0.78539816,  1.47112767])
        >>> arctan(field(point_space(2), val=[1, 10])).val
        array([ 0.78539816,  1.47112767])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.arctan(x.val),target=x.target)
    else:
        return np.arctan(np.array(x))

def arctanh(x):
    """
        Returns the hyperbolic arc tangent of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arctanhx : {scalar, array, field}
            arctanh of `x` to the specified base.

        See also
        --------
        arccosh
        arcsinh

        Examples
        --------
        >>> arctanh([0,0.5])
        array([ 0.        ,  0.54930614])
        >>> arctanh(field(point_space(2), val=[0, 0.5])).val
        array([ 0.        ,  0.54930614])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.arctanh(x.val),target=x.target)
    else:
        return np.arctanh(np.array(x))

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def sqrt(x):
    """
        Returns the square root of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        sqrtx : {scalar, array, field}
            Square root of `x`.

        Examples
        --------
        >>> sqrt([10,100])
        array([ 10.       ,  31.6227766])
        >>> sqrt(field(point_space(2), val=[10, 100])).val
        array([ 10.       ,  31.6227766])

    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.sqrt(x.val),target=x.target)
    else:
        return np.sqrt(np.array(x))

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def exp(x):
    """
        Returns the exponential of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        expx : {scalar, array, field}
            Exponential of `x` to the specified base.

        See also
        --------
        log

        Examples
        --------
        >>> exp([10,100])
        array([  2.20264658e+04,   2.68811714e+43])
        >>> exp(field(point_space(2), val=[10, 100])).val
        array([  2.20264658e+04,   2.68811714e+43])
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.exp(x.val),target=x.target)
    else:
        return np.exp(np.array(x))

def log(x,base=None):
    """
        Returns the logarithm with respect to a specified base.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.
        base : {scalar, list, array, field}, *optional*
            Base of the logarithm (default: Euler's number).

        Returns
        -------
        logx : {scalar, array, field}
            Logarithm of `x` to the specified base.

        See also
        --------
        exp

        Examples
        --------
        >>> log([100, 1000], base=10)
        array([ 2.,  3.])
        >>> log(field(point_space(2), val=[100, 1000]), base=10).val
        array([ 2.,  3.])

    """
    if(base is None):
        if(isinstance(x,field)):
            return field(x.domain,val=np.log(x.val),target=x.target)
        else:
            return np.log(np.array(x))

    base = np.array(base)
    if(np.all(base>0)):
        if(isinstance(x,field)):
            return field(x.domain,val=np.log(x.val)/np.log(base).astype(x.domain.datatype),target=x.target)
        else:
            return np.log(np.array(x))/np.log(base)
    else:
        raise ValueError(about._errors.cstring("ERROR: invalid input basis."))

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def conjugate(x):
    """
        Computes the complex conjugate of a given object.

        Parameters
        ----------
        x : {ndarray, field}
            The object to be complex conjugated.

        Returns
        -------
        conjx : {ndarray,field}
            The complex conjugated object.
    """
    if(isinstance(x,field)):
        return field(x.domain,val=np.conjugate(x.val),target=x.target)
    else:
        return np.conjugate(np.array(x))

##-----------------------------------------------------------------------------


##-----------------------------------------------------------------------------





##=============================================================================

class operator(object):
    """
        ..                                                      __
        ..                                                    /  /_
        ..    ______    ______    _______   _____   ____ __  /   _/  ______    _____
        ..  /   _   | /   _   | /   __  / /   __/ /   _   / /  /   /   _   | /   __/
        .. /  /_/  / /  /_/  / /  /____/ /  /    /  /_/  / /  /_  /  /_/  / /  /
        .. \______/ /   ____/  \______/ /__/     \______|  \___/  \______/ /__/     class
        ..         /__/

        NIFTY base class for (linear) operators

        The base NIFTY operator class is an abstract class from which other
        specific operator subclasses, including those preimplemented in NIFTY
        (e.g. the diagonal operator class) must be derived.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool, *optional*
            Indicates whether the operator is self-adjoint or not
            (default: False)
        uni : bool, *optional*
            Indicates whether the operator is unitary or not
            (default: False)
        imp : bool, *optional*
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not (default: False)
        target : space, *optional*
            The space wherein the operator output lives (default: domain)
        para : {single object, list of objects}, *optional*
            This is a freeform list of parameters that derivatives of the
            operator class can use. Not used in the base operators.
            (default: None)

        See Also
        --------
        diagonal_operator :  An operator class for handling purely diagonal
            operators.
        power_operator : Similar to diagonal_operator but with handy features
            for dealing with diagonal operators whose diagonal
            consists of a power spectrum.
        vecvec_operator : Operators constructed from the outer product of two
            fields
        response_operator : Implements a modeled instrument response which
            translates a signal into data space.
        projection_operator : An operator that projects out one or more
            components in a basis, e.g. a spectral band
            of Fourier components.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
        para : {single object, list of objects}
            This is a freeform list of parameters that derivatives of the
            operator class can use. Not used in the base operators.
    """
    def __init__(self,domain,sym=False,uni=False,imp=False,target=None,para=None):
        """
            Sets the attributes for an operator class instance.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            sym : bool, *optional*
                Indicates whether the operator is self-adjoint or not
                (default: False)
            uni : bool, *optional*
                Indicates whether the operator is unitary or not
                (default: False)
            imp : bool, *optional*
                Indicates whether the incorporation of volume weights in
                multiplications is already implemented in the `multiply`
                instance methods or not (default: False)
            target : space, *optional*
                The space wherein the operator output lives (default: domain)
            para : {object, list of objects}, *optional*
                This is a freeform list of parameters that derivatives of the
                operator class can use. Not used in the base operators.
                (default: None)

            Returns
            -------
            None
        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        self.sym = bool(sym)
        self.uni = bool(uni)

        if(self.domain.discrete):
            self.imp = False
        else:
            self.imp = bool(imp)

        if(target is None)or(self.sym)or(self.uni):
            target = self.domain
        if(not isinstance(target,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.target = target

        if(para is not None):
            self.para = para

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def nrow(self):
        """
            Computes the number of rows.

            Returns
            -------
            nrow : int
                number of rows (equal to the dimension of the codomain)
        """
        return self.target.dim(split=False)

    def ncol(self):
        """
            Computes the number of columns

            Returns
            -------
            nrow : int
                number of columns (equal to the dimension of the domain)
        """
        return self.domain.dim(split=False)

    def dim(self,axis=None):
        """
            Computes the dimension of the space

            Parameters
            ----------
            axis : int, *optional*
                Axis along which the dimension is to be calculated.
                (default: None)

            Returns
            -------
            dim : {int, ndarray}
                The dimension(s) of the operator.

        """
        if(axis is None):
            return np.array([self.nrow(),self.ncol()])
        elif(axis==0):
            return self.nrow()
        elif(axis==1):
            return self.ncol()
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input axis."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_para(self,newpara):
        """
            Sets the parameters and creates the `para` property if it does
            not exist

            Parameters
            ----------
            newpara : {object, list of objects}
                A single parameter or a list of parameters.

            Returns
            -------
            None

        """
        self.para = newpara

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'multiply'."))

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'adjoint_multiply'."))

    def _inverse_multiply(self,x,**kwargs): ## > applies the inverse operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'inverse_multiply'."))

    def _adjoint_inverse_multiply(self,x,**kwargs): ## > applies the inverse adjoint operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'adjoint_inverse_multiply'."))

    def _inverse_adjoint_multiply(self,x,**kwargs): ## > applies the adjoint inverse operator to a given field
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'inverse_adjoint_multiply'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _briefing(self,x,domain,inverse): ## > prepares x for `multiply`
        ## inspect x
        if(not isinstance(x,field)):
            x_ = field(domain,val=x,target=None)
        else:
            ## check x.domain
            if(domain==x.domain):
                x_ = x
            ## transform
            else:
                x_ = x.transform(target=domain,overwrite=False)
        ## weight if ...
        if(not self.imp)and(not domain.discrete)and(not inverse):
                x_ = x_.weight(power=1,overwrite=False)
        return x_

    def _debriefing(self,x,x_,target,inverse): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        else:
            ## inspect x_
            if(not isinstance(x_,field)):
                x_ = field(target,val=x_,target=None)
            elif(x_.domain!=target):
                raise ValueError(about._errors.cstring("ERROR: invalid output domain."))
            ## weight if ...
            if(not self.imp)and(not target.discrete)and(inverse):
                x_ = x_.weight(power=-1,overwrite=False)
            ## inspect x
            if(isinstance(x,field)):
                ## repair ...
                if(self.domain==self.target!=x.domain):
                    x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
                if(x_.domain==x.domain)and(x_.target!=x.target):
                    x_.set_target(newtarget=x.target) ## ... codomain
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def times(self,x,**kwargs):
        """
            Applies the operator to a given object

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain of the operator.

            Returns
            -------
            Ox : field
                Mapped field on the target domain of the operator.
        """
        ## prepare
        x_ = self._briefing(x,self.domain,False)
        ## apply operator
        x_ = self._multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,False)

    def __call__(self,x,**kwargs):
        return self.times(x,**kwargs)

    def adjoint_times(self,x,**kwargs):
        """
            Applies the adjoint operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OAx : field
                Mapped field on the domain of the operator.

        """
        ## check whether self-adjoint
        if(self.sym):
            return self.times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.inverse_times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.target,False)
        ## apply operator
        x_ = self._adjoint_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.domain,False)

    def inverse_times(self,x,**kwargs):
        """
            Applies the inverse operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the domain space of the operator.

            Returns
            -------
            OIx : field
                Mapped field on the target space of the operator.
        """
        ## check whether self-inverse
        if(self.sym)and(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.target,True)
        ## apply operator
        x_ = self._inverse_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.domain,True)


    def adjoint_inverse_times(self,x,**kwargs):
        """
            Applies the inverse adjoint operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OAIx : field
                Mapped field on the domain of the operator.

        """
        ## check whether self-adjoint
        if(self.sym):
            return self.inverse_times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.domain,True)
        ## apply operator
        x_ = self._adjoint_inverse_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,True)

    def inverse_adjoint_times(self,x,**kwargs):
        """
            Applies the adjoint inverse operator to a given object.

            Parameters
            ----------
            x : {scalar, ndarray, field}
                Scalars are interpreted as constant arrays, and an array will
                be interpreted as a field on the target space of the operator.

            Returns
            -------
            OIAx : field
                Mapped field on the domain of the operator.

        """
        ## check whether self-adjoint
        if(self.sym):
            return self.inverse_times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.times(x,**kwargs)

        ## prepare
        x_ = self._briefing(x,self.domain,True)
        ## apply operator
        x_ = self._inverse_adjoint_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False,loop=False):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations
        """
        if(domain is None):
            domain = self.domain
        return trace_probing(self,function=self.times,domain=domain,target=target,random=random,ncpu=ncpu,nrun=nrun,nper=nper,var=var)(loop=loop)

    def inverse_tr(self,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False,loop=False):
        """
            Computes the trace of the inverse operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: Nonoe)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the inverse operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations
        """
        if(domain is None):
            domain = self.target
        return trace_probing(self,function=self.inverse_times,domain=domain,target=target,random=random,ncpu=ncpu,nrun=nrun,nper=nper,var=var)(loop=loop)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self,bare=False,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False,save=False,path="tmp",prefix="",loop=False):
        """
            Computes the diagonal of the operator via probing.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.domain
        diag = diagonal_probing(self,function=self.times,domain=domain,target=target,random=random,ncpu=ncpu,nrun=nrun,nper=nper,var=var,save=save,path=path,prefix=prefix)(loop=loop)
        ## weight if ...
        if(not domain.discrete)and(bare):
            if(isinstance(diag,tuple)): ## diag == (diag,variance)
                return domain.calc_weight(diag[0],power=-1),domain.calc_weight(diag[1],power=-1)
            else:
                return domain.calc_weight(diag,power=-1)
        else:
            return diag

    def inverse_diag(self,bare=False,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False,save=False,path="tmp",prefix="",loop=False):
        """
            Computes the diagonal of the inverse operator via probing.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The diagonal of the inverse matrix
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        diag = diagonal_probing(self,function=self.inverse_times,domain=domain,target=target,random=random,ncpu=ncpu,nrun=nrun,nper=nper,var=var,save=save,path=path,prefix=prefix)(loop=loop)
        ## weight if ...
        if(not domain.discrete)and(bare):
            if(isinstance(diag,tuple)): ## diag == (diag,variance)
                return domain.calc_weight(diag[0],power=-1),domain.calc_weight(diag[1],power=-1)
            else:
                return domain.calc_weight(diag,power=-1)
        else:
            return diag

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the operator

            Returns
            -------
            det : float
                The determinant
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'det'."))

    def inverse_det(self):
        """
            Computes the determinant of the inverse operator

            Returns
            -------
            det : float
                The determinant
        """
        raise NotImplementedError(about._errors.cstring("ERROR: no generic instance method 'inverse_det'."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def hat(self,bare=False,domain=None,target=None,**kwargs):
        """
            Translates the operator's diagonal into a field

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            x : field
                The matrix diagonal as a field living on the operator
                domain space

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.domain
        return field(domain,val=self.diag(bare=bare,domain=domain,target=target,var=False,**kwargs),target=target)

    def inverse_hat(self,bare=False,domain=None,target=None,**kwargs):
        """
            Translates the inverse operator's diagonal into a field

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: 8)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            x : field
                The matrix diagonal as a field living on the operator
                domain space

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        return field(domain,val=self.inverse_diag(bare=bare,domain=domain,target=target,var=False,**kwargs),target=target)

    def hathat(self,domain=None,**kwargs):
        """
            Translates the operator's diagonal into a diagonal operator

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            D : diagonal_operator
                The matrix diagonal as an operator

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.domain
        return diagonal_operator(domain=domain,diag=self.diag(bare=False,domain=domain,var=False,**kwargs),bare=False)

    def inverse_hathat(self,domain=None,**kwargs):
        """
            Translates the inverse operator's diagonal into a diagonal
            operator

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            D : diagonal_operator
                The diagonal of the inverse matrix as an operator

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None):
            domain = self.target
        return diagonal_operator(domain=domain,diag=self.inverse_diag(bare=False,domain=domain,var=False,**kwargs),bare=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.operator>"

##=============================================================================



##-----------------------------------------------------------------------------

class diagonal_operator(operator):
    """
        ..           __   __                                                     __
        ..         /  / /__/                                                   /  /
        ..    ____/  /  __   ____ __   ____ __   ______    __ ___    ____ __  /  /
        ..  /   _   / /  / /   _   / /   _   / /   _   | /   _   | /   _   / /  /
        .. /  /_/  / /  / /  /_/  / /  /_/  / /  /_/  / /  / /  / /  /_/  / /  /_
        .. \______| /__/  \______|  \___   /  \______/ /__/ /__/  \______|  \___/  operator class
        ..                         /______/

        NIFTY subclass for diagonal operators

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If no domain is given
            then the diag parameter *must* be a field and the domain
            of that field is assumed. (default: None)
        diag : {scalar, ndarray, field}
            The diagonal entries of the operator. For a scalar, a constant
            diagonal is defined having the value provided. If no domain
            is given, diag must be a field. (default: 1)
        bare : bool, *optional*
            whether the diagonal entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: False)

        Notes
        -----
        The ambiguity of `bare` or non-bare diagonal entries is based
        on the choice of a matrix representation of the operator in
        question. The naive choice of absorbing the volume weights
        into the matrix leads to a matrix-vector calculus with the
        non-bare entries which seems intuitive, though. The choice of
        keeping matrix entries and volume weights separate deals with the
        bare entries that allow for correct interpretation of the matrix
        entries; e.g., as variance in case of an covariance operator.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            A field containing the diagonal entries of the matrix.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
    """
    def __init__(self,domain=None,diag=1,bare=False):
        """
            Sets the standard operator properties and `values`.

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If no domain is given
                then the diag parameter *must* be a field and the domain
                of that field is assumed. (default: None)
            diag : {scalar, ndarray, field}, *optional*
                The diagonal entries of the operator. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)

            Returns
            -------
            None

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)and(isinstance(diag,field)):
            domain = diag.domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        diag = self.domain.enforce_values(diag,extend=True)
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            diag = self.domain.calc_weight(diag,power=1)
        ## check complexity
        if(np.all(np.imag(diag)==0)):
            self.val = np.real(diag)
            self.sym = True
        else:
            self.val = diag
            about.infos.cprint("INFO: non-self-adjoint complex diagonal operator.")
            self.sym = False

        ## check whether identity
        if(np.all(diag==1)):
            self.uni = True
        else:
            self.uni = False

        self.imp = True ## correctly implemented for efficiency
        self.target = self.domain

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_diag(self,newdiag,bare=False):
        """
            Sets the diagonal of the diagonal operator

            Parameters
            ----------
            newdiag : {scalar, ndarray, field}
                The new diagonal entries of the operator. For a scalar, a
                constant diagonal is defined having the value provided. If
                no domain is given, diag must be a field.

            bare : bool, *optional*
                whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)

            Returns
            -------
            None
        """
        newdiag = self.domain.enforce_values(newdiag,extend=True)
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            newdiag = self.domain.calc_weight(newdiag,power=1)
        ## check complexity
        if(np.all(np.imag(newdiag)==0)):
            self.val = np.real(newdiag)
            self.sym = True
        else:
            self.val = newdiag
            about.infos.cprint("INFO: non-self-adjoint complex diagonal operator.")
            self.sym = False

        ## check whether identity
        if(np.all(newdiag==1)):
            self.uni = True
        else:
            self.uni = False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        x_ = field(self.target,val=None,target=x.target)
        x_.val = x.val*self.val ## bypasses self.domain.enforce_values
        return x_

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        x_ = field(self.domain,val=None,target=x.target)
        x_.val = x.val*np.conjugate(self.val) ## bypasses self.domain.enforce_values
        return x_

    def _inverse_multiply(self,x,**kwargs): ## > applies the inverse operator to a given field
        if(np.any(self.val==0)):
            raise AttributeError(about._errors.cstring("ERROR: singular operator."))
        else:
            x_ = field(self.domain,val=None,target=x.target)
            x_.val = x.val/self.val ## bypasses self.domain.enforce_values
            return x_

    def _adjoint_inverse_multiply(self,x,**kwargs): ## > applies the inverse adjoint operator to a given field
        if(np.any(self.val==0)):
            raise AttributeError(about._errors.cstring("ERROR: singular operator."))
        else:
            x_ = field(self.target,val=None,target=x.target)
            x_.val = x.val/np.conjugate(self.val) ## bypasses self.domain.enforce_values
            return x_

    def _inverse_adjoint_multiply(self,x,**kwargs): ## > applies the adjoint inverse operator to a given field
        if(np.any(self.val==0)):
            raise AttributeError(about._errors.cstring("ERROR: singular operator."))
        else:
            x_ = field(self.target,val=None,target=x.target)
            x_.val = x.val*np.conjugate(1/self.val) ## bypasses self.domain.enforce_values
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,**kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if(domain is None)or(domain==self.domain):
            if(self.uni): ## identity
                return (self.domain.datatype(self.domain.dof())).real
            elif(self.domain.dim(split=False)<self.domain.dof()): ## hidden degrees of freedom
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),self.val) ## discrete inner product
            else:
                return np.sum(self.val,axis=None,dtype=None,out=None)
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## check degrees of freedom
                if(self.domain.dof()>domain.dof()):
                    about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(self.domain.dof())+" / "+str(domain.dof())+" ).")
                return (domain.datatype(domain.dof())).real
            else:
                return super(diagonal_operator,self).tr(domain=domain,**kwargs) ## probing

    def inverse_tr(self,domain=None,**kwargs):
        """
            Computes the trace of the inverse operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the inverse operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if(np.any(self.val==0)):
            raise AttributeError(about._errors.cstring("ERROR: singular operator."))

        if(domain is None)or(domain==self.target):
            if(self.uni): ## identity
                return np.real(self.domain.datatype(self.domain.dof()))
            elif(self.domain.dim(split=False)<self.domain.dof()): ## hidden degrees of freedom
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),1/self.val) ## discrete inner product
            else:
                return np.sum(1/self.val,axis=None,dtype=None,out=None)
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## check degrees of freedom
                if(self.domain.dof()>domain.dof()):
                    about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(self.domain.dof())+" / "+str(domain.dof())+" ).")
                return np.real(domain.datatype(domain.dof()))
            else:
                return super(diagonal_operator,self).inverse_tr(domain=domain,**kwargs) ## probing

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self,bare=False,domain=None,**kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)or(domain==self.domain):
            ## weight if ...
            if(not self.domain.discrete)and(bare):
                diag = self.domain.calc_weight(self.val,power=-1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
                return diag
            else:
                return self.val
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## weight if ...
                if(not domain.discrete)and(bare):
                    return np.real(domain.calc_weight(domain.enforce_values(1,extend=True),power=-1))
                else:
                    return np.real(domain.enforce_values(1,extend=True))
            else:
                return super(diagonal_operator,self).diag(bare=bare,domain=domain,**kwargs) ## probing

    def inverse_diag(self,bare=False,domain=None,**kwargs):
        """
            Computes the diagonal of the inverse operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The diagonal of the inverse matrix
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            See Also
            --------
            probing : The class used to perform probing operations

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)or(domain==self.target):
            ## weight if ...
            if(not self.domain.discrete)and(bare):
                diag = self.domain.calc_weight(1/self.val,power=-1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
                return diag
            else:
                return 1/self.val
        else:
            if(self.uni): ## identity
                if(not isinstance(domain,space)):
                    raise TypeError(about._errors.cstring("ERROR: invalid input."))
                ## weight if ...
                if(not domain.discrete)and(bare):
                    return np.real(domain.calc_weight(domain.enforce_values(1,extend=True),power=-1))
                else:
                    return np.real(domain.enforce_values(1,extend=True))
            else:
                return super(diagonal_operator,self).inverse_diag(bare=bare,domain=domain,**kwargs) ## probing

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_random_field(self,domain=None,target=None,**kwargs):
        """
            Generates a Gaussian random field with variance equal to the
            diagonal.

            Parameters
            ----------
            domain : space, *optional*
                space wherein the field lives (default: None, indicates
                to use self.domain)
            target : space, *optional*
                space wherein the transform of the field lives
                (default: None, indicates to use target of domain)

            Returns
            -------
            x : field
                Random field.

        """
        if(len(kwargs)):  ## TODO: remove **kwargs in future version
            about.warnings.cprint("WARNING: deprecated keyword(s).")
        ## weight if ...
        if(not self.domain.discrete):
            diag = self.domain.calc_weight(self.val,power=-1)
            ## check complexity
            if(np.all(np.imag(diag)==0)):
                diag = np.real(diag)
        else:
            diag = self.val

        if(domain is None)or(domain==self.domain):
            return field(self.domain,val=None,target=target,random="gau",var=self.diag(bare=True,domain=self.domain))
        else:
            return field(self.domain,val=None,target=domain,random="gau",var=self.diag(bare=True,domain=self.domain)).transform(target=domain,overwrite=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.diagonal_operator>"

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

def identity(domain):
    """
        Returns an identity operator.

        The identity operator is represented by a `diagonal_operator` instance,
        which is applicable to a field-like object; i.e., a scalar, list,
        array or field. (The identity operator is unrelated to PYTHON's
        built-in function :py:func:`id`.)

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.

        Returns
        -------
        id : diagonal_operator
            The identity operator as a `diagonal_operator` instance.

        See Also
        --------
        diagonal_operator

        Examples
        --------
        >>> I = identity(rg_space(8,dist=0.2))
        >>> I.diag()
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        >>> I.diag(bare=True)
        array([ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.])
        >>> I.tr()
        8.0
        >>> I(3)
        <nifty.field>
        >>> I(3).val
        array([ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.])
        >>> I(np.arange(8))[:]
        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> f = I.get_random_field()
        >>> print(I(f) - f)
        nifty.field instance
        - domain      = <nifty.rg_space>
        - val         = [...]
          - min.,max. = [0.0, 0.0]
          - med.,mean = [0.0, 0.0]
        - target      = <nifty.rg_space>
        >>> I.times(f) ## equal to I(f)
        <nifty.field>
        >>> I.inverse_times(f)
        <nifty.field>

    """
    return diagonal_operator(domain=domain,diag=1,bare=False)

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------

class power_operator(diagonal_operator):
    """
        ..      ______    ______   __     __   _______   _____
        ..    /   _   | /   _   | |  |/\/  / /   __  / /   __/
        ..   /  /_/  / /  /_/  /  |       / /  /____/ /  /
        ..  /   ____/  \______/   |__/\__/  \______/ /__/     operator class
        .. /__/

        NIFTY subclass for (signal-covariance-type) diagonal operators containing a power spectrum

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If no domain is given
            then the diag parameter *must* be a field and the domain
            of that field is assumed. (default: None)
        spec : {scalar, ndarray, field}
            The power spectrum. For a scalar, a constant power
            spectrum is defined having the value provided. If no domain
            is given, diag must be a field. (default: 1)
        bare : bool, *optional*
            whether the entries are `bare` or not
            (mandatory for the correct incorporation of volume weights)
            (default: False)
        pindex : ndarray, *optional*
            indexing array, obtainable from domain.get_power_index
            (default: None)

        Notes
        -----
        The ambiguity of `bare` or non-bare diagonal entries is based
        on the choice of a matrix representation of the operator in
        question. The naive choice of absorbing the volume weights
        into the matrix leads to a matrix-vector calculus with the
        non-bare entries which seems intuitive, though. The choice of
        keeping matrix entries and volume weights separate deals with the
        bare entries that allow for correct interpretation of the matrix
        entries; e.g., as variance in case of an covariance operator.

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            A field containing the diagonal entries of the matrix.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives

    """
    def __init__(self,domain,spec=1,bare=True,pindex=None):
        """
            Sets the diagonal operator's standard properties

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If no domain is given
                then the diag parameter *must* be a field and the domain
                of that field is assumed. (default: None)
            spec : {scalar, ndarray, field}
                The power spectrum. For a scalar, a constant power
                spectrum is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_index
                (default: None)

            Returns
            -------
            None
        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        ## check power_index
        if(pindex is None):
            try:
                pindex = self.domain.get_power_index(irreducible=False)
            except(AttributeError):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
        else:
            pindex = np.array(pindex,dtype=np.int)
            if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))
        ## set diagonal
        try:
            diag = self.domain.enforce_power(spec,size=np.max(pindex,axis=None,out=None)+1)[pindex]
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            self.val = np.real(self.domain.calc_weight(diag,power=1))
        else:
            self.val = diag

        self.sym = True

        ## check whether identity
        if(np.all(spec==1)):
            self.uni = True
        else:
            self.uni = False

        self.imp = True
        self.target = self.domain

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_power(self,newspec,bare=None,pindex=None):
        """
            Sets the power spectrum of the diagonal operator

            Parameters
            ----------
            newspec : {scalar, ndarray, field}
                The entries of the operator. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_index
                (default: None)

            Returns
            -------
            None
        """
        if(bare is None):
            about.warnings.cprint("WARNING: bare keyword set to default.")
            bare = True

        ## check power_index
        if(pindex is None):
            try:
                pindex = self.domain.get_power_index(irreducible=False)
            except(AttributeError):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
        else:
            pindex = np.array(pindex,dtype=np.int)
            if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))
        ## set diagonal
        try:
            diag = self.domain.enforce_power(newspec,size=np.max(pindex,axis=None,out=None)+1)[pindex]
        except(AttributeError):
            raise ValueError(about._errors.cstring("ERROR: invalid input."))
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            self.val = np.real(self.domain.calc_weight(diag,power=1))
        else:
            self.val = diag

        ## check whether identity
        if(np.all(newspec==1)):
            self.uni = True
        else:
            self.uni = False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_power(self,bare=True,pundex=None,pindex=None,**kwargs):
        """
            Computes the power spectrum

            Parameters
            ----------
            bare : bool, *optional*
                whether the entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            pundex : ndarray, *optional*
                unindexing array, obtainable from domain.get_power_undex
                (default: None)
            pindex : ndarray, *optional*
                indexing array, obtainable from domain.get_power_index
                (default: None)

            Returns
            -------
            spec : ndarray
                The power spectrum
        """
        if(len(kwargs)):  ## TODO: remove **kwargs in future version
            about.warnings.cprint("WARNING: deprecated keyword(s).")
        ## weight if ...
        if(not self.domain.discrete)and(bare):
            diag = np.real(self.domain.calc_weight(self.val,power=-1))
        else:
            diag = self.val

        if(pundex is None):
            pundex = self.domain.get_power_undex(pindex=pindex)
        return diag[pundex]

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_projection_operator(self,pindex=None):
        """
            Generates a spectral projection operator

            Parameters
            ----------
            pindex : ndarray
                indexing array obtainable from domain.get_power_index
                (default: None)

            Returns
            -------
            P : projection_operator
        """
        ## check power_index
        if(pindex is None):
            pindex = self.domain.get_power_index(irreducible=False)
        else:
            pindex = np.array(pindex,dtype=np.int)
            if(not np.all(np.array(np.shape(pindex))==self.domain.dim(split=True))):
                raise ValueError(about._errors.cstring("ERROR: shape mismatch ( "+str(np.array(np.shape(pindex)))+" <> "+str(self.domain.dim(split=True))+" )."))

        return projection_operator(self.domain,assign=pindex)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.power_operator>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class projection_operator(operator):
    """
        ..                                     __                       __     __
        ..                                   /__/                     /  /_  /__/
        ..      ______    _____   ______     __   _______   _______  /   _/  __   ______    __ ___
        ..    /   _   | /   __/ /   _   |  /  / /   __  / /   ____/ /  /   /  / /   _   | /   _   |
        ..   /  /_/  / /  /    /  /_/  /  /  / /  /____/ /  /____  /  /_  /  / /  /_/  / /  / /  /
        ..  /   ____/ /__/     \______/  /  /  \______/  \______/  \___/ /__/  \______/ /__/ /__/  operator class
        .. /__/                        /___/

        NIFTY subclass for projection operators

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        assign : ndarray, *optional*
            Assignments of domain items to projection bands. An array
            of integers, negative integers are associated with the
            nullspace of the projection. (default: None)

        Examples
        --------
        >>> space = point_space(3)
        >>> P = projection_operator(space, assign=[0, 1, 0])
        >>> P.bands()
        2
        >>> P([1, 2, 3], band=0) # equal to P.times(field(space,val=[1, 2, 3]))
        <nifty.field>
        >>> P([1, 2, 3], band=0).domain
        <nifty.point_space>
        >>> P([1, 2, 3], band=0).val # projection on band 0 (items 0 and 2)
        array([ 1.,  0.,  3.])
        >>> P([1, 2, 3], band=1).val # projection on band 1 (item 1)
        array([ 0.,  2.,  0.])
        >>> P([1, 2, 3])
        <nifty.field>
        >>> P([1, 2, 3]).domain
        <nifty.nested_space>
        >>> P([1, 2, 3]).val # projection on all bands
        array([[ 1.,  0.,  3.],
               [ 0.,  2.,  0.]])

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        ind : ndarray
            Assignments of domain items to projection bands.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives

    """
    def __init__(self,domain,assign=None):
        """
            Sets the standard operator properties and `indexing`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            assign : ndarray, *optional*
                Assignments of domain items to projection bands. An array
                of integers, negative integers are associated with the
                nullspace of the projection. (default: None)

            Returns
            -------
            None
        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        ## check assignment(s)
        if(assign is None):
            assign = np.arange(self.domain.dim(split=False),dtype=np.int).reshape(self.domain.dim(split=True),order='C')
        else:
            assign = self.domain.enforce_shape(assign).astype(np.int)
        assign = np.maximum(-1,assign) ## condensing all negative integers
        ## build indexing
        ind = [[]]*len(set(assign.flatten(order='C')))
        if(np.size(self.domain.dim(split=True))==1):
            for ii in range(self.domain.dim(split=True)):
                if(assign[ii]!=-1):
                    ind[assign[ii]] = ind[assign[ii]]+[ii] ## adding index
        else:
            for ii in np.ndindex(tuple(self.domain.dim(split=True))):
                if(assign[ii]!=-1):
                    ind[assign[ii]] = ind[assign[ii]]+[ii] ## adding index
        self.ind = []
        for ii in ind:
            if(len(ii)>0):
                self.ind = self.ind+[np.array(ii,dtype=np.int).T]
            else:
                about.warnings.cprint("WARNING: empty projection band removed.")

        self.sym = True
#        about.infos.cprint("INFO: pseudo unitary projection operator.")
        self.uni = False
        self.imp = True

        self.target = nested_space([point_space(len(self.ind),datatype=self.domain.datatype),self.domain])

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def bands(self):
        """
            Computes the number of projection bands

            Returns
            -------
            bands : int
                The number of projection bands
        """
        return len(self.ind)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def rho(self):
        """
            Computes the number of degrees of freedom per projection band.

            Returns
            -------
            rho : ndarray
                The number of degrees of freedom per projection band.
        """
        rho = np.empty(len(self.ind),dtype=np.int,order='C')
        if(self.domain.dim(split=False)==self.domain.dof()): ## no hidden degrees of freedom
            for ii in range(len(self.ind)):
                rho[ii] = np.size(self.ind[ii].T,axis=0)
        else: ## hidden degrees of freedom
            mof = np.round(np.real(self.domain.calc_weight(self.domain.get_meta_volume(total=False),power=-1)),decimals=0,out=None).astype(np.int) ## meta degrees of freedom
            for ii in range(len(self.ind)):
                rho[ii] = np.sum(mof[self.ind[ii].tolist()],axis=None,dtype=np.int,out=None)
        return rho

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,band=None,bandsup=None,**kwargs):
        """
            Applies the operator to a given field.

            Parameters
            ----------
            x : valid field
            band : int, *optional*
                Projection band whereon to project. (default: None)
            bandsup: list of integers, *optional*
                List of projection bands whereon to project and which to sum
                up. The `band` keyword is prefered over `bandsup`.
                (default: None)

            Returns
            -------
            Px : field
                projected field(!)
        """
        if(band is not None):
            band = int(band)
            if(band>self.bands()-1)or(band<0):
                raise TypeError(about._errors.cstring("ERROR: invalid band."))
            Px = field(self.domain,val=None,target=x.target)
            Px[self.ind[band].tolist()] += x[self.ind[band].tolist()]
            return Px

        elif(bandsup is not None):
            bandsup = [int(bb) for bb in bandsup if -1<bb<self.bands()]
            if(bandsup==[]):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            Px = field(self.domain,val=None,target=x.target)
            for bb in bandsup:
                Px[self.ind[bb].tolist()] += x[self.ind[bb].tolist()]
            return Px

        else:
            Px = field(self.target,val=None,target=nested_space([point_space(len(self.ind),datatype=x.target.datatype),x.target]))
            for bb in range(self.bands()):
                Px[bb][self.ind[bb].tolist()] += x[self.ind[bb].tolist()]
            return Px

    def _inverse_multiply(self,x,**kwargs):
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _debriefing(self,x,x_,target,inverse): ## > evaluates x and x_ after `multiply`
        if(x_ is None):
            return None
        else:
            ## weight if ...
            if(not self.imp)and(not target.discrete)and(inverse):
                x_ = x_.weight(power=-1,overwrite=False)
            ## inspect x
            if(isinstance(x,field)):
                if(x_.domain==self.target):
                    ## repair ...
                    if(x_.domain.nest[-1]!=x.domain):
                        x_ = x_.transform(target=nested_space([point_space(len(self.ind),datatype=x.domain.datatype),x.domain]),overwrite=False) ## ... domain
                    if(x_.target.nest[-1]!=x.target):
                        x_.set_target(newtarget=nested_space([point_space(len(self.ind),datatype=x.target.datatype),x.target])) ## ... codomain
                else:
                    ## repair ...
                    if(x_.domain!=x.domain):
                        x_ = x_.transform(target=x.domain,overwrite=False) ## ... domain
                    if(x_.target!=x.target):
                        x_.set_target(newtarget=x.target) ## ... codomain
            return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def pseudo_tr(self,x,**kwargs):
        """
            Computes the pseudo trace of a given object for all projection bands

            Parameters
            ----------
            x : {field, operator}
                The object whose pseudo-trace is to be computed. If the input is
                a field, the pseudo trace equals the trace of
                the projection operator mutliplied by a vector-vector operator
                corresponding to the input field. This is also equal to the
                pseudo inner product of the field with projected field itself.
                If the input is a operator, the pseudo trace equals the trace of
                the projection operator multiplied by the input operator.
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Pseudo trace for all projection bands
        """
        if(isinstance(x,operator)):
            ## compute non-bare diagonal of the operator x
            x = x.diag(bare=False,domain=self.domain,target=x.domain,var=False,**kwargs)

        elif(isinstance(x,field)):
            ## check domain
            if(self.domain==x.domain):
                x = x.val
            else:
                x = x.transform(target=self.domain,overwrite=False).val
            ## compute non-bare diagonal of the vector-vector operator corresponding to the field x
            x = x*np.conjugate(x)
            ## weight
            if(not self.domain.discrete):
                x = self.domain.calc_weight(x,power=1)

        else:
            raise TypeError(about._errors.cstring("ERROR: invalid input."))

        x = np.real(x)
        if(not self.domain.dim(split=False)==self.domain.dof()):
            x *= np.round(np.real(self.domain.calc_weight(self.domain.get_meta_volume(total=False),power=-1)),decimals=0,out=None).astype(np.int) ## meta degrees of freedom

        tr = np.empty(self.bands(),dtype=x.dtype,order='C')
        for bb in range(self.bands()):
            tr[bb] = np.sum(x[self.ind[bb].tolist()],axis=None,dtype=None,out=None)
        return tr

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.projection_operator>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class vecvec_operator(operator):
    """
        ..                                                                 __
        ..                                                             __/  /__
        ..  __   __   _______   _______  __   __   _______   _______ /__    __/
        .. |  |/  / /   __  / /   ____/ |  |/  / /   __  / /   ____/   /__/
        .. |     / /  /____/ /  /____   |     / /  /____/ /  /____
        .. |____/  \______/  \______/   |____/  \______/  \______/            operator class

        NIFTY subclass for vector-vector operators

        Parameters
        ----------
        domain : space, *optional*
            The space wherein valid arguments live. If none is given, the
            space of the field given in val is used. (default: None)
        val : {scalar, ndarray, field}, *optional*
            The field from which to construct the operator. For a scalar, a constant
            field is defined having the value provided. If no domain
            is given, val must be a field. (default: 1)

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        val : ndarray
            The field from which the operator is derived.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives.
    """
    def __init__(self,domain=None,val=1):
        """
            Sets the standard operator properties and `values`.

            Parameters
            ----------
            domain : space, *optional*
                The space wherein valid arguments live. If none is given, the
                space of the field given in val is used. (default: None)
            val : {scalar, ndarray, field}, *optional*
                The field from which to construct the operator. For a scalar, a constant
                field is defined having the value provided. If no domain
                is given, val must be a field. (default: 1)

            Returns
            -------
            None
        """
        if(domain is None)and(isinstance(val,field)):
            domain = val.domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain
        self.val = self.domain.enforce_values(val,extend=True)
        self.sym = True
        self.uni = False
        self.imp = False
        self.target = self.domain

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_val(self,newval):
        """
            Sets the field values of the operator

            Parameters
            ----------
            newval : {scalar, ndarray, field}
                The new field values. For a scalar, a constant
                diagonal is defined having the value provided. If no domain
                is given, diag must be a field. (default: 1)

            Returns
            -------
            None
        """
        self.val = self.domain.enforce_values(newval,extend=True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        x_ = field(self.target,val=None,target=x.target)
        x_.val = self.val*self.domain.calc_dot(self.val,x.val) ## bypasses self.domain.enforce_values
        return x_

    def _inverse_multiply(self,x,**kwargs):
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,**kwargs):
        """
            Computes the trace of the operator

            Parameters
            ----------
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            tr : float
                Trace of the operator
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

        """
        if(domain is None)or(domain==self.domain):
            if(not self.domain.discrete):
                return self.domain.calc_dot(self.val,self.domain.calc_weight(self.val,power=1))
            else:
                return self.domain.calc_dot(self.val,self.val)
        else:
            return super(vecvec_operator,self).tr(domain=domain,**kwargs) ## probing

    def inverse_tr(self):
        """
        Inverse is ill-defined for this operator.
        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def diag(self,bare=False,domain=None,**kwargs):
        """
            Computes the diagonal of the operator.

            Parameters
            ----------
            bare : bool, *optional*
                Indicatese whether the diagonal entries are `bare` or not
                (mandatory for the correct incorporation of volume weights)
                (default: False)
            domain : space, *optional*
                space wherein the probes live (default: self.domain)
            target : space, *optional*
                space wherein the transform of the probes live
                (default: None, applies target of the domain)
            random : string, *optional*
                Specifies the pseudo random number generator. Valid
                options are "pm1" for a random vector of +/-1, or "gau"
                for a random vector with entries drawn from a Gaussian
                distribution with zero mean and unit variance.
                (default: "pm1")
            ncpu : int, *optional*
                number of used CPUs to use. (default: 2)
            nrun : int, *optional*
                total number of probes (default: 8)
            nper : int, *optional*
                number of tasks performed by one process (default: None)
            var : bool, *optional*
                Indicates whether to additionally return the probing variance
                or not (default: False).
            save : bool, *optional*
                whether all individual probing results are saved or not
                (default: False)
            path : string, *optional*
                path wherein the results are saved (default: "tmp")
            prefix : string, *optional*
                prefix for all saved files (default: "")
            loop : bool, *optional*
                Indicates whether or not to perform a loop i.e., to
                parallelise (default: False)

            Returns
            -------
            diag : ndarray
                The matrix diagonal
            delta : float, *optional*
                Probing variance of the trace. Returned if `var` is True in
                of probing case.

            Notes
            -----
            The ambiguity of `bare` or non-bare diagonal entries is based
            on the choice of a matrix representation of the operator in
            question. The naive choice of absorbing the volume weights
            into the matrix leads to a matrix-vector calculus with the
            non-bare entries which seems intuitive, though. The choice of
            keeping matrix entries and volume weights separate deals with the
            bare entries that allow for correct interpretation of the matrix
            entries; e.g., as variance in case of an covariance operator.

        """
        if(domain is None)or(domain==self.domain):
            diag = np.real(self.val*np.conjugate(self.val)) ## bare diagonal
            ## weight if ...
            if(not self.domain.discrete)and(not bare):
                return self.domain.calc_weight(diag,power=1)
            else:
                return diag
        else:
            return super(vecvec_operator,self).diag(bare=bare,domain=domain,**kwargs) ## probing

    def inverse_diag(self):
        """
        Inverse is ill-defined for this operator.
        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the operator

            Returns
            -------
            det : 0
                The determinant
        """
        return 0

    def inverse_det(self):
        """
        Inverse is ill-defined for this operator.
        """
        raise AttributeError(about._errors.cstring("ERROR: singular operator."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.vecvec_operator>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class response_operator(operator):
    """
        ..     _____   _______   _______   ______    ______    __ ___    _______   _______
        ..   /   __/ /   __  / /  _____/ /   _   | /   _   | /   _   | /  _____/ /   __  /
        ..  /  /    /  /____/ /_____  / /  /_/  / /  /_/  / /  / /  / /_____  / /  /____/
        .. /__/     \______/ /_______/ /   ____/  \______/ /__/ /__/ /_______/  \______/  operator class
                                    /__/

        NIFTY subclass for response operators (of a certain family)

        Any response operator handles Gaussian convolutions, itemwise masking,
        and selective mappings.

        Parameters
        ----------
        domain : space
            The space wherein valid arguments live.
        sigma : float, *optional*
            The standard deviation of the Gaussian kernel. Zero indicates
            no convolution. (default: 0)
        mask : {scalar, ndarray}, *optional*
            Masking values for arguments (default: 1)
        assign : {list, ndarray}, *optional*
            Assignments of codomain items to domain items. A list of
            indices/ index tuples or a one/ two-dimensional array.
            (default: None)
        den : bool, *optional*
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
            (default: False)
        target : space, *optional*
            The space wherein the operator output lives (default: domain)

        Attributes
        ----------
        domain : space
            The space wherein valid arguments live.
        sym : bool
            Indicates whether the operator is self-adjoint or not
        uni : bool
            Indicates whether the operator is unitary or not
        imp : bool
            Indicates whether the incorporation of volume weights in
            multiplications is already implemented in the `multiply`
            instance methods or not
        target : space
            The space wherein the operator output lives
        sigma : float
            The standard deviation of the Gaussian kernel. Zero indicates
            no convolution.
        mask : {scalar, ndarray}
            Masking values for arguments
        assign : {list, ndarray}
            Assignments of codomain items to domain items. A list of
            indices/ index tuples or a one/ two-dimensional array.
        den : bool
            Whether to consider the arguments as densities or not.
            Mandatory for the correct incorporation of volume weights.
    """
    def __init__(self,domain,sigma=0,mask=1,assign=None,den=False,target=None):
        """
            Sets the standard properties and `density`, `sigma`, `mask` and `assignment(s)`.

            Parameters
            ----------
            domain : space
                The space wherein valid arguments live.
            sigma : float, *optional*
                The standard deviation of the Gaussian kernel. Zero indicates
                no convolution. (default: 0)
            mask : {scalar, ndarray}, *optional*
                Masking values for arguments (default: 1)
            assign : {list, ndarray}, *optional*
                Assignments of codomain items to domain items. A list of
                indices/ index tuples or a one/ two-dimensional array.
                (default: None)
            den : bool, *optional*
                Whether to consider the arguments as densities or not.
                Mandatory for the correct incorporation of volume weights.
                (default: False)
            target : space, *optional*
                The space wherein the operator output lives (default: domain)

            Returns
            -------
            None
        """
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = domain

        self.sym = False
        self.uni = False
        self.imp = False
        self.den = bool(den)

        self.mask = self.domain.enforce_values(mask,extend=False)

        ## check sigma
        if(sigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        self.sigma = sigma

        ## check assignment(s)
        if(np.size(self.domain.dim(split=True))==1):
            if(assign is None):
                assign = np.arange(self.domain.dim(split=False),dtype=np.int)
            elif(np.isscalar(assign)):
                assign = np.array([assign],dtype=np.int)
                if(assign[0]>=self.domain.dim(split=False))or(assign[0]<-self.domain.dim(split=False)):
                    raise IndexError(about._errors.cstring("ERROR: invalid bounds."))
            else:
                assign = np.array(assign,dtype=np.int)
                if(np.ndim(assign)!=1):
                    raise ValueError(about._errors.cstring("ERROR: invalid input."))
                elif(np.any(assign>=self.domain.dim(split=False)))or(np.any(assign<-self.domain.dim(split=False))):
                    raise IndexError(about._errors.cstring("ERROR: invalid bounds."))
        else:
            if(assign is None):
                assign = np.array([ii for ii in np.ndindex(tuple(self.domain.dim(split=True)))],dtype=np.int)
            elif(np.isscalar(assign)):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                assign = np.array(assign,dtype=np.int)
                if(np.ndim(assign)!=2)or(np.size(assign,axis=1)!=np.size(self.domain.dim(split=True))):
                    raise ValueError(about._errors.cstring("ERROR: invalid input."))
                for ii in range(np.size(assign,axis=1)):
                    if(np.any(assign[:,ii]>=self.domain.dim(split=True)[ii]))or(np.any(assign[:,ii]<-self.domain.dim(split=True)[ii])):
                        raise IndexError(about._errors.cstring("ERROR: invalid bounds."))
        self.assign = assign.T ## transpose

        if(target is None):
            ## set target
            target = point_space(np.size(assign,axis=0),datatype=self.domain.datatype)
        else:
            ## check target
            if(not isinstance(target,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            elif(not target.discrete):
                raise ValueError(about._errors.cstring("ERROR: continuous codomain.")) ## discrete(!)
            elif(np.size(target.dim(split=True))!=1):
                raise ValueError(about._errors.cstring("ERROR: structured codomain.")) ## unstructured(!)
            elif(np.size(assign,axis=0)!=target.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(assign,axis=0))+" <> "+str(target.dim(split=False))+" )."))
        self.target = target

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_sigma(self,newsigma):
        """
            Sets the standard deviation of the response operator, indicating
            the amount of convolution.

            Parameters
            ----------
            sigma : float
                The standard deviation of the Gaussian kernel. Zero indicates
                no convolution.

            Returns
            -------
            None
        """
        ## check sigma
        if(newsigma<0):
            raise ValueError(about._errors.cstring("ERROR: invalid sigma."))
        self.sigma = newsigma

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_mask(self,newmask):
        """
            Sets the masking values of the response operator

            Parameters
            ----------
            newmask : {scalar, ndarray}
                masking values for arguments

            Returns
            -------
            None
        """
        self.mask = self.domain.enforce_values(newmask,extend=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the operator to a given field
        ## weight
        if(self.domain.discrete)and(self.den):
            x_ = self.domain.calc_weight(x.val,power=1)
        elif(not self.domain.discrete)and(not self.den):
            x_ = self.domain.calc_weight(x.val,power=-1)
        else:
            x_ = x.val
        ## smooth
        x_ = self.domain.calc_smooth(x_,sigma=self.sigma)
        ## mask
        x_ = self.mask*x_
        ## assign
        return x_[self.assign.tolist()]

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        x_ = np.zeros(self.domain.dim(split=True),dtype=self.domain.datatype,order='C')
        ## assign (transposed)
        x_[self.assign.tolist()] += x.val.flatten(order='C')
        ## mask
        x_ = self.mask*x_
        ## smooth
        x_ = self.domain.calc_smooth(x_,sigma=self.sigma)
        ## weight
        if(self.domain.discrete)and(self.den):
            x_ = self.domain.calc_weight(x_,power=1)
        elif(not self.domain.discrete)and(not self.den):
            x_ = self.domain.calc_weight(x_,power=-1)
        return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.response_operator>"

##-----------------------------------------------------------------------------

## IDEA: explicit_operator





##=============================================================================

class probing(object):
    """
        ..                                    __        __
        ..                                  /  /      /__/
        ..      ______    _____   ______   /  /___    __   __ ___    ____ __
        ..    /   _   | /   __/ /   _   | /   _   | /  / /   _   | /   _   /
        ..   /  /_/  / /  /    /  /_/  / /  /_/  / /  / /  / /  / /  /_/  /
        ..  /   ____/ /__/     \______/  \______/ /__/ /__/ /__/  \____  /  class
        .. /__/                                                  /______/

        NIFTY class for probing (using multiprocessing)

        This is the base NIFTY probing class from which other probing classes
        (e.g. diagonal probing) are derived.

        When called, a probing class instance evaluates an operator or a
        function using random fields, whose components are random variables
        with mean 0 and variance 1. When an instance is called it returns the
        mean value of f(probe), where probe is a random field with mean 0 and
        variance 1. The mean is calculated as 1/N Sum[ f(probe_i) ].

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used from parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
            set to `ncpu**2`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)


        See Also
        --------
        diagonal_probing : A probing class to get the diagonal of an operator
        trace_probing : A probing class to get the trace of an operator


        Attributes
        ----------
        function : function
            the function, that is applied to the probes
        domain : space
            the space, where the probes live in
        target : space
            the codomain of `domain`
        random : string
            the random number generator used to create the probes
            (either "pm1" or "gau")
        ncpu : int
            the number of cpus used for probing
        nrun : int
            the number of probes to be evaluated, when the instance is called
        nper : int
            number of probes, that will be evaluated by one worker
        var : bool
            whether the variance will be additionally returned, when the
            instance is called

    """
    def __init__(self,op=None,function=None,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False):
        """
        initializes a probing instance

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used from parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
            set to `ncpu**2`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)

        """
        if(op is None):
            ## check whether callable
            if(function is None)or(not hasattr(function,"__call__")):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            ## check given domain
            if(domain is None)or(not isinstance(domain,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
        else:
            if(not isinstance(op,operator)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            ## check whether callable
            if(function is None)or(not hasattr(function,"__call__")):
                function = op.times
            elif(op==function):
                function = op.times
            ## check whether correctly bound
            if(op!=function.im_self):
                raise NameError(about._errors.cstring("ERROR: invalid input."))
            ## check given domain
            if(domain is None)or(not isinstance(domain,space)):
                if(function in [op.inverse_times,op.adjoint_times]):
                    domain = op.target
                else:
                    domain = op.domain
            else:
                if(function in [op.inverse_times,op.adjoint_times]):
                    op.target.check_codomain(domain) ## a bit pointless
                else:
                    op.domain.check_codomain(domain) ## a bit pointless

        self.function = function
        self.domain = domain

        if(target is None):
            target = domain.get_codomain()
        ## check codomain
        self.domain.check_codomain(target) ## a bit pointless
        self.target = target

        if(random not in ["pm1","gau"]):
            raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(random)+"'."))
        self.random = random

        self.ncpu = int(max(1,ncpu))
        self.nrun = int(max(self.ncpu**2,nrun))
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1,min(self.nrun//self.ncpu,nper)))

        self.var = bool(var)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def configure(self,**kwargs):
        """
            changes the attributes of the instance

            Parameters
            ----------
            random : string, *optional*
                the random number generator used to create the probes (default: "pm1")
            ncpu : int, *optional*
                the number of cpus to be used for parallel probing. (default: 2)
            nrun : int, *optional*
                the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
                set to `ncpu**2`. (default: 8)
            nper : int, *optional*
                number of probes, that will be evaluated by one worker (default: 8)
            var : bool, *optional*
                whether the variance will be additionally returned (default: False)

        """
        if(kwargs.has_key("random")):
            if(kwargs.get("random") not in ["pm1","gau"]):
                raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(kwargs.get("random"))+"'."))
            self.random = random

        if(kwargs.has_key("ncpu")):
            self.ncpu = int(max(1,kwargs.get("ncpu")))
        if(kwargs.has_key("nrun")):
            self.nrun = int(max(self.ncpu**2,kwargs.get("nrun")))
        if(kwargs.has_key("nper")):
            if(kwargs.get("nper") is None):
                self.nper = None
            else:
                self.nper = int(max(1,min(self.nrun//self.ncpu,kwargs.get("nper"))))

        if(kwargs.has_key("var")):
            self.var = bool(kwargs.get("var"))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def gen_probe(self):
        """
            Generates a single probe

            Returns
            -------
            probe : field
                a random field living in `domain` with mean 0 and variance 1 in
                each component

        """
        return field(self.domain,val=None,target=self.target,random=self.random)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self,idnum,probe):
        """
            Computes a single probing result given one probe

            Parameters
            ----------
            probe : field
                the field on which `function` will be applied
            idnum : int
                    the identification number of the probing

            Returns
            -------
            result : array-like
                the result of applying `function` to `probe`. The exact type
                depends on the function.

        """
        f = self.function(probe)
        if(isinstance(f,field)):
            return f.val
        else:
            return f

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def evaluate(self,results):
        """
            evaluates the probing results

            Parameters
            ----------
            results : list
                the list containing the results of the individual probings.
                The type of the list elements depends on the function.

            Returns
            -------
            final : array-like
                    the final probing result. 1/N Sum[ probing(probe_i) ]
            var : array-like
                    the variance of the final probing result.
                (N(N-1))^(-1) Sum[ ( probing(probe_i) - final)^2 ]
                If the variance is returned, the return will be a tuple with
                `final` in the zeroth entry and `var` in the first entry.

        """
        if(len(results)==0):
            return None
        elif(self.var):
            return np.mean(np.array(results),axis=0,dtype=None,out=None),np.var(np.array(results),axis=0,dtype=None,out=None,ddof=0)/(len(results)-1)
        else:
            return np.mean(np.array(results),axis=0,dtype=None,out=None)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _progress(self,idnum): ## > prints progress status by in upto 10 dots
        tenths = 1+(10*idnum//self.nrun)
        about.infos.cflush(("\b")*10+('.')*tenths+(' ')*(10-tenths))

    def _single_probing(self,zipped): ## > performs one probing operation
        ## generate probe
        np.random.seed(zipped[0])
        probe = self.gen_probe()
        ## do the actual probing
        self._progress(zipped[1])
        return self.probing(zipped[1],probe)

    def _serial_probing(self,zipped): ## > performs the probing operation serially
        try:
            return self._single_probing(zipped)
        except:
            ## kill pool
            os.kill()

    def _parallel_probing(self): ## > performs the probing operations in parallel
        ## define random seed
        seed = np.random.randint(10**8,high=None,size=self.nrun)
        ## build pool
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: multiprocessing "+(' ')*10))
            so.flush()
        pool = mp(processes=self.ncpu,initializer=None,initargs=(),maxtasksperchild=self.nper)
        try:
            ## retrieve results
            results = pool.map_async(self._serial_probing,zip(seed,np.arange(self.nrun,dtype=np.int)),chunksize=None,callback=None).get(timeout=None)
            ## close pool
            about.infos.cflush(" done.")
            pool.close()
        except:
            ## terminate pool
            pool.terminate()
            raise Exception(about._errors.cstring("ERROR: unknown. NOTE: pool terminated.")) ## traceback by looping
        ## cleanup
        results = [rr for rr in results if(rr is not None)]
        if(len(results)<self.nrun):
            about.infos.cflush(" ( %u probe(s) failed, effectiveness == %.1f%% )\n"%(self.nrun-len(results),100*len(results)/self.nrun))
        else:
            about.infos.cflush("\n")
        ## evaluate
        return self.evaluate(results)

    def _nonparallel_probing(self): ## > performs the probing operations one after another
        ## define random seed
        seed = np.random.randint(10**8,high=None,size=self.nrun)
        ## retrieve results
        if(about.infos.status):
            so.write(about.infos.cstring("INFO: looping "+(' ')*10))
            so.flush()
        results = map(self._single_probing,zip(seed,np.arange(self.nrun,dtype=np.int)))
        about.infos.cflush(" done.")
        ## cleanup
        results = [rr for rr in results if(rr is not None)]
        if(len(results)<self.nrun):
            about.infos.cflush(" ( %u probe(s) failed, effectiveness == %.1f%% )\n"%(self.nrun-len(results),100*len(results)/self.nrun))
        else:
            about.infos.cflush("\n")
        ## evaluate
        return self.evaluate(results)

    def __call__(self,loop=False,**kwargs):
        """

            starts the probing process.
            All keyword arguments that can be given to `configure` can also be
            given to `__call__` and have the same effect.

            Parameters
            ----------
            loop : bool, *optional*
                if `loop` is True, then multiprocessing will be disabled and
                all probes are evaluated by a single worker (default: False)

            Returns
            -------
            results : see **Returns** in `evaluate`

            other parameters
            ----------------
            kwargs : see **Parameters** in `configure`

        """
        self.configure(**kwargs)
        if(not about.multiprocessing.status)or(loop):
            return self._nonparallel_probing()
        else:
            return self._parallel_probing()

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.probing>"

##=============================================================================



##-----------------------------------------------------------------------------

class trace_probing(probing):
    """
        ..      __
        ..    /  /_
        ..   /   _/  _____   ____ __   _______   _______
        ..  /  /   /   __/ /   _   / /   ____/ /   __  /
        .. /  /_  /  /    /  /_/  / /  /____  /  /____/
        .. \___/ /__/     \______|  \______/  \______/  probing class

        NIFTY subclass for trace probing (using multiprocessing)

        When called, a trace_probing class instance samples the trace of an
        operator or a function using random fields, whose components are random
        variables with mean 0 and variance 1. When an instance is called it
        returns the mean value of the scalar product of probe and f(probe),
        where probe is a random        field with mean 0 and variance 1.
        The mean is calculated as 1/N Sum[ probe_i.dot(f(probe_i)) ].

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used from parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
            set to `ncpu**2`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)


        See Also
        --------
        probing : The base probing class
        diagonal_probing : A probing class to get the diagonal of an operator
        operator.tr : the trace function uses trace probing in the case of non
            diagonal operators


        Attributes
        ----------
        function : function
            the function, that is applied to the probes
        domain : space
            the space, where the probes live in
        target : space
            the codomain of `domain`
        random : string
            the random number generator used to create the probes
            (either "pm1" or "gau")
        ncpu : int
            the number of cpus used for probing
        nrun : int
            the number of probes to be evaluated, when the instance is called
        nper : int
            number of probes, that will be evaluated by one worker
        var : bool
            whether the variance will be additionally returned, when the
            instance is called

    """
    def __init__(self,op,function=None,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False):
        """
        initializes a trace probing instance

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used from parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
            set to `ncpu**2`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)

        """
        if(not isinstance(op,operator)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(op.nrow()!=op.ncol()):
            raise ValueError(about._errors.cstring("ERROR: trace ill-defined for "+str(op.nrow())+" x "+str(op.ncol())+" matrices."))

        ## check whether callable
        if(function is None)or(not hasattr(function,"__call__")):
            function = op.times
        elif(op==function):
            function = op.times
        ## check whether correctly bound
        if(op!=function.im_self):
            raise NameError(about._errors.cstring("ERROR: invalid input."))
        self.function = function

        ## check given domain
        if(domain is None)or(not isinstance(domain,space)):
            if(self.function in [op.inverse_times,op.adjoint_times]):
                domain = op.target
            else:
                domain = op.domain
        elif(not op.domain.check_codomain(domain))or(not op.target.check_codomain(domain)): ## restrictive
            raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        self.domain = domain

        if(target is None):
            target = domain.get_codomain()
        ## check codomain
        self.domain.check_codomain(target) ## a bit pointless
        self.target = target

        ## check degrees of freedom
        if(op.domain.dof()>self.domain.dof()):
            about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(op.domain.dof())+" / "+str(self.domain.dof())+" ).")

        if(random not in ["pm1","gau"]):
            raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(random)+"'."))
        self.random = random

        self.ncpu = int(max(1,ncpu))
        self.nrun = int(max(self.ncpu**2,nrun))
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1,min(self.nrun//self.ncpu,nper)))

        self.var = bool(var)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self,idnum,probe):
        """
            Computes a single probing result given one probe

            Parameters
            ----------
            probe : field
                the field on which `function` will be applied
            idnum : int
                    the identification number of the probing

            Returns
            -------
            result : float
                    the result of `probe.dot(function(probe))`
        """
        f = self.function(probe)
        if(f is None):
            return None
        else:
            return self.domain.calc_dot(probe.val,f.val) ## discrete inner product

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.trace_probing>"

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

class diagonal_probing(probing):
    """
        ..           __   __                                                     __
        ..         /  / /__/                                                   /  /
        ..    ____/  /  __   ____ __   ____ __   ______    __ ___    ____ __  /  /
        ..  /   _   / /  / /   _   / /   _   / /   _   | /   _   | /   _   / /  /
        .. /  /_/  / /  / /  /_/  / /  /_/  / /  /_/  / /  / /  / /  /_/  / /  /_
        .. \______| /__/  \______|  \___   /  \______/ /__/ /__/  \______|  \___/  probing class
        ..                         /______/

        NIFTY subclass for diagonal probing (using multiprocessing)

        When called, a diagonal_probing class instance samples the diagonal of
        an operator or a function using random fields, whose components are
        random variables with mean 0 and variance 1. When an instance is called
        it returns the mean value of probe*f(probe), where probe is a random
        field with mean 0 and variance 1.
        The mean is calculated as 1/N Sum[ probe_i*f(probe_i) ]
        ('*' denoting component wise multiplication)

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used for parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
            set to `ncpu**2`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)
        save : bool, *optional*
            If `save` is True, then the probing results will be written to the
            hard disk instead of being saved in the RAM. This is recommended
            for high dimensional fields whose probes would otherwise fill up
            the memory. (default: False)
        path : string, *optional*
            the path, where the probing results are saved, if `save` is True.
            (default: "tmp")
        prefix : string, *optional*
            a prefix for the saved probing results. The saved results will be
            named using that prefix and an 8-digit number
            (e.g. "<prefix>00000001.npy"). (default: "")


        See Also
        --------
        trace_probing : A probing class to get the trace of an operator
        probing : The base probing class
        operator.diag : The diag function uses diagonal probing in the case of
            non diagonal operators
        operator.hat : The hat function uses diagonal probing in the case of
            non diagonal operators


        Attributes
        ----------
        function : function
            the function, that is applied to the probes
        domain : space
            the space, where the probes live in
        target : space
            the codomain of `domain`
        random : string
            the random number generator used to create the probes
            (either "pm1" or "gau")
        ncpu : int
            the number of cpus used for probing
        nrun : int
            the number of probes to be evaluated, when the instance is called
        nper : int
            number of probes, that will be evaluated by one worker
        var : bool
            whether the variance will be additionally returned, when the
            instance is called
        save : {string, None}
            the path and prefix for saved probe files. None in the case where
            the probing results are stored in the RAM.

    """
    def __init__(self,op,function=None,domain=None,target=None,random="pm1",ncpu=2,nrun=8,nper=None,var=False,save=False,path="tmp",prefix=""):
        """
        initializes a diagonal probing instance

        Parameters
        ----------
        op : operator
            The operator specified by `op` is the operator to be probed.
            If no operator is given, then probing will be done by applying
            `function` to the probes. (default: None)
        function : function, *optional*
            If no operator has been specified as `op`, then specification of
            `function` is non optional. This is the function, that is applied
            to the probes. (default: `op.times`)
        domain : space, *optional*
            If no operator has been specified as `op`, then specification of
            `domain` is non optional. This is the space that the probes live
            in. (default: `op.domain`)
        target : domain, *optional*
            `target` is the codomain of `domain`
            (default: `op.domain.get_codomain()`)
        random : string, *optional*
            the distribution from which the probes are drawn. `random` can be
            either "pm1" or "gau". "pm1" is a uniform distribution over {+1,-1}
            or {+1,+i,-1,-i}, respectively. "gau" is a normal distribution with
            zero-mean and unit-variance (default: "pm1")
        ncpu : int, *optional*
            the number of cpus to be used for parallel probing. (default: 2)
        nrun : int, *optional*
            the number of probes to be evaluated. If `nrun<ncpu**2`, it will be
            set to `ncpu**2`. (default: 8)
        nper : int, *optional*
            this number specifies how many probes will be evaluated by one
            worker. Afterwards a new worker will be created to evaluate a chunk
            of `nper` probes.
            If for example `nper=nrun/ncpu`, then every worker will be created
            for every cpu. This can lead to the case, that all workers but one
            are already finished, but have to wait for the last worker that
            might still have a considerable amount of evaluations left. This is
            obviously not very effective.
            If on the other hand `nper=1`, then for each evaluation a worker will
            be created. In this case all cpus will work until nrun probes have
            been evaluated.
            It is recommended to leave `nper` as the default value. (default: 8)
        var : bool, *optional*
            If `var` is True, then the variance of the sampled function will
            also be returned. The result is then a tuple with the mean in the
            zeroth entry and the variance in the first entry. (default: False)
        save : bool, *optional*
            If `save` is True, then the probing results will be written to the
            hard disk instead of being saved in the RAM. This is recommended
            for high dimensional fields whose probes would otherwise fill up
            the memory. (default: False)
        path : string, *optional*
            the path, where the probing results are saved, if `save` is True.
            (default: "tmp")
        prefix : string, *optional*
            a prefix for the saved probing results. The saved results will be
            named using that prefix and an 8-digit number
            (e.g. "<prefix>00000001.npy"). (default: "")

        """

        if(not isinstance(op,operator)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(op.nrow()!=op.ncol()):
            raise ValueError(about._errors.cstring("ERROR: diagonal ill-defined for "+str(op.nrow())+" x "+str(op.ncol())+" matrices."))

        ## check whether callable
        if(function is None)or(not hasattr(function,"__call__")):
            function = op.times
        elif(op==function):
            function = op.times
        ## check whether correctly bound
        if(op!=function.im_self):
            raise NameError(about._errors.cstring("ERROR: invalid input."))
        self.function = function

        ## check given domain
        if(domain is None)or(not isinstance(domain,space)):
            if(self.function in [op.inverse_times,op.adjoint_times]):
                domain = op.target
            else:
                domain = op.domain
        elif(not op.domain.check_codomain(domain))or(not op.target.check_codomain(domain)): ## restrictive
            raise ValueError(about._errors.cstring("ERROR: incompatible domains."))
        self.domain = domain

        if(target is None):
            target = domain.get_codomain()
        ## check codomain
        self.domain.check_codomain(target) ## a bit pointless
        self.target = target

        ## check degrees of freedom
        if(self.domain.dof()>op.domain.dof()):
            about.infos.cprint("INFO: variant numbers of degrees of freedom ( "+str(self.domain.dof())+" / "+str(op.domain.dof())+" ).")

        if(random not in ["pm1","gau"]):
            raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(random)+"'."))
        self.random = random

        self.ncpu = int(max(1,ncpu))
        self.nrun = int(max(self.ncpu**2,nrun))
        if(nper is None):
            self.nper = None
        else:
            self.nper = int(max(1,min(self.nrun//self.ncpu,nper)))

        self.var = bool(var)

        if(save):
            path = os.path.expanduser(str(path))
            if(not os.path.exists(path)):
                os.makedirs(path)
            self.save = os.path.join(path,str(prefix)) ## (back)slash inserted if needed
        else:
            self.save = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def configure(self,**kwargs):
        """
            changes the attributes of the instance

            Parameters
            ----------
            random : string, *optional*
                the random number generator used to create the probes
                (default: "pm1")
            ncpu : int, *optional*
                the number of cpus to be used for parallel probing
                (default: 2)
            nrun : int, *optional*
                the number of probes to be evaluated. If `nrun<ncpu**2`, it will
                be set to `ncpu**2`. (default: 8)
            nper : int, *optional*
                number of probes, that will be evaluated by one worker
                (default: 8)
            var : bool, *optional*
                whether the variance will be additionally returned
                (default: False)
            save : bool, *optional*
                whether the individual probing results will be saved to the HDD
                (default: False)
            path : string, *optional*
                the path, where the probing results are saved (default: "tmp")
            prefix : string, *optional*
                a prefix for the saved probing results (default: "")

        """
        if(kwargs.has_key("random")):
            if(kwargs.get("random") not in ["pm1","gau"]):
                raise ValueError(about._errors.cstring("ERROR: unsupported random key '"+str(kwargs.get("random"))+"'."))
            self.random = random

        if(kwargs.has_key("ncpu")):
            self.ncpu = int(max(1,kwargs.get("ncpu")))
        if(kwargs.has_key("nrun")):
            self.nrun = int(max(self.ncpu**2,kwargs.get("nrun")))
        if(kwargs.has_key("nper")):
            if(kwargs.get("nper") is None):
                self.nper = None
            else:
                self.nper = int(max(1,min(self.nrun//self.ncpu,kwargs.get("nper"))))

        if(kwargs.has_key("var")):
            self.var = bool(kwargs.get("var"))

        if(kwargs.has_key("save")):
            if(kwargs.get("save")):
                if(kwargs.has_key("path")):
                    path = kwargs.get("path")
                else:
                    if(self.save is not None):
                        about.warnings.cprint("WARNING: save path set to default.")
                    path = "tmp"
                if(kwargs.has_key("prefix")):
                    prefix = kwargs.get("prefix")
                else:
                    if(self.save is not None):
                        about.warnings.cprint("WARNING: save prefix set to default.")
                    prefix = ""
                path = os.path.expanduser(str(path))
                if(not os.path.exists(path)):
                    os.makedirs(path)
                self.save = os.path.join(path,str(prefix)) ## (back)slash inserted if needed
            else:
                self.save = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def probing(self,idnum,probe):

        """
            Computes a single probing result given one probe

            Parameters
            ----------
            probe : field
                the field on which `function` will be applied
            idnum : int
                    the identification number of the probing

            Returns
            -------
            result : ndarray
                    the result of `probe*(function(probe))`
        """
        f = self.function(probe)
        if(f is None):
            return None
        else:
            if(self.save is None):
                return np.conjugate(probe.val)*f.val
            else:
                result = np.conjugate(probe.val)*f.val
                np.save(self.save+"%08u"%idnum,result)
                return idnum

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def evaluate(self,results):
        """
            evaluates the probing results

            Parameters
            ----------
            results : list
                the list of ndarrays containing the results of the individual
                probings.

            Returns
            -------
            final : ndarray
                    the final probing result. 1/N Sum[ probe_i*f(probe_i) ]
            var : ndarray
                    the variance of the final probing result.
                (N(N-1))^(-1) Sum[ (probe_i*f(probe_i) - final)^2 ]
                If the variance is returned, the return will be a tuple with
                final in the zeroth entry and var in the first entry.

        """
        num = len(results)
        if(num==0):
            return None
        elif(self.save is None):
            if(self.var):
                return np.mean(np.array(results),axis=0,dtype=None,out=None),np.var(np.array(results),axis=0,dtype=None,out=None,ddof=0)/(num-1)
            else:
                return np.mean(np.array(results),axis=0,dtype=None,out=None)
        else:
            final = np.copy(np.load(self.save+"%08u.npy"%results[0],mmap_mode=None))
            for ii in range(1,num):
                final += np.load(self.save+"%08u.npy"%results[ii],mmap_mode=None)
            if(self.var):
                var = np.zeros(self.domain.dim(split=True),dtype=self.domain.datatype,order='C')
                for ii in range(num):
                    var += (final-np.load(self.save+"%08u.npy"%results[ii],mmap_mode=None))**2
                return final/num,var/(num*(num-1))
            else:
                return final/num

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.diagonal_probing>"

##-----------------------------------------------------------------------------

## IDEA: diagonal_inference

