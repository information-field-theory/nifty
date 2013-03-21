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

    NIFTY demo for (critical) Wiener filtering of Gaussian random signals.

"""
from __future__ import division
from nifty import *
from nifty.nifty_cmaps import *
from nifty.nifty_power import *
from scipy.sparse.linalg import LinearOperator as lo
from scipy.sparse.linalg import cg


note = notification()


##-----------------------------------------------------------------------------

## spaces
r1 = rg_space(512,1,zerocenter=False)
r2 = rg_space(64,2)
h = hp_space(16)
g = gl_space(48)
z = s_space = k = k_space = p = d_space = None

## power spectrum (and more)
power = powerindex = powerundex = kindex = rho = None

## operators
S = Sk = R = N = Nj = D = None

## fields
s = n = d = j = m = None

## propagator class
class propagator_operator(operator):
    """
        This is the information propagator from the Wiener filter formula.
        It is defined by its inverse. It is given the prior signal covariance S,
        the noise covariance N and the response R in para.
    """
    def _inverse_multiply(self,x):
        ## The inverse can be calculated directly
        S,N,R = self.para
        return S.inverse_times(x)+R.adjoint_times(N.inverse_times(R.times(x)))

    ## the inverse multiplication and multiplication with S modified to return 1d arrays
    _matvec = (lambda self,x: self.inverse_times(x).val.flatten())
    _precon = (lambda self,x: self.para[0].times(x).val.flatten())

    def _multiply(self,x):
        """
            the operator is defined by its inverse, so multiplication has to be
            done by inverting the inverse numerically using the conjugate gradient
            method from scipy
        """
        A = lo(shape=tuple(self.dim()),matvec=self._matvec,dtype=self.domain.datatype) ## linear operator
        b = x.val.flatten()
        x_,info = cg(A,b,x0=None,tol=1.0E-5,maxiter=10*len(x),xtype=None,M=None,callback=None) ## conjugate gradient
        if(info==0):
            return x_
        else:
            note.cprint("NOTE: conjugate gradient failed.")
            return None

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

def setup(space,s2n=3,nvar=None):
    """
        sets up the spaces, operators and fields

        Parameters
        ----------
        space : space
            the signal lives in `space`
        s2n : positive number, *optional*
            `s2n` is the signal to noise ratio (default: 3)
        nvar = positive number, *optional*
            the noise variance, `nvar` will be calculated according to
            `s2n` if not specified (default: None)
    """
    global z,s_space,k,k_space,p,d_space,power,powerindex,powerundex,kindex,rho,S,Sk,R,N,Nj,D,s,n,d,j,m

    ## signal space
    z = s_space = space

    ## conjugate space
    k = k_space = s_space.get_codomain()
    ## the power indices are calculated once and saved
    powerindex = k_space.get_power_index()
    powerundex = k_space.get_power_undex()
    kindex,rho = k_space.get_power_index(irreducible=True)

    ## power spectrum
    power = [42/(kk+1)**3 for kk in kindex]

    ## prior signal covariance operator (power operator)
    S = power_operator(k_space,spec=power,pindex=powerindex)
    ## projection operator to the spectral bands
    Sk = S.get_projection_operator(pindex=powerindex)
    ## the Gaussian random field generated from its power operator S
    s = S.get_random_field(domain=s_space,target=k_space)

    ## response
    R = response_operator(s_space,sigma=0,mask=1)
    ## data space
    p = d_space = R.target

    ## calculating the noise covariance
    if(nvar is None):
        svar = np.var(s.val) ## given unit response
        nvar = svar/s2n**2
    ## noise covariance operator
    N = diagonal_operator(d_space,diag=nvar,bare=True)
    ## Gaussian noise generated from its covariance N
    n = N.get_random_field(domain=d_space,target=d_space)

    ## data
    d = R(s)+n

##-----------------------------------------------------------------------------

##=============================================================================

def run(space=r1,s2n=3,nvar=None,**kwargs):
    """
        runs the demo of the generalised Wiener filter

        Parameters
        ----------
        space : space, *optional*
            `space` can be any space from nifty, that supports the plotting
            routine (default: r1 = rg_space(512,1,zerocenter=False))
        s2n : positive number, *optional*
            `s2n` is the signal to noise (default: 3)
        nvar : positive number, *optional*
            the noise variance, `nvar` will be calculated according to
            `s2n` if not specified (default: None)
    """
    global z,s_space,k,k_space,p,d_space,power,powerindex,powerundex,kindex,rho,S,Sk,R,N,Nj,D,s,n,d,j,m

    ## setting up signal, noise, data and the operators S, N and R
    setup(space,s2n=s2n,nvar=nvar)

    ## information source
    j = R.adjoint_times(N.inverse_times(d))

    ## information propagator
    D = propagator_operator(s_space,sym=True,imp=True,para=[S,N,R])

    ## reconstructed map
    m = D(j)
    if(m is None):
        return None

    ## fields
    s.plot(title="signal",**kwargs)
#    n.cast_domain(s_space,newtarget=k_space)
#    n.plot(title="noise",**kwargs)
#    n.cast_domain(d_space,newtarget=d_space)
    d.cast_domain(s_space,newtarget=k_space)
    d.plot(title="data",vmin=np.min(s.val),vmax=np.max(s.val),**kwargs)
    d.cast_domain(d_space,newtarget=d_space)
    m.plot(title="reconstructed map",vmin=np.min(s.val),vmax=np.max(s.val),**kwargs)

    ## power spectrum
#    s.plot(title="power spectra",power=True,other=(m,power),mono=False,kindex=kindex)

    ## uncertainty
#    uncertainty = D.hat(bare=True,nrun=D.domain.dim()//4,target=k_space)
#    if(np.all(uncertainty.val>0)):
#        sqrt(uncertainty).plot(title="standard deviation",**kwargs)

##=============================================================================

##-----------------------------------------------------------------------------

def run_critical(space=r2,s2n=3,nvar=None,q=1E-12,alpha=1,perception=[1,0],**kwargs):
    """
        runs the demo of the critical generalised Wiener filter

        Parameters
        ----------
        space : space, *optional*
            `space` can be any space from nifty, that supports the plotting
            routine (default: r2 = rg_space(64,2))
        s2n : positive number, *optional*
            `s2n` is the signal to noise (default: 3)
        nvar : positive number, *optional*
            the noise variance, `nvar` will be calculated according to
            `s2n` if not specified (default: None)
        q : positive number, *optional*
            `q` is the minimal power on all scales (default: 1E-12)
        alpha : a number >= 1, *optional*
            `alpha` = 1 means Jeffreys prior for the power spectrum (default: 1)
        perception : array of shape (2,1), *optional*
            perception[0] is delta, perception[1] is epsilon. They are tuning
            factors for the filter (default: [1,0])

        See Also
        --------
        infer_power

    """
    global z,s_space,k,k_space,p,d_space,power,powerindex,powerundex,kindex,rho,S,Sk,R,N,Nj,D,s,n,d,j,m

    ## setting up signal, noise, data and the operators S, N and R
    setup(space,s2n=s2n,nvar=nvar)
    if(perception[1] is None):
        perception[1] = rho/2*(perception[0]-1)

    ## information source
    j = R.adjoint_times(N.inverse_times(d))

    ## unknown power spectrum, the power operator is given an initial value
    S.set_power(42,bare=True,pindex=powerindex) ## The answer is 42. I double checked.
    ## the power spectrum is drawn from the first guess power operator
    pk = S.get_power(bare=False) ## non-bare(!)

    ## information propagator
    D = propagator_operator(s_space,sym=True,imp=True,para=[S,N,R])

    ## iterative reconstruction of the power spectrum and the map
    iteration = 0
    while(True):
        iteration += 1

        ## the Wiener filter reconstruction using the current power spectrum
        m = D(j)
        if(m is None):
            return None

        ## measuring a new estimated power spectrum from the current reconstruction
        b1 = Sk.pseudo_tr(m) ## == Sk(m).pseudo_dot(m), but faster
        b2 = Sk.pseudo_tr(D,nrun=np.sqrt(Sk.domain.dim())//4) ## probing of the partial traces of D
        pk_new = (2*q+b1+perception[0]*b2)/(rho+2*(alpha-1+perception[1])) ## non-bare(!)
        pk_new = smooth_power(pk_new,kindex,mode="2s",exclude=min(8,len(power))) ## smoothing
        ## the power operator is given the new spectrum
        S.set_power(pk_new,bare=False,pindex=powerindex) ## auto-updates D

        ## check convergence
        log_change = np.max(np.abs(log(pk_new/pk)))
        if(log_change<=0.01):
            note.cprint("NOTE: accuracy reached in iteration %u."%(iteration))
            break
        else:
            note.cprint("NOTE: log-change == %4.3f ( > 1%% ) in iteration %u."%(log_change,iteration))
            pk = np.copy(pk_new)

    ## fields
    s.plot(title="signal",**kwargs)
#    n.cast_domain(s_space,newtarget=k_space)
#    n.plot(title="noise",**kwargs)
#    n.cast_domain(d_space,newtarget=d_space)
    d.cast_domain(s_space,newtarget=k_space)
    d.plot(title="data",vmin=np.min(s.val),vmax=np.max(s.val),**kwargs)
    d.cast_domain(d_space,newtarget=d_space)
    m.plot(title="reconstructed map",vmin=np.min(s.val),vmax=np.max(s.val),**kwargs)

    ## power spectrum
    s.plot(title="power spectra",power=True,other=(S.get_power(pundex=powerundex),power),mono=False,kindex=kindex)

    ## uncertainty
#    uncertainty = D.hat(bare=True,nrun=D.domain.dim()//4,target=k_space)
#    if(np.all(uncertainty.val>0)):
#        sqrt(uncertainty).plot(title="standard deviation")

##-----------------------------------------------------------------------------

