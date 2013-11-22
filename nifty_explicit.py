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
    ..     /__/ /__/ /__/ /__/    \___/  \___   /  explicit
    ..                                  /______/

    TODO: documentation

"""
from __future__ import division
#import numpy as np
from nifty_core import *


##-----------------------------------------------------------------------------

class matrix_operator(operator):
    """
        TODO: documentation

    """
    epsilon = 1E-12 ## absolute precision for comparisons to identity

    def __init__(self,domain,matrix,bare=True,sym=None,uni=None,target=None,para=None):
        """
            TODO: documentation

        """
        ## check domain
        if(not isinstance(domain,space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        elif(np.size(matrix,axis=None)%domain.dim(split=False)!=0):
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(matrix,axis=None))+" <> "+str(domain.dim(split=False))+" )."))
        self.domain = domain

        ## check shape
        val = np.array(matrix).reshape(-1,self.domain.dim(split=False))
        if(val.size>1048576):
            about.warnings.cprint("WARNING: matrix size > 2 ** 20.")

        ## check target
        if(target is not None):
            if(not isinstance(target,space)):
                raise TypeError(about._errors.cstring("ERROR: invalid input."))
            elif(val.shape[0]!=target.dim(split=False)):
                raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(val.shape[0])+" <> "+str(target.dim(split=False))+" )."))
            elif(target!=self.domain):
                sym = False
                uni = False
        elif(val.shape[0]==val.shape[1]):
            target = self.domain
        else:
            raise TypeError(about._errors.cstring("ERROR: insufficient input."))
        self.target = target

        ## check flags
        if(val.shape[0]==val.shape[1]):
            if(sym is None):
                adj = np.conjugate(val.T)
                sym = np.all(np.absolute(val-adj)<self.epsilon)
                if(uni is None):
                    uni = np.all(np.absolute(np.dot(val,adj)-np.diag(np.ones(len(val),dtype=np.int,order='C'),k=0))<self.epsilon)
            elif(uni is None):
                adj = np.conjugate(val.T)
                uni = np.all(np.absolute(np.dot(val,adj)-np.diag(np.ones(len(val),dtype=np.int,order='C'),k=0))<self.epsilon)
        else:
            sym = False
            uni = False
        self.sym = bool(sym)
        self.uni = bool(uni)

        ## check datatype
        if(np.iscomplexobj(val)):
            if(np.all(np.imag(val)==0)):
                val = np.real(val).astype(min(val.dtype,self.domain.vol.dtype,self.target.vol.dtype))
            else:
                val = val.astype(min(val.dtype,self.domain.datatype,self.target.datatype))
        else:
            val = val.astype(min(val.dtype,self.domain.vol.dtype,self.target.vol.dtype))
        ## weight if ... (given `domain`, `target` and `val`)
        if(isinstance(bare,tuple)):
            if(len(bare)!=2):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                val = self._calc_weight_rows(val,-bool(bare[0]))
                val = self._calc_weight_cols(val,-bool(bare[1]))
        elif(not bare):
            val = self._calc_weight_rows(val,-1)
        self.val = val

        if(self.domain.discrete)and(self.target.discrete): ## FIXME: operator ???
            self.imp = True
        else:
            self.imp = False

        ## set parameters
        if(para is not None):
            self.para = para

        self.inv = None ## defined when needed

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_matrix(self,newmatrix,bare=True,sym=None,uni=None,):
        """
            TODO: documentation

        """
        self.val = newmatrix
        self.inv = None

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_weight_row(self,x,power): ## > weight row and flatten
        return self.domain.calc_weight(x,power=power).flatten(order='C')

    def _calc_weight_col(self,x,power): ## > weight column and flatten
        return self.target.calc_weight(x,power=power).flatten(order='C')

    def _calc_weight_rows(self,X,power=1): ## > weight all rows
        return np.apply_along_axis(self._calc_weight_row,0,X,power)

    def _calc_weight_cols(self,X,power=1): ## > weight all columns
        return np.apply_along_axis(self._calc_weight_col,1,X,power)

    def weight(self,rowpower=0,colpower=0,overwrite=False):
        """
            TODO: documentation

        """
        if(overwrite):
            if(rowpower): ## rowpower <> 0
                self.val = self._calc_weight_rows(self.val,rowpower)
            if(colpower): ## colpower <> 0
                self.val = self._calc_weight_cols(self.val,colpower)
        else:
            X = np.copy(self.val)
            if(rowpower): ## rowpower <> 0
                X = self._calc_weight_rows(X,rowpower)
            if(colpower): ## colpower <> 0
                X = self._calc_weight_cols(X,colpower)
            return X

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the matirx to a given field
        x_ = field(self.target,val=np.dot(self.val,x.val.flatten(order='C'),out=None),target=x.target)
        return x_

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        x_ = field(self.domain,val=np.dot(np.conjugate(self.val.T),x.val.flatten(order='C'),out=None),target=x.target)
        return x_

    def _inverse_multiply(self,x,**kwargs): ## > applies the inverse operator to a given field
        x_ = field(self.domain,val=np.dot(self.inv,x.val.flatten(order='C'),out=None),target=x.target)
        return x_

    def _inverse_adjoint_multiply(self,x,**kwargs): ## > applies the adjoint inverse operator to a given field
        x_ = field(self.target,val=np.dot(np.conjugate(self.inv.T),x.val.flatten(order='C'),out=None),target=x.target)
        return x_

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        ## check whether self-inverse
        if(self.sym)and(self.uni):
            return self.times(x,**kwargs)

        ## check whether square matrix
        elif(self.nrow()!=self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(op.nrow())+" x "+str(op.ncol())+" matrices."))
        ## set inverse if ...
        elif(self.inv is None):
            self.inv = np.linalg.inv(self.val)

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

            Notes
            -----
            For linear operators represented by square matrices, inversion and
            adjungation and inversion commute.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        return self.inverse_adjoint_times(x,**kwargs)

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

            Notes
            -----
            For linear operators represented by square matrices, inversion and
            adjungation and inversion commute.

            Raises
            ------
            ValueError
                If it is no square matrix.

        """
        ## check whether square matrix
        if(self.nrow()!=self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(op.nrow())+" x "+str(op.ncol())+" matrices."))

        ## check whether self-adjoint
        if(self.sym):
            return self.inverse_times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.times(x,**kwargs)

        ## set inverse if ...
        elif(self.inv is None):
            self.inv = np.linalg.inv(self.val)

        ## prepare
        x_ = self._briefing(x,self.domain,True)
        ## apply operator
        x_ = self._inverse_adjoint_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



##-----------------------------------------------------------------------------























