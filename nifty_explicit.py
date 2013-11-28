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
    ..                 __   ____   __
    ..               /__/ /   _/ /  /_
    ..     __ ___    __  /  /_  /   _/  __   __
    ..   /   _   | /  / /   _/ /  /   /  / /  /
    ..  /  / /  / /  / /  /   /  /_  /  /_/  /
    .. /__/ /__/ /__/ /__/    \___/  \___   /  explicit
    ..                              /______/

    TODO: documentation

"""
from __future__ import division
#import numpy as np
from nifty_core import *


##-----------------------------------------------------------------------------

class explicit_operator(operator):
    """
        ..
        ..
        ..                                    __     __             __   __
        ..                                  /  /   /__/           /__/ /  /_
        ..    _______  __   __    ______   /  /    __   _______   __  /   _/
        ..  /   __  / \  \/  /  /   _   | /  /   /  / /   ____/ /  / /  /
        .. /  /____/  /     /  /  /_/  / /  /_  /  / /  /____  /  / /  /_
        .. \______/  /__/\__\ /   ____/  \___/ /__/  \______/ /__/  \___/  operator
        ..                   /__/

        TODO: documentation

    """
    epsilon = 1E-12 ## absolute precision for comparisons to identity

    def __init__(self,domain,matrix,bare=True,sym=None,uni=None,target=None):
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
        val = np.array(matrix).reshape((-1,self.domain.dim(split=False)))
        if(val.size>1048576):
            about.infos.cprint("INFO: matrix size > 2 ** 20.")

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

        ## check datatype
        if(np.any(np.iscomplex(val))):
            datatype,purelyreal = max(min(val.dtype,self.domain.datatype),min(val.dtype,self.target.datatype)),False
        else:
            datatype,purelyreal = max(min(val.dtype,self.domain.vol.dtype),min(val.dtype,self.target.vol.dtype)),True
        ## weight if ... (given `domain` and `target`)
        if(isinstance(bare,tuple)):
            if(len(bare)!=2):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                val = self._calc_weight_rows(val,-bool(bare[0]))
                val = self._calc_weight_cols(val,-bool(bare[1]))
        elif(not bare):
            val = self._calc_weight_rows(val,-1)
        if(purelyreal):
            self.val = np.real(val).astype(datatype)
        else:
            self.val = val.astype(datatype)







#        if(np.iscomplexobj(val)):
#            if(np.all(np.imag(val)==0)):
#                val = np.real(val).astype(min(val.dtype,self.domain.vol.dtype,self.target.vol.dtype))
#            else:
#                val = val.astype(min(val.dtype,self.domain.datatype,self.target.datatype))
#        else:
#            val = val.astype(min(val.dtype,self.domain.vol.dtype,self.target.vol.dtype))
#        ## weight if ... (given `domain` and `target`)
#        if(isinstance(bare,tuple)):
#            if(len(bare)!=2):
#                raise ValueError(about._errors.cstring("ERROR: invalid input."))
#            else:
#                val = self._calc_weight_rows(val,-bool(bare[0]))
#                val = self._calc_weight_cols(val,-bool(bare[1]))
#        elif(not bare):
#            val = self._calc_weight_rows(val,-1)
#        self.val = val

        ## check hidden degrees of freedom
        self._hidden = np.array([self.domain.dim(split=False)<self.domain.dof(),self.target.dim(split=False)<self.target.dof()],dtype=np.bool)
#        if(np.any(self._hidden)):
#            about.infos.cprint("INFO: inappropriate space.")

        ## check flags
        self.sym,self.uni = self._check_flags(sym=sym,uni=uni)
        if(self.domain.discrete)and(self.target.discrete):
            self.imp = True
        else:
            self.imp = False ## bare matrix is stored for efficiency

        self._inv = None ## defined when needed

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _check_flags(self,sym=None,uni=None): ## > determine `sym` and `uni`
        if(self.val.shape[0]==self.val.shape[1]):
            if(sym is None):
                adj = np.conjugate(self.val.T)
                sym = np.all(np.absolute(self.val-adj)<self.epsilon)
                if(uni is None):
                    uni = np.all(np.absolute(self._calc_mul(adj,0)-np.diag(1/self.target.get_meta_volume(total=False),k=0))<self.epsilon)
            elif(uni is None):
                adj = np.conjugate(self.val.T)
                uni = np.all(np.absolute(self._calc_mul(adj,0)-np.diag(1/self.target.get_meta_volume(total=False),k=0))<self.epsilon)
            return bool(sym),bool(uni)
        else:
            return False,False

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _set_inverse(self): ## > define inverse matrix
        if(self._inv is None):
            if(np.any(self._hidden)):
                about.warnings.cprint("WARNING: inappropriate inversion.")
            self._inv = np.linalg.inv(self.weight(rowpower=1,colpower=1,overwrite=False))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_matrix(self,newmatrix,bare=True,sym=None,uni=None):
        """
            TODO: documentation

        """
        ## check shape
        val = np.array(newmatrix).reshape((-1,self.domain.dim(split=False)))
        if(val.size>1048576):
            about.warnings.cprint("WARNING: matrix size > 2 ** 20.")

        ## check datatype
        if(np.iscomplexobj(val)):
            if(np.all(np.imag(val)==0)):
                val = np.real(val).astype(min(val.dtype,self.domain.vol.dtype,self.target.vol.dtype))
            else:
                val = val.astype(min(val.dtype,self.domain.datatype,self.target.datatype))
        else:
            val = val.astype(min(val.dtype,self.domain.vol.dtype,self.target.vol.dtype))
        ## weight if ... (given `domain` and `target`)
        if(isinstance(bare,tuple)):
            if(len(bare)!=2):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                val = self._calc_weight_rows(val,power=-bool(bare[0]))
                val = self._calc_weight_cols(val,power=-bool(bare[1]))
        elif(not bare):
            val = self._calc_weight_rows(val,-1)
        self.val = val

        ## check flags
        self.sym,self.uni = self._check_flags(sym=sym,uni=uni)
        self._inv = None ## reset

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_matrix(self,bare=True):
        """
            TODO: documentation

        """
        if(bare==True)or(self.imp):
            return self.val
        ## weight if ...
        elif(isinstance(bare,tuple)):
            if(len(bare)!=2):
                raise ValueError(about._errors.cstring("ERROR: invalid input."))
            else:
                return self.weight(rowpower=bool(bare[0]),colpower=bool(bare[1]),overwrite=False)
        elif(not bare):
            return self.weight(rowpower=bool(bare),colpower=0,overwrite=False)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calc_weight_row(self,x,power): ## > weight row and flatten
        return self.domain.calc_weight(x,power=power).flatten(order='C')

    def _calc_weight_col(self,x,power): ## > weight column and flatten
        return self.target.calc_weight(x,power=power).flatten(order='C')

    def _calc_weight_rows(self,X,power=1): ## > weight all rows
        return np.apply_along_axis(self._calc_weight_row,1,X,power)

    def _calc_weight_cols(self,X,power=1): ## > weight all columns
        return np.apply_along_axis(self._calc_weight_col,0,X,power)

    def weight(self,rowpower=0,colpower=0,overwrite=False):
        """
            TODO: documentation

        """
        if(overwrite):
            if(not self.domain.discrete)and(rowpower): ## rowpower <> 0
                self.val = self._calc_weight_rows(self.val,rowpower)
            if(not self.target.discrete)and(colpower): ## colpower <> 0
                self.val = self._calc_weight_cols(self.val,colpower)
        else:
            X = np.copy(self.val)
            if(not self.domain.discrete)and(rowpower): ## rowpower <> 0
                X = self._calc_weight_rows(X,rowpower)
            if(not self.target.discrete)and(colpower): ## colpower <> 0
                X = self._calc_weight_cols(X,colpower)
            return X

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _multiply(self,x,**kwargs): ## > applies the matirx to a given field
        if(self._hidden[0]): ## hidden degrees of freedom
            x_ = np.apply_along_axis(self.domain.calc_dot,1,self.val,np.conjugate(x.val))
        else:
            x_ = np.dot(self.val,x.val.flatten(order='C'),out=None)
        return x_

    def _adjoint_multiply(self,x,**kwargs): ## > applies the adjoint operator to a given field
        if(self._hidden[1]): ## hidden degrees of freedom
            x_ = np.apply_along_axis(self.target.calc_dot,0,np.conjugate(self.val),np.conjugate(x.val))
        else:
            x_ = np.dot(np.conjugate(self.val.T),x.val.flatten(order='C'),out=None)
        return x_

    def _inverse_multiply(self,x,**kwargs): ## > applies the inverse operator to a given field
        if(self._hidden[1]): ## hidden degrees of freedom
            x_ = np.apply_along_axis(self.target.calc_dot,1,self._inv,np.conjugate(x.val))
        else:
            x_ = np.dot(self._inv,x.val.flatten(order='C'),out=None)
        return x_

    def _inverse_adjoint_multiply(self,x,**kwargs): ## > applies the adjoint inverse operator to a given field
        if(self._hidden[0]): ## hidden degrees of freedom
            x_ = np.apply_along_axis(self.domain.calc_dot,0,np.conjugate(self.val),np.conjugate(x.val))
        else:
            x_ = np.dot(np.conjugate(self._inv.T),x.val.flatten(order='C'),out=None)
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
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))

        self._set_inverse()

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
        return self._inverse_adjoint_times(x,**kwargs)

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
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))

        ## check whether self-adjoint
        if(self.sym):
            return self._inverse_times(x,**kwargs)
        ## check whether unitary
        if(self.uni):
            return self.times(x,**kwargs)

        self._set_inverse()

        ## prepare
        x_ = self._briefing(x,self.domain,True)
        ## apply operator
        x_ = self._inverse_adjoint_multiply(x_,**kwargs)
        ## evaluate
        return self._debriefing(x,x_,self.target,True)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def tr(self,domain=None,**kwargs):
        """
            Computes the trace of the operator.

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
                number of tasks performed by one process (default: 1)
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
        if(self.domain!=self.target):
            raise ValueError(about._errors.cstring("ERROR: trace ill-defined."))

        if(domain is None)or(domain==self.domain):
            diag = self.diag(bare=False,domain=self.domain)
            if(self._hidden[0]): ## hidden degrees of freedom
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),diag) ## discrete inner product
            else:
                return np.sum(diag,axis=None,dtype=None,out=None)
        else:
            return super(explicit_operator,self).tr(domain=domain,**kwargs) ## probing

    def inverse_tr(self,domain=None,**kwargs):
        """
            Computes the trace of the inverse operator.

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
                number of tasks performed by one process (default: 1)
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
        if(self.domain!=self.target):
            raise ValueError(about._errors.cstring("ERROR: trace ill-defined."))

        if(domain is None)or(domain==self.domain):
            diag = self.inverse_diag(bare=False,domain=self.domain)
            if(self._hidden[0]): ## hidden degrees of freedom
                return self.domain.calc_dot(np.ones(self.domain.dim(split=True),dtype=self.domain.datatype,order='C'),diag) ## discrete inner product
            else:
                return np.sum(diag,axis=None,dtype=None,out=None)
        else:
            return super(explicit_operator,self).tr(domain=domain,**kwargs) ## probing

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
                number of tasks performed by one process (default: 1)
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
        if(self.val.shape[0]!=self.val.shape[1]):
            raise ValueError(about._errors.cstring("ERROR: diagonal ill-defined for "+str(self.val.shape[0])+" x "+str(self.val.shape[1])+" matrices."))
        if(self.domain!=self.target)and(not bare):
            about.warnings.cprint("WARNING: ambiguous non-bare diagonal.")

        if(domain is None)or(domain==self.domain):
            diag = np.diagonal(self.val,offset=0,axis1=0,axis2=1)
            ## weight if ...
            if(not self.domain.discrete)and(not bare):
                diag = self.domain.calc_weight(diag,power=1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
            return diag
        elif(domain==self.target):
            diag = np.diagonal(np.conjugate(self.val.T),offset=0,axis1=0,axis2=1)
            ## weight if ...
            if(not self.target.discrete)and(not bare):
                diag = self.target.calc_weight(diag,power=1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
            return diag
        else:
            return super(explicit_operator,self).diag(bare=bare,domain=domain,**kwargs) ## probing

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
                space wherein the probes live (default: self.target)
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
                number of tasks performed by one process (default: 1)
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
        if(self.val.shape[0]!=self.val.shape[1]):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(self.val.shape[0])+" x "+str(self.val.shape[1])+" matrices."))
        if(self.domain!=self.target)and(not bare):
            about.warnings.cprint("WARNING: ambiguous non-bare diagonal.")

        if(domain is None)or(domain==self.target):
            self._set_inverse()
            diag = np.diagonal(self._inv,offset=0,axis1=0,axis2=1)
            ## weight if ...
            if(not self.target.discrete)and(not bare):
                diag = self.target.calc_weight(diag,power=1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
            return diag
        elif(domain==self.domain):
            self._set_inverse()
            diag = np.diagonal(np.conjugate(self._inv.T),offset=0,axis1=0,axis2=1)
            ## weight if ...
            if(not self.domain.discrete)and(not bare):
                diag = self.domain.calc_weight(diag,power=1)
                ## check complexity
                if(np.all(np.imag(diag)==0)):
                    diag = np.real(diag)
            return diag
        else:
            return super(explicit_operator,self).inverse_diag(bare=bare,domain=domain,**kwargs) ## probing

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def det(self):
        """
            Computes the determinant of the matrix.

            Returns
            -------
            det : float
                The determinant

        """
        if(self.domain!=self.target):
            raise ValueError(about._errors.cstring("ERROR: determinant ill-defined."))

        if(np.any(self._hidden)):
            about.warnings.cprint("WARNING: inappropriate determinant calculation.")
        return np.linalg.det(self.weight(rowpower=0.5,colpower=0.5,overwrite=False))

    def inverse_det(self):
        """
            Computes the determinant of the inverse matrix.

            Returns
            -------
            det : float
                The determinant

        """
        det = self.det()
        if(det<>0):
            return 1/det
        else:
            raise ValueError(about._errors.cstring("ERROR: singular matrix."))

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __len__(self):
        return int(self.nrow()[0])

    def __getitem__(self,key):
        return self.val[key]

    def __setitem__(self,key,value):
        self.val[key] = self.val.dtype(value)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __pos__(self):
        return explicit_operator(self.domain,self.val,bare=True,sym=self.sym,uni=self.uni,target=self.target)

    def __neg__(self):
        return explicit_operator(self.domain,-self.val,bare=True,sym=self.sym,uni=self.uni,target=self.target)

    def __abs__(self):
        return explicit_operator(self.domain,np.absolute(self.val),bare=True,sym=self.sym,uni=self.uni,target=self.target)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def transpose(self):
        """
            Computes the transposed matrix.

            Returns
            -------
            T : explicit_operator
                The transposed matrix.

        """
        return explicit_operator(self.domain,self.val.T,bare=True,sym=self.sym,uni=self.uni,target=self.target)

    def conjugate(self):
        """
            Computes the complex conjugated matrix.

            Returns
            -------
            CC : explicit_operator
                The complex conjugated matrix.

        """
        return explicit_operator(self.domain,np.conjugate(self.val),bare=True,sym=self.sym,uni=self.uni,target=self.target)

    def adjoint(self):
        """
            Computes the adjoint matrix.

            Returns
            -------
            A : explicit_operator
                The adjoint matrix.

        """
        return explicit_operator(self.target,np.conjugate(self.val.T),bare=True,sym=self.sym,uni=self.uni,target=self.domain)

    def inverse(self):
        """
            Computes the inverted matrix.

            Returns
            -------
            I : explicit_operator
                The inverted matrix.

        """
        ## check whether square matrix
        if(self.nrow()!=self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
        self._set_inverse()
        return explicit_operator(self.target,self._inv,bare=True,sym=self.sym,uni=self.uni,target=self.domain)

    def adjoint_inverse(self):
        """
            Computes the adjoint inverted matrix.

            Returns
            -------
            AI : explicit_operator
                The adjoint inverted matrix.

        """
        return self.inverse_adjoint()

    def inverse_adjoint(self):
        """
            Computes the inverted adjoint matrix.

            Returns
            -------
            IA : explicit_operator
                The inverted adjoint matrix.

        """
        ## check whether square matrix
        if(self.nrow()!=self.ncol()):
            raise ValueError(about._errors.cstring("ERROR: inverse ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
        self._set_inverse()
        return explicit_operator(self.target,np.conjugate(self._inv.T),bare=True,sym=self.sym,uni=self.uni,target=self.domain)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __add__(self,X): ## __add__ : self + X
        if(isinstance(X,operator)):
            if(self.domain!=X.domain):
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            sym = (self.sym and X.sym)
            uni = None
            if(isinstance(X,explicit_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                matrix = self.val+X.val
            elif(isinstance(X,diagonal_operator)):
                if(self.target.dim(split=False)!=X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring("ERROR: incompatible codomains."))
                matrix = self.val+np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                matrix = self.val+np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            if(self.nrow()!=self.ncol()):
                raise ValueError(about._errors.cstring("ERROR: identity ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
            sym = self.sym
            uni = None
            X = self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype)
            matrix = self.val+np.diag(X,k=0)
        elif(np.size(X)==np.size(self.val)):
            sym = None
            uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
            matrix = self.val+X
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.ncol())+" )."))
        return explicit_operator(self.domain,matrix,bare=True,sym=sym,uni=uni,target=self.target)

    __radd__ = __add__  ## __add__ : X + self

    def __iadd__(self,X): ## __iadd__ : self += X
        if(isinstance(X,operator)):
            if(self.domain!=X.domain):
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            self.sym = (self.sym and X.sym)
            self.uni = None
            if(isinstance(X,explicit_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                self.val += X.val
            elif(isinstance(X,diagonal_operator)):
                if(self.target.dim(split=False)!=X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring("ERROR: incompatible codomains."))
                self.val += np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                self.val += np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            if(self.nrow()!=self.ncol()):
                raise ValueError(about._errors.cstring("ERROR: identity ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
            self.uni = None
            X = self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype)
            self.val += np.diag(X,k=0)
        elif(np.size(X)==np.size(self.val)):
            self.sym = None
            self.uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
            self.val += X
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.ncol())+" )."))

        ## check flags
        self.sym,self.uni = self._check_flags(sym=self.sym,uni=self.uni)

        self._inv = None ## reset

        return self

    def __sub__(self,X): ## __sub__ : self - X
        if(isinstance(X,operator)):
            if(self.domain!=X.domain):
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            sym = (self.sym and X.sym)
            uni = None
            if(isinstance(X,explicit_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                matrix = self.val-X.val
            elif(isinstance(X,diagonal_operator)):
                if(self.target.dim(split=False)!=X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring("ERROR: incompatible codomains."))
                matrix = self.val-np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                matrix = self.val-np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            if(self.nrow()!=self.ncol()):
                raise ValueError(about._errors.cstring("ERROR: identity ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
            sym = self.sym
            uni = None
            X = self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype)
            matrix = self.val-np.diag(X,k=0)
        elif(np.size(X)==np.size(self.val)):
            sym = None
            uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
            matrix = self.val-X
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.ncol())+" )."))
        return explicit_operator(self.domain,matrix,bare=True,sym=sym,uni=uni,target=self.target)

    def __rsub__(self,X): ## __rsub__ : X - self
        if(isinstance(X,operator)):
            if(self.domain!=X.domain):
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            sym = (self.sym and X.sym)
            uni = None
            if(isinstance(X,explicit_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                matrix = X.val-self.val
            elif(isinstance(X,diagonal_operator)):
                if(self.target.dim(split=False)!=X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring("ERROR: incompatible codomains."))
                matrix = np.diag(X.diag(bare=True,domain=None),k=0)-self.val ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                matrix = np.tensordot(X.val,X.val,axes=0)-self.val
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            if(self.nrow()!=self.ncol()):
                raise ValueError(about._errors.cstring("ERROR: identity ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
            sym = self.sym
            uni = None
            X = self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype)
            matrix = np.diag(X,k=0)-self.val
        elif(np.size(X)==np.size(self.val)):
            sym = None
            uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
            matrix = X-self.val
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.ncol())+" )."))
        return explicit_operator(self.domain,matrix,bare=True,sym=sym,uni=uni,target=self.target)

    def __isub__(self,X): ## __isub__ : self -= X
        if(isinstance(X,operator)):
            if(self.domain!=X.domain):
                raise ValueError(about._errors.cstring("ERROR: inequal domains."))
            self.sym = (self.sym and X.sym)
            self.uni = None
            if(isinstance(X,explicit_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                self.val -= X.val
            elif(isinstance(X,diagonal_operator)):
                if(self.target.dim(split=False)!=X.target.dim(split=False))or(not self.target.check_codomain(X.target)):
                    raise ValueError(about._errors.cstring("ERROR: incompatible codomains."))
                self.val -= np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                if(self.target!=X.target):
                    raise ValueError(about._errors.cstring("ERROR: inequal codomains."))
                self.val -= np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            if(self.nrow()!=self.ncol()):
                raise ValueError(about._errors.cstring("ERROR: identity ill-defined for "+str(self.nrow())+" x "+str(self.ncol())+" matrices."))
            self.uni = None
            X = self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype)
            self.val -= np.diag(X,k=0)
        elif(np.size(X)==np.size(self.val)):
            self.sym = None
            self.uni = None
            X = np.array(X).reshape(self.val.shape)
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
            self.val -= X
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.ncol())+" )."))

        ## check flags
        self.sym,self.uni = self._check_flags(sym=self.sym,uni=self.uni)

        self._inv = None ## reset

        return self

    def _calc_mul(self,X,side): ## > multiplies self with X ...
        if(side==0): ## ... from right
            if(self._hidden[0]): ## hidden degrees of freedom
                return np.array([np.apply_along_axis(self.domain.calc_dot,0,X,np.conjugate(rr)) for rr in self.weight(rowpower=1,colpower=0,overwrite=False)])
            else:
                return np.dot(self.weight(rowpower=1,colpower=0,overwrite=False),X,out=None)
        elif(side==1): ## ... from left
            if(self._hidden[1]): ## hidden degrees of freedom
                return np.array([np.apply_along_axis(self.target.calc_dot,0,self.weight(rowpower=0,colpower=1,overwrite=False),np.conjugate(rr)) for rr in X])
            else:
                return np.dot(X,self.weight(rowpower=0,colpower=1,overwrite=False),out=None)
        else:
            raise ValueError(about._errors.cstring("ERROR: invalid input."))

    def __mul__(self,X): ## __mul__ : self * X
        if(isinstance(X,operator)):
            if(self.domain!=X.target):
                raise ValueError(about._errors.cstring("ERROR: incompatible spaces."))
            newdomain = X.domain
            sym = None
            uni = (self.uni and X.uni)
            if(isinstance(X,explicit_operator)):
                X = X.val
            elif(isinstance(X,diagonal_operator)):
                X = np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                X = np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            newdomain = self.domain
            sym = None
            uni = None
            X = np.diag(self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype),k=0)
        elif(np.size(X)==self.val.shape[1]**2):
            newdomain = self.domain
            sym = None
            uni = None
            X = np.array(X).reshape((self.val.shape[1],self.val.shape[1]))
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.nrow())+" )."))
        return explicit_operator(newdomain,self._calc_mul(X,0),bare=True,sym=sym,uni=uni,target=self.target)

    def __rmul__(self,X): ## __mul__ : X * self
        if(isinstance(X,operator)):
            if(X.domain!=self.target):
                raise ValueError(about._errors.cstring("ERROR: incompatible spaces."))
            newtarget = X.target
            sym = None
            uni = (self.uni and X.uni)
            if(isinstance(X,explicit_operator)):
                X = X.val
            elif(isinstance(X,diagonal_operator)):
                X = np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                X = np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            newtarget = self.target
            sym = None
            uni = None
            X = np.diag(self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype),k=0)
        elif(np.size(X)==self.val.shape[0]**2):
            newtarget = self.target
            sym = None
            uni = None
            X = np.array(X).reshape((self.val.shape[0],self.val.shape[0]))
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.ncol())+" x "+str(self.ncol())+" )."))

        return explicit_operator(self.domain,self._calc_mul(X,1),bare=True,sym=sym,uni=uni,target=newtarget)

    def __imul__(self,X): ## __imul__ : self *= X
        if(isinstance(X,operator)):
            if(self.domain!=X.target):
                raise ValueError(about._errors.cstring("ERROR: incompatible spaces."))
            if(self.domain!=X.domain):
                raise ValueError(about._errors.cstring("ERROR: incompatible operator."))
            self.sym = None
            self.uni = (self.uni and X.uni)
            if(isinstance(X,explicit_operator)):
                X = X.val
            elif(isinstance(X,diagonal_operator)):
                X = np.diag(X.diag(bare=True,domain=None),k=0) ## domain == X.domain
            elif(isinstance(X,vecvec_operator)):
                X = np.tensordot(X.val,X.val,axes=0)
            else:
                raise TypeError(about._errors.cstring("ERROR: unsupported or incompatible operator."))
        elif(np.size(X)==1):
            self.sym = None
            self.uni = None
            X = np.diag(self.domain.calc_weight(X*np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype),k=0)
        elif(np.size(X)==self.val.shape[1]**2):
            self.sym = None
            self.uni = None
            X = np.array(X).reshape((self.val.shape[1],self.val.shape[1]))
            if(np.all(np.isreal(X))):
                X = np.real(X).astype(min(self.domain.vol.dtype,self.target.vol.dtype))
            else:
                X = X.astype(min(self.domain.datatype,self.target.datatype))
        else:
            raise ValueError(about._errors.cstring("ERROR: dimension mismatch ( "+str(np.size(X))+" <> "+str(self.nrow())+" x "+str(self.nrow())+" )."))

        ## multiply
        self.val = self._calc_mul(X,0)

        ## check flags
        self.sym,self.uni = self._check_flags(sym=self.sym,uni=self.uni)

        self._inv = None ## reset

        return self

    def __div__(self,X):
        raise Exception(about._errors.cstring("ERROR: matrix division ill-defined."))

    __rdiv__ = __div__
    __idiv__ = __div__
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __itruediv__ = __idiv__

    def __pow__(self,x): ## __pow__(): self ** x
        if(not isinstance(x,(int,long))):
            raise TypeError(about._errors.cstring("ERROR: non-integer exponent."))
        elif(self.domain<>self.target):
            raise ValueError(about._errors.cstring("ERROR: incompatible spaces."))
        elif(x==0):
            return identity(self.domain)
        elif(x<0):
            return self.inverse().__pow__(-x)
        elif(x==1):
            return self
        else:
            matrix = self._calc_mul(self.val,0)
            for ii in xrange(x-1):
                matrix = self._calc_mul(matrix,0)
            return explicit_operator(self.domain,matrix,bare=True,sym=None,uni=self.uni,target=self.target)

    def __rpow__(self,X): ## __pow__(): X ** self
        raise Exception(about._errors.cstring("ERROR: matrix exponential ill-defined."))

    def __ipow__(self,x): ## __pow__(): self **= x
        if(not isinstance(x,(int,long))):
            raise TypeError(about._errors.cstring("ERROR: non-integer exponent."))
        elif(self.domain<>self.target):
            raise ValueError(about._errors.cstring("ERROR: incompatible spaces."))
        elif(x==0):
            self.val = np.diag(self.domain.calc_weight(np.ones(self.domain.dim(split=False),dtype=np.int,order='C'),power=-1).astype(self.val.dtype),k=0)
        elif(x<0):
            self.val = (self.inverse().__pow__(-x)).val
        elif(x==1):
            pass
        else:
            matrix = self._calc_mul(self.val,0)
            for ii in xrange(x-1):
                matrix = self._calc_mul(matrix,0)
            self.val = matrix

        ## check flags
        self.sym,self.uni = self._check_flags(sym=None,uni=self.uni)

        self._inv = None ## reset

        return self

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_plot(self,X,title="",vmin=None,vmax=None,unit="",norm=None,cmap=None,cbar=True,**kwargs):
        """
            Creates a plot of the matrix according to the given specifications.

            Parameters
            ----------
            X : numpy.ndarray
                Array containing the matrix.

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
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).

        """
        if(not pl.isinteractive())and(not bool(kwargs.get("save",False))):
            about.warnings.cprint("WARNING: interactive mode off.")

        if(vmin is None):
            vmin = np.min(X,axis=None,out=None)
        if(vmax is None):
            vmax = np.max(X,axis=None,out=None)
        if(norm=="log")and(vmin<=0):
            raise ValueError(about._errors.cstring("ERROR: nonpositive value(s)."))

        s_ = np.array([X.shape[1]/max(X.shape),X.shape[0]/max(X.shape)*(1.0+0.159*bool(cbar))])
        fig = pl.figure(num=None,figsize=(6.4*s_[0],6.4*s_[1]),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.06/s_[0],0.06/s_[1],1.0-0.12/s_[0],1.0-0.12/s_[1]])
        if(norm=="log"):
            n_ = ln(vmin=vmin,vmax=vmax)
        else:
            n_ = None
        sub = ax0.pcolormesh(X[::-1,:],cmap=cmap,norm=n_,vmin=vmin,vmax=vmax)
        ax0.set_xlim(0,X.shape[1])
        ax0.set_xticks([],minor=False)
        ax0.set_ylim(0,X.shape[0])
        ax0.set_yticks([],minor=False)
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

        if(bool(kwargs.get("save",False))):
            fig.savefig(str(kwargs.get("save")),dpi=None,facecolor=None,edgecolor=None,orientation='portrait',papertype=None,format=None,transparent=False,bbox_inches=None,pad_inches=0.1)
            pl.close(fig)
        else:
            fig.canvas.draw()

    def plot(self,**kwargs):
        """
            Creates a plot of the matrix according to the given specifications.

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
            cmap : matplotlib.colors.LinearSegmentedColormap, *optional*
                Color map to be used for two-dimensional plots (default: None).
            cbar : bool, *optional*
                Whether to show the color bar or not (default: True).
            save : string, *optional*
                Valid file name where the figure is to be stored, by default
                the figure is not saved (default: False).

        """
        interactive = pl.isinteractive()
        pl.matplotlib.interactive(not bool(kwargs.get("save",False)))

        if(np.any(np.iscomplex(self.val))):
            about.infos.cprint("INFO: absolute values and phases are plotted.")
            if(kwargs.has_key("title")):
                title = kwargs.get("title")+" "
                kwargs.__delitem__("title")
            else:
                title = ""
            self.get_plot(np.absolute(self.val),title=title+"(absolute)",**kwargs)
            if(kwargs.has_key("vmin")):
                kwargs.__delitem__("vmin")
            if(kwargs.has_key("vmin")):
                kwargs.__delitem__("vmax")
            if(kwargs.has_key("unit")):
                kwargs["unit"] = "rad"
            if(kwargs.has_key("norm")):
                kwargs["norm"] = None
            if(not kwargs.has_key("cmap")):
                kwargs["cmap"] = pl.cm.hsv_r
            self.get_plot(np.angle(self.val,deg=False),title=title+"(phase)",vmin=0,vmax=6.28319,**kwargs) ## vmax == 2 pi
        else:
            self.get_plot(np.real(self.val),**kwargs)

        pl.matplotlib.interactive(interactive)

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __repr__(self):
        return "<nifty.explicit_operator>"

##-----------------------------------------------------------------------------

## IDEA: explicit_probing

















