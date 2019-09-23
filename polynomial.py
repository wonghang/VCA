#!/usr/bin/python3
# -*- coding: utf-8 -*-
from itertools import chain
import copy
import operator as op

__all__ = ['Polynomial','PolynomialRing']
#
# a very simple polynomial ring implementation over K using python dict
# Polynomial(5,complex) will represent the ring C[x1,x2,x3,x4,x5]
#
PRINT_INDET1="xyz"
PRINT_INDET2="abcdefghijklmnopqrstuvwxyz"
PRINT_SUPERSCRIPT='\u2070\xb9\xb2\xb3\u2074\u2075\u2076\u2077\u2078\u2079'

def _add_tuple(t1,t2):
    x = list(t1)
    for i in range(len(t1)):
        x[i] = x[i] + t2[i]
    return tuple(x)

def _sup_int(n):
    return "".join([PRINT_SUPERSCRIPT[int(c)] for c in str(n)])

def _sub_min_nonzero(terms,k):
    m = None
    for M in terms.keys():
        if M[k] and (m is None or m > M[k]):
            m = M[k]
    return m

def _eval_poly_r(x,terms):
    rcount = list(map(op.methodcaller('count',0),list(zip(*list(terms.keys())))))
    K = min(rcount)

    if K >= len(terms)-1:
        r = 0
        for (M,coeff) in terms.items():
            for (b,n) in zip(x,M):
                if n == 0: continue
                elif n == 1: coeff *= b
                else: coeff *= b**n
            r += coeff
        return r
    else:
        k = rcount.index(K)
        indeps = {}
        depend = {}
        n = _sub_min_nonzero(terms,k) 
        for (M,coeff) in terms.items():
            if M[k] != 0:
                depend[M[:k]+(M[k]-n,)+M[k+1:]] = coeff
            else:
                indeps[M] = coeff
        r = x[k]**n * _eval_poly_r(x,depend)
        if indeps:
            return r + _eval_poly_r(x,indeps)
        else:
            return r

class PolynomialRingMetaClass(type):
    _instances = {}
    def __call__(cls,n,K=float):
        if (n,K) not in cls._instances:
            cls._instances[(n,K)] = super(PolynomialRingMetaClass,cls).__call__(n,K)
        return cls._instances[(n,K)]

class PolynomialRing(metaclass=PolynomialRingMetaClass):
    def __init__(self,n,K):
        self.n = n
        self.K = K

        self.KZERO = K(0)
        self.KONE = K(1)

        tmp = [0] * n
        self.ZERO_KEY = tuple(tmp)
        k = []
        for i in range(n):
            tmp[i] = 1
            k.append(tuple(tmp))
            tmp[i] = 0
        self.TERM_KEY = k

        self.tcache = {}

    def strfmt_term(self,k):
        if k in self.tcache:
            return self.tcache[k]

        if self.n <= len(PRINT_INDET1):
            p = PRINT_INDET1
        elif self.n <= len(PRINT_INDET2):
            p = PRINT_INDET2
        else:
            p = None

        if p is not None:
            r = []
            for (i,s) in enumerate(k):
                if s == 0:
                    continue
                elif s == 1:
                    r.append(p[i])
                else:
                    r.append("%s%s" % (p[i],_sup_int(s)))
#                    r.append("%s^%d" % (p[i],s))
            r = "".join(r)
        else:
            r = []
            for (i,s) in enumerate(k):
                if s == 0:
                    continue
                elif s == 1:
                    r.append("x%d" % i)
                else:
                    r.append("%s%s" % (i,_sup_int(s)))
#                    r.append("x%d^%d" % (i,s))
            r = "".join(r)

        self.tcache[k] = r
        return r

    def __eq__(self,other):
        if self.n == other.n and self.K == other.K:
            return True
        else:
            return False

    def __str__(self):
        return "<PolynomialRing with %d indeterminate over %s at 0x%x>" % (self.n,str(self.K),id(self))

    def ZERO(self):
        return Polynomial(self)

    def ONE(self):
        f = Polynomial(self)
        f.set_const(self.KONE)
        return f

    def constant_term(self,c):
        f = Polynomial(self)
        f._dat[self.ZERO_KEY] = self.K(c)
        return f

    def monomial(self,i):
        f = Polynomial(self)
        f._dat[self.TERM_KEY[i]] = self.KONE
        return f

    __call__=constant_term

class Polynomial:
    def op_check_domain(f):
        def wrapper(self,other):
            if self.R != other.R:
                raise ValueError("Operation on two different ring")
            return f(self,other)
        return wrapper

    def require_immutable(f):
        def wrapper(self,*a,**ka):
            if self._mutable:
                raise NotImplementedError("Operation require immutable copy")
            else:
                return f(self,*a,**ka)
        return wrapper

    def require_mutable(f):
        def wrapper(self,*a,**ka):
            if not self._mutable:
                raise NotImplementedError("Operation require mutable copy")
            else:
                return f(self,*a,**ka)
        return wrapper

    def __init__(self,R,dat=None):
        self.R = R
        if dat is None:
            self._dat = {}
        else:
            self._dat = dat

        self._mutable = True
        self._hash = None

    def __getstate__(self):
        return (self.R.n,self.R.K,self._dat,self._mutable)

    def __setstate__(self,state):
        (n,K,self._dat,self._mutable) = state
        self.R = PolynomialRing(n,K)
        self._hash = None

    def __str__(self):
        R = self.R
        r = []
        for k in sorted(list(self._dat.keys()),reverse=True):
            v = self._dat[k]
            if v == R.KZERO:
                continue
            elif v == R.KONE:
                s = R.strfmt_term(k)
                if len(s) == 0: s = '1'
            else:
                s = str(v) + R.strfmt_term(k)
            r.append(s)
        if len(r) == 0:
            return '0'
        else:
            return " + ".join(r)

    def __repr__(self):
        return "<" + str(self) + ">"

    def immutable(self):
        if self._mutable:
            self.reduce()
            self._mutable = False
        return self

    def immutable_copy(self):
        c = copy.copy(self)
        return c.immutable()

    def mutable_copy(self):
        c = copy.copy(self)
        c._mutable = True
        return c

    @require_immutable
    def __hash__(self):
        if self._hash is not None:
            return self._hash
        else:
            d = self._dat
            h = hash(",".join(("%s:%d" % (k,hash(d[k])) for k in sorted(self._dat.keys()))))
            h += id(self.R)
            self._hash = h
            return self._hash

    def coeff(self,*x):
        R = self.R
        if len(x) != R.n:
            raise ValueError("Invalid number of indeterminate")
        return self._dat.get(x,R.KZERO)
 
    def const(self):
        R = self.R
        return self._dat.get(R.ZERO_KEY,R.KZERO)

    @require_mutable
    def set_coeff(self,c,*x):
        R = self.R
        if len(x) != R.n:
            raise ValueError("Invalid number of indeterminate")
        self._dat[x] = self._dat.get(x,R.KZERO) + c
        self.reduce()

    @require_mutable
    def set_const(self,c):
        R = self.R
        self._dat[R.ZERO_KEY] = c

    @require_mutable
    def reduce(self):
        self._dat = dict((k,v) for (k,v) in self._dat.items() if v != self.R.KZERO)

    def __call__(self,*x):                
        if len(x) != self.R.n:
            raise ValueError("Invalid number of indeterminate")

        return _eval_poly_r(x,self._dat)

    @op_check_domain
    def __add__(self,other):
        R = self.R

        odat = other._dat
        ndat = copy.copy(self._dat)
        for (k,v) in odat.items():
            ndat[k] = ndat.get(k,R.KZERO) + v

        N = Polynomial(R,ndat)
        N.reduce()
        return N
        
    @op_check_domain
    def __sub__(self,other):
        R = self.R

        odat = other._dat
        ndat = copy.copy(self._dat)
        for (k,v) in odat.items():
            ndat[k] = ndat.get(k,R.KZERO) - v

        N = Polynomial(R,ndat)
        N.reduce()
        return N

    @op_check_domain
    def __mul__(self,other):
        R = self.R

        dat = self._dat
        odat = other._dat
        ndat = {}
        for (k1,v1) in dat.items():
            for (k2,v2) in odat.items():
                k = _add_tuple(k1,k2)
                ndat[k] = ndat.get(k,R.KZERO) + v1*v2
        N = Polynomial(R,ndat)
        N.reduce()
        return N

    @op_check_domain
    def __eq__(self,other):
        R = self.R
        dat = self._dat
        odat = other._dat
        for k in chain(dat.keys(),odat.keys()):
            if dat.get(k,R.KZERO) != odat.get(k,R.KZERO):
                return False

        return True

    def unsafe_power(self,n):
        if n == 0:
            return self.R.ONE()
        elif n == 1:
            return copy.copy(self)
        else:
            (d,r) = divmod(n,2)
            x = self.unsafe_power(d)
            x = x*x
            if r == 0:
                return x
            else:
                return x*self
            
    def __pow__(self,n):
        if isinstance(n,int) and n >= 0:
            return self.unsafe_power(n)
        else:
            raise ValueError("Power of polynomial must be positive integer")

    def degree(self):
        return max((sum(x) for x in self._dat.keys()))

if __name__ == "__main__":
    R = PolynomialRing(3,float)

    x = R.monomial(0)
    y = R.monomial(1)
    z = R.monomial(2)

    e = R.ONE()

    f = (x+y+z)**10
    g = (x+y+z)**3 + (x+y+e)**3
    print(f)
    print(f(1,2,3) - (1+2+3)**10)
    print(f(4.123,5.111,6.333) - (4.123+5.111+6.333)**10)
