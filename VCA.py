import numpy as np
from numpy import linalg
from polynomial import *

__all__ = ['VCA']

# An implementation of Vanishing Component Analysis in the following paper
# Livni, R., Lehavi, D., Schein, S., Nachliely, H., Shalev-Shwartz, S., & Globerson, A. (2013).
# Vanishing Component Analysis. 
# In Proceedings of the 30th International Conference on Machine Learning (ICML-13) (pp. 597-605).

class VCA:
    def __init__(self,Sm,epsilon):
        self.Sm = Sm
        self.S_row = Sm.shape[0]
        self.S_col = Sm.shape[1]
        self.epsilon = epsilon
        self.Ring = PolynomialRing(self.S_col)

        self.f_F = None
        self.f_V = None
        
    def find_range_null(self,F,C,print_D):
        Sm = self.Sm
        k = len(C)
        m = self.S_row
        Ring = self.Ring
        e = self.epsilon

        def _apply_Sm(f):
            ret = np.empty((m,),dtype=np.float64)
            for j in range(m):
                ret[j] = f(*Sm[j])
            return ret
            
        C_t = []
        for (i,f_i) in enumerate(C): # C is okay for both list or set object
            f_ti = f_i
            for g in F: # F should be a set, and g is immutable polynomial
                dotprod = sum([f_i(*Sm[r]) * g(*Sm[r]) for r in range(m)])
                f_ti -= g * Ring(dotprod) # convert the dot-product to an element in the ring
            C_t.append(f_ti)
        # construct matrix A
        A = np.empty(shape=(k,m))
        for i in range(k):
            f_ti = C_t[i]
            A[i] = _apply_Sm(f_ti)
        # perform SVD decomposition
        (L,D,U) = linalg.svd(A.T)
        if print_D:
            print("largest singular value %g, lowest singular value %g" % (np.max(D),np.min(D)))

        l = len(D)
        V1 = set()
        F1 = set()
        for i in range(k):
            g = Ring.ZERO()
            for j in range(k):
                g += Ring(U[i,j])*C_t[j] # the paper is U[j,i], but A = LDU^T, linalg return U as A = LDU
            g.immutable()
            if i < l and D[i] > e:
                f = g * Ring(1.0/linalg.norm(_apply_Sm(g)))
                f.immutable()
                F1.add(f)
            else:
                V1.add(g)
        return (F1,V1)

    def fit(self,print_D=False):
        find_range_null=self.find_range_null

        Ring = self.Ring
        m = self.S_row
        n = self.S_col

        inv_sqrt_m = 1.0/np.sqrt(m)
        F = set()
        F.add(Ring(inv_sqrt_m).immutable())
        V = set()

        C = [Ring.monomial(i) for i in range(n)]
        (F1,V1) = find_range_null(F,C,print_D=print_D)
        F = F.union(F1)
        V = V.union(V1)
        t = 2
        F_t = F1   # F_t is F_{t-1} in the paper
        while F_t:
            C_t = set()
            for g in F_t:
                for h in F1:
                    C_t.add((g*h).immutable())
            # C_t is empty if Ft is empty, checked in while-loop
            (F_t,V_t) = find_range_null(F,C_t,print_D=print_D)
            F = F.union(F_t)
            V = V.union(V_t)

        self.f_F = list(F)
        self.f_V = list(V)
        
    def transform(self,X):
        # Theorem 7.1
        m = X.shape[0]
        V = self.f_V
        n = len(V)
        ret = np.empty((m,n),dtype=np.float64)
        for i in range(m):
            for j in range(n):
                ret[i,j] = V[j](*X[i])
        ret = np.abs(ret)
        return ret
        
if __name__ == "__main__":
    pass
