# VCA

VCA: An implementation of Vanishing Component Analysis in python

The algorithm is based on the following paper:

```Livni, Roi, et al. "Vanishing component analysis." International Conference on Machine Learning. 2013.```

It is just a test of the above algorithm and compare it with Kernel SVM (especially polynomial kernel).

The code requires [scikit-learn](https://scikit-learn.org/).

The code contains a simple symbolic polynomial code. 

I think it is quite slow and may not be numerical stable.
 
The polynomial evaluation code is a naive implementation. Horner's method or more sophisticated algorithm such as Polynomial Homotopy Continuation Methods should be used.

To run the test code:

```
$ python3 test_VCA.py
Total size 1000
Train size 667
Test size 333
---------------------
Model                          | In-sample accuracy   | Out-sample accuracy 
Linear SVC                     | 0.605697             | 0.618619            
Kernel (RBF) SVC               | 0.863568             | 0.882883            
Kernel (Poly, deg=2) SVC       | 0.677661             | 0.639640            
Kernel (Poly, deg=3) SVC       | 0.727136             | 0.714715            
Kernel (Poly, deg=4) SVC       | 0.667166             | 0.630631            
Kernel (Poly, deg=5) SVC       | 0.704648             | 0.690691            
Computing VCA...
largest singular value 26.6546, lowest singular value 25.2696
largest singular value 0.0539178, lowest singular value 0.025033
# of vanishing components 6, using threshold=0.1
Model                          | In-sample accuracy   | Out-sample accuracy 
Linear SVC + VCA               | 0.689655             | 0.669670            
```
