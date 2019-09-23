#!/usr/bin/python3
import csv
import numpy as np
from numpy import linalg
from sklearn import svm
from sklearn.metrics import confusion_matrix
from VCA import *

def generate_data(N=1000):
    X = np.empty(shape=(N,3),dtype=np.float64)
    tY = np.empty(shape=(N,),dtype=np.float64)
    for k in range(N):
        x = np.random.uniform(size=(3,))
        X[k] = x
        tY[k] = x[0]*x[0] + x[0]*x[1] - x[2]*x[2] + x[1]*x[2]
    
    Y = np.zeros(shape=(N,),dtype=int)

    M = np.max(tY)
    m = np.min(tY)
    d = (M-m)/4
    
    d0 = m + d
    d1 = m + d*2
    d2 = m + d*3

    Y[np.logical_and(d0 < tY,tY <= d1)] = 1
    Y[d2 < tY] = 1

    idx = np.random.choice(N,size=N//100)
    Y[idx] = ~Y[idx] # make some wrong label
    
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X = (X-mu)/sigma
    return (X,Y,(mu,sigma))

def thorsten_joachims_heuristic_C(X):
    m = X.shape[0]
    (U,S,V) = linalg.svd(X,full_matrices=False)
    Z = U@np.diag(S*S)@U.T
    C = 1/np.mean(np.sqrt(S))
    return C

def evaluate_header():
    print("{:<30s} | {:<20s} | {:<20s}".format("Model","In-sample accuracy","Out-sample accuracy"))
    
def evaluate_performance(name,X_train,Y_train,X_test,Y_test,model):
    in_R2 = model.score(X_train,Y_train)
    out_R2 = model.score(X_test,Y_test)
    print("{:<30s} | {:<20f} | {:<20f}".format(name,in_R2,out_R2))
    # Y_predict = model.predict(X)
    # con = confusion_matrix(Y_true,Y_predict)
    # print("%d %d\n%d %d" % (con[0,0],con[0,1],con[1,0],con[1,1]))
    
def main():
    # (X,Y,scale) = load_data()
    (X,Y,scale) = generate_data()

    total_size = X.shape[0]
    test_size = total_size // 3
    train_size = total_size - test_size

    X_train = X[0:train_size]
    Y_train = Y[0:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]

    print("Total size %d" % total_size)
    print("Train size %d" % train_size)
    print("Test size %d" % test_size)
    print("---------------------")
    C = thorsten_joachims_heuristic_C(X_train)
    tol = 1e-4
    random_state = 12345
    evaluate_header()
    # linear SVC
    clf = svm.LinearSVC(C=C,random_state=random_state,dual=False,tol=tol)
    clf.fit(X_train,Y_train)
    evaluate_performance("Linear SVC",X_train,Y_train,X_test,Y_test,clf)
    # Kernel RBF SVC
    clf = svm.SVC(C=C,kernel='rbf',gamma='scale',random_state=random_state,tol=tol)
    clf.fit(X_train,Y_train)
    evaluate_performance("Kernel (RBF) SVC",X_train,Y_train,X_test,Y_test,clf)
    # Kernel Poly SVC
    for deg in [2,3,4,5]:
        clf = svm.SVC(C=C,kernel='poly',gamma='scale',degree=deg,random_state=random_state,tol=tol)
        clf.fit(X_train,Y_train)
        evaluate_performance("Kernel (Poly, deg=%d) SVC" % deg,X_train,Y_train,X_test,Y_test,clf)

    print("Computing VCA...")
    epsilon = 0.1
    vca = VCA(X_train,epsilon=epsilon)
    vca.fit(print_D=True)
    print("# of vanishing components %d, using threshold=%g" % (len(vca.f_V),epsilon))

    X_vca_train = vca.transform(X_train)
    # scale the result again
    mu = np.mean(X_vca_train,axis=0)
    sigma = np.std(X_vca_train,axis=0)
    X_vca_train = (X_vca_train-mu)/sigma

    X_vca_test = vca.transform(X_test)
    X_vca_test = (X_vca_test-mu)/sigma
    
    C = thorsten_joachims_heuristic_C(X_vca_train)
    clf = svm.LinearSVC(C=C,random_state=random_state,dual=False,tol=tol)
    clf.fit(X_vca_train,Y_train)
    evaluate_header()
    evaluate_performance("Linear SVC + VCA",X_vca_train,Y_train,X_vca_test,Y_test,clf)
    
if __name__ == "__main__":
    np.random.seed(12345)
    main()
    
        
