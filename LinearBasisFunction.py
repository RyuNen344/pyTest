#%%

import numpy as np
import matplotlib.pyplot as plt

outfile = np.load('ch5_data.npz')
X = outfile['X']
X_min = outfile['X_min']
X_max = outfile['X_max']
X_n = outfile['X_n']
T = outfile['T']

def gauss(x,mu,s):
    return np.exp(-(x-mu)**2/(2*s**2))

def gauss_func(w,x):
    m = len(w) - 1
    mu = np.linspace(5,30,m)
    s = mu[1] - mu[0]
    y = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gauss(x,mu[j],s)
    y = y + w[m]
    return y
    
#線形基底関数モデルMSE
def mse_gauss_func(x,t,w):
    y = gauss_func(w,x)
    mse = np.mean((y-t)**2)
    return mse

#線形基底関数モデル　厳密解
def fit_gauss_fun(x,t,m):
    mu = np.linspace(5,30,m)
    s = mu[1] - mu[0]
    n = x.shape[0]
    psl = np.ones((n,m+1))
    for j in range(m):
        psl[:,j]=gauss(x,mu[j],s)
    psl_T = np.transpose(psl)

    b = np.linalg.inv(psl_T.dot(psl))
    c  = b.dot(psl_T)
    w = c.dot(t)
    return w

def show_gauss_func(w):
    xb = np.linspace(X_min,X_max,100)
    y = gauss_func(w,xb)
    plt.plot(xb,y,c=[.5,.5,.5],lw=4)

M = 6
plt.figure(figsize=(4,4))
W = fit_gauss_fun(X,T,M)
show_gauss_func(W)
plt.plot(X,T,marker="o",linestyle="None",color="cornflowerblue",markeredgecolor="white")
plt.grid(True)
plt.xlim(X_min,X_max)
mse = mse_gauss_func(X,T,W)
print("W="+str(np.round(W,1)))
print("SD={0:.2f}cm".format(np.sqrt(mse)))
plt.show()
