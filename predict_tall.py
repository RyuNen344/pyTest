#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#解析解
def fit_line(x,t):
    mx = np.mean(x)
    mt = np.mean(t)
    mtx = np.mean(t * x)
    mxx = np.mean(x * x)
    w0 = (mtx - mt* mx) / (mxx - mx**2)
    w1 = mt -w0 * mx
    return np.array([w0,w1])

def show_line(w):
    xb = np.linspace(X_min,X_max,100)
    y = w[0] * xb + w[1]
    plt.plot(xb,y,color=(.5,.5,.5),linewidth=4)

    #平均誤差関数
def mse_line(x,t,w):
    y = w[0] * x + w[1]
    mse= np.mean((y-t)**2)
    return mse

def show_data2(ax,x0,x1,t):
    for i in range(len(x0)):
        ax.plot([x0[i],x0[i]],[x1[i],x1[i]],[120,t[i]],color='white')
        ax.plot(x0,x1,t,'o',color='cornflowerblue',markeredgecolor='white',markersize=6,markeredgewidth=0.5)
        ax.view_init(elev=35,azim=-75)

#面の表示
def show_plane(ax,w):
    px0 = np.linspace(x0_min,x0_max,5)    
    px1 = np.linspace(x1_min,x1_max,5)
    px0,px1 = np.meshgrid(px0,px1)
    y = w[0]*px + w[1] * px1 + w[2]
    ax.plot_surface(px0,px1,y,rstride=1,cstride=1,alpha = 0.3,color="bule",edgecolor="black")

#面のMSE
def mse_plane(x0,x1,t,w):
    y = w[0] * x0 + w[1] * x1 + w[2]
    mse = np.mean((y - t)**2)
    return mse

np.random.seed(seed=1)
X_min = 4
X_max = 30
X_n = 16
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170,108,0.2]
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
X0 = X
X0_min = 5
X0_max = 30
X1 = 23 * (T / 100)**2 + 2 * np.random.randn(X_n)
X1_min = 40
X1_max = 75

# print(np.round(X0,2))
# print(np.round(X1,2))
# print(np.round(T,2))


W = fit_line(X,T)
print("w0 = {0:.3f},w1 = {1:.3f}".format(W[0],W[1]))
mse = mse_line(X,T,W)
print("SD={0:.3f}cm".format(np.sqrt(mse)))
ax = plt.subplot(1,1,1,projection='3d')
show_data2(ax,X0,X1,T)
plt.show()