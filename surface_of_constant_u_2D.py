import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm      
from matplotlib.widgets import Slider

vals = np.linspace(-2,2,100)
xgrid, ygrid = np.meshgrid(vals,vals)   

def f(x, y, argWx, argWy, argomega, u):
    
    ret = np.transpose([argWx*x + argWy*y + 2*argomega*x*y, argWx*x + argWy*y + 2*argomega*x*y, argWx*x + argWy*y + 2*argomega*x*y] )
    res = argWx*x + argWy*y + 2*argomega*x*y
    for xidx, x in enumerate(res):
        for yidx, y in enumerate(x):
            if res[yidx, xidx] > u: #Switched coordinates as a lousy correction to the transpose made earlier
                ret[xidx, yidx, 0]= 0
                ret[xidx, yidx, 1]= 1
            elif res[yidx, xidx] < u:
                ret[xidx, yidx, 1]= 0
                ret[xidx, yidx, 0]= 1
            #if res[yidx, xidx] > u - 0.1  and res[xidx, yidx] < u + 0.1 : #Chosen such that threshold scales with omega
            #    res[xidx, yidx] = -10

    return ret

Wx = 1
Wy = 1
omega = 0
u = 0.5

ax = plt.subplot(111)
plt.subplots_adjust(left=0.15, bottom=0.5)
fig = plt.imshow(f(xgrid, ygrid, Wx, Wy, omega, u), cm.gray)

fig.axes.get_xaxis().set_visible(True)
fig.axes.get_yaxis().set_visible(True)
plt.xlabel("x1")
plt.ylabel("x2")


axomega = plt.axes([0.15, 0.1, 0.65, 0.03])
somega = Slider(axomega, 'omega', -5, 5, valinit=omega)
axWx = plt.axes([0.15, 0.3, 0.65, 0.03])
sWx = Slider(axWx, 'Wx', -1, 1, valinit=Wx)
axWy = plt.axes([0.15, 0.2, 0.65, 0.03])
sWy = Slider(axWy, 'Wy', -1, 1, valinit=Wy)

axu = plt.axes([0.15, 0.4, 0.65, 0.03])
su = Slider(axu, 'u', -10, 10, valinit=u)

def update(val):
    fig.set_data(f(xgrid, ygrid, sWx.val, sWy.val, somega.val, su.val))
    #line, = plt.plot(xgrid[0], plotIsoLine(xgrid[0],sWx.val, sWy.val, somega.val, su.val)[1])

somega.on_changed(update)
sWx.on_changed(update)
sWy.on_changed(update)
sWy.on_changed(update)
su.on_changed(update)

plt.sca(ax)
t = np.linspace(-2, 2, 100)
def plotIsoLine(x, argWx, argWy, argomega, u):
    if argomega != 0:
        print((u - argWx - argWy ) / (2 * argomega) / (x + argWx/(2 * argomega)) + (argWy/(2 * argomega)))
        return (x,(u - argWx - argWy ) / (2 * argomega) / (x + argWx/(2 * argomega)) + (argWy/(2 * argomega)))
    else:
        print("omega is 0")
        return t, t
line, = plt.plot(t, plotIsoLine(t,sWx.val, sWy.val, somega.val, su.val)[1],  lw = 1, color = 'blue')

plt.show()