import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm      
from matplotlib.widgets import Slider

vals = np.linspace(-2,2,100)
xgrid, ygrid = np.meshgrid(vals,vals)   

def f(x, y, argWsx, argWsy, argomegas, argWtx, argWty, argomegat):
    
    ret = np.transpose([argWsx*x + argWsy*y + 2*argomegas*x*y, argWsx*x + argWsy*y + 2*argomegas*x*y, argWsx*x + argWsy*y + 2*argomegas*x*y] )/ 2
    ress = argWsx*x + argWsy*y + 2*argomegas*x*y 
    rest = argWtx*x + argWty*y + 2*argomegat*x*y
    for xidx, x in enumerate(ress):
        for yidx, y in enumerate(x):
            if ress[yidx, xidx]*rest[yidx, xidx] > 0: #Switched coordinates as a lousy correction to the transpose made earlier
                ret[xidx, yidx, 0]= 0
                ret[xidx, yidx, 1]= 1
            elif ress[yidx, xidx]*rest[yidx, xidx] < 0:
                ret[xidx, yidx, 1]= 0
                ret[xidx, yidx, 0]= 1
            #if res[yidx, xidx] > u - 0.1  and res[xidx, yidx] < u + 0.1 : #Chosen such that threshold scales with omega
            #    res[xidx, yidx] = -10

    return ret

Wsx = 1
Wsy = 1
omegas = 0
us = 0.5

Wtx = 1
Wty = 1
omegat = 0
ut = 0.5

ax = plt.subplot(111)
plt.subplots_adjust(left=0.15, bottom=0.55)
fig = plt.imshow(f(xgrid, ygrid, Wsx, Wsy, omegas, Wtx, Wty, omegat), cm.gray, origin = 'lower')
plt.axis("off")
fig.axes.get_xaxis().set_visible(True)
fig.axes.get_yaxis().set_visible(True)
plt.xlabel("x1")
plt.ylabel("x2")


axomegas = plt.axes([0.15, 0.1, 0.3, 0.03])
somegas = Slider(axomegas, 'omegas', -5, 5, valinit=omegas)
axWsx = plt.axes([0.15, 0.3, 0.3, 0.03])
sWsx = Slider(axWsx, 'Wsx', -1, 1, valinit=Wsx)
axWsy = plt.axes([0.15, 0.2, 0.3, 0.03])
sWsy = Slider(axWsy, 'Wsy', -1, 1, valinit=Wsy)


axomegat = plt.axes([0.6, 0.1, 0.3, 0.03])
somegat = Slider(axomegat, 'omegat', -5, 5, valinit=omegat)
axWtx = plt.axes([0.6, 0.3, 0.3, 0.03])
sWtx = Slider(axWtx, 'Wtx', -1, 1, valinit=Wtx)
axWty = plt.axes([0.6, 0.2, 0.3, 0.03])
sWty = Slider(axWty, 'Wty', -1, 1, valinit=Wty)

def update(val):
    fig.set_data(f(xgrid, ygrid, sWsx.val, sWsy.val, somegas.val,
                sWtx.val, sWty.val, somegat.val))

somegas.on_changed(update)
sWsx.on_changed(update)
sWsy.on_changed(update)
sWsy.on_changed(update)

somegat.on_changed(update)
sWtx.on_changed(update)
sWty.on_changed(update)
sWty.on_changed(update)

plt.sca(ax)

plt.show()