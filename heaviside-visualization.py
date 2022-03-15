import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm      
from matplotlib.widgets import Slider

vals = np.linspace(-2,2,100)
xgrid, ygrid = np.meshgrid(vals,vals)  

'''
This code serves to give an intuition for how the parameters of a MP student-teacher scenario determine
the whether the student and teacher give the same output label. Note that for the perceptron case, everything becomes linear.
u denotes a threshold, and may be set to 0. 
This code also illustrates how the delta-function approach using activation space becomes far more complicated. 
''' 

def f(x, y, argWs1, argWs2, argomegas, argWt1, argWt2, argomegat):
    
    ret = np.transpose([argWs1*x + argWs2*y + 2*argomegas*x*y, argWs1*x + argWs2*y + 2*argomegas*x*y, argWs1*x + argWs2*y + 2*argomegas*x*y] )/ 2
    ress = argWs1*x + argWs2*y + 2*argomegas*x*y 
    rest = argWt1*x + argWt2*y + 2*argomegat*x*y
    for xidx, x in enumerate(ress):
        for yidx, y in enumerate(x):
            if ress[yidx, xidx]*rest[yidx, xidx] > 0: #Switched coordinates as a lousy correction to the transpose made earlier
                ret[xidx, yidx, 0]= 0
                ret[xidx, yidx, 1]= 1
            elif ress[yidx, xidx]*rest[yidx, xidx] < 0:
                ret[xidx, yidx, 1]= 0
                ret[xidx, yidx, 0]= 1

    return ret

Ws1 = 1
Ws2 = 1
omegas = 0
us = 0.5

Wt1 = 1
Wt2 = 1
omegat = 0
ut = 0.5

ax = plt.subplot(111)
plt.subplots_adjust(left=0.15, bottom=0.55)
fig = plt.imshow(f(xgrid, ygrid, Ws1, Ws2, omegas, Wt1, Wt2, omegat), cm.gray, origin = 'lower')
plt.axis("off")
fig.axes.get_xaxis().set_visible(True)
fig.axes.get_yaxis().set_visible(True)
plt.xlabel("x1")
plt.ylabel("x2")


axomegas = plt.axes([0.15, 0.1, 0.3, 0.03])
somegas = Slider(axomegas, 'omegas', -5, 5, valinit=omegas)
axWs1 = plt.axes([0.15, 0.3, 0.3, 0.03])
sWs1 = Slider(axWs1, 'Ws1', -1, 1, valinit=Ws1)
axWs2 = plt.axes([0.15, 0.2, 0.3, 0.03])
sWs2 = Slider(axWs2, 'Ws2', -1, 1, valinit=Ws2)


axomegat = plt.axes([0.6, 0.1, 0.3, 0.03])
somegat = Slider(axomegat, 'omegat', -5, 5, valinit=omegat)
axWt1 = plt.axes([0.6, 0.3, 0.3, 0.03])
sWt1 = Slider(axWt1, 'Wt1', -1, 1, valinit=Wt1)
axWt2 = plt.axes([0.6, 0.2, 0.3, 0.03])
sWt2 = Slider(axWt2, 'Wt2', -1, 1, valinit=Wt2)

def update(val):
    fig.set_data(f(xgrid, ygrid, sWs1.val, sWs2.val, somegas.val,
                sWt1.val, sWt2.val, somegat.val))

somegas.on_changed(update)
sWs1.on_changed(update)
sWs2.on_changed(update)
sWs2.on_changed(update)

somegat.on_changed(update)
sWt1.on_changed(update)
sWt2.on_changed(update)
sWt2.on_changed(update)

plt.sca(ax)

plt.show()