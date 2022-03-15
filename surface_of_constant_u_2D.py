import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm      
from matplotlib.widgets import Slider

'''
This code serves to give an intuition for how the parameters of a MP determine the label.
Note how the nonlinearity immediately results in a very different picture. 
u denotes a threshold, and may be set to 0. 
This code also illustrates how the delta-function approach using activation space becomes far more complicated.
''' 

vals = np.linspace(-2,2,100)
xgrid, ygrid = np.meshgrid(vals,vals)   

def f(x, y, argW1, argW2, argomega, u):
    
    ret = np.transpose([argW1*x + argW2*y + 2*argomega*x*y, argW1*x + argW2*y + 2*argomega*x*y, argW1*x + argW2*y + 2*argomega*x*y] )/ 2
    res = argW1*x + argW2*y + 2*argomega*x*y 
    for xidx, x in enumerate(res):
        for yidx, y in enumerate(x):
            if res[yidx, xidx] > u: #Switched coordinates as a lousy correction to the transpose made earlier
                ret[xidx, yidx, 0]= 0
                ret[xidx, yidx, 1]= 1
            elif res[yidx, xidx] < u:
                ret[xidx, yidx, 1]= 0
                ret[xidx, yidx, 0]= 1

    return ret

W1 = 1
W2 = 1
omega = 0
u = 0.5

ax = plt.subplot(111)
plt.subplots_adjust(left=0.15, bottom=0.55)
fig = plt.imshow(f(xgrid, ygrid, W1, W2, omega, u), cm.gray, origin = 'lower')
plt.axis("off")
fig.axes.get_xaxis().set_visible(True)
fig.axes.get_yaxis().set_visible(True)
plt.xlabel("x1")
plt.ylabel("x2")


axomega = plt.axes([0.15, 0.1, 0.65, 0.03])
somega = Slider(axomega, 'omega', -5, 5, valinit=omega)
axW1 = plt.axes([0.15, 0.3, 0.65, 0.03])
sW1 = Slider(axW1, 'W1', -1, 1, valinit=W1)
axW2 = plt.axes([0.15, 0.2, 0.65, 0.03])
sW2 = Slider(axW2, 'W2', -1, 1, valinit=W2)

axu = plt.axes([0.15, 0.4, 0.65, 0.03])
su = Slider(axu, 'u', -10, 10, valinit=u)

def update(val):
    fig.set_data(f(xgrid, ygrid, sW1.val, sW2.val, somega.val, su.val))

somega.on_changed(update)
sW1.on_changed(update)
sW2.on_changed(update)
su.on_changed(update)

plt.sca(ax)
t = np.linspace(-2, 2, 100)
def plotIsoLine(x, argWx, argWy, argomega, u):
    if argomega != 0:
        print((u - argW1 - argW2 ) / (2 * argomega) / (x + argW1/(2 * argomega)) + (argW2/(2 * argomega)))
        return (x,(u - argW1 - argW2 ) / (2 * argomega) / (x + argW1/(2 * argomega)) + (argW2/(2 * argomega)))
    else:
        print("omega is 0")
        return t, t

plt.show()