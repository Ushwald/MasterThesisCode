from perceptron import BinPerceptron
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class BinNPC:
    """This version of the NPC takes in symmetric binary inputs (-1 or 1)
    It works simply by implicitly defining an N^2 perceptron and assigning the appropriate N^2 weights"""

    def __init__(self, N: int):
        # initialize a random MNPC with M clusters in the layer
        self.N = N
        # Initialize NPC paremeters however you wish. Actual restrictions on NPC parameters should also go into the training function
        # Normalize W, and experiment with restrictions on omega
        self.W = (np.random.rand(N) - 0.5) * 2
        self.W = self.W / np.linalg.norm(self.W)# * N 
        self.omega_i = (np.random.rand(N, N) - 0.5)# * N 
        for i in range(N):
            self.omega_i[i, i] = 0
            for j in range(i):
                self.omega_i[i, j] = self.omega_i[j, i]

    def getActivations(self, examples):
        activations = []
        for example in np.transpose(examples):
            activation =  np.dot(self.W, example)
            for i in range(self.N):
                for j in range(self.N):
                    activation += self.omega_i[i, j] * example[i] * example[j]
            activations.append(activation)
        return activations

def getX(N):
    return np.random.normal(0.0, 1.0, N)


N = 8
p = 10000
alpha_val = min(1.0 / p * 5000, 1.0)
npc1 = BinNPC(N)
npc2 = BinNPC(N)


#npc2.omega_i = np.zeros(shape = npc2.omega_i.shape)
#npc1.omega_i = np.zeros(shape = npc1.omega_i.shape)

#print("W: {}".format(npc2.W))
#print("Omega: {}".format(npc2.omega_i))


"""
for idx, row in enumerate(npc1.omega_i):
    for jdx, item in enumerate(row):
        if not (jdx - idx == 2 or idx - jdx == 2):
            npc1.omega_i[idx, jdx] = 0
            npc2.omega_i[idx, jdx] = 0
print(npc1.omega_i)
"""



# Create a point cloud plot of u, \underscore u activations for gaussian distributed inputs

def setParamsToZero():
    npc1.W = np.zeros(shape = npc1.W.shape)
    npc2.W = np.zeros(shape = npc2.W.shape)
    npc1.omega_i = np.zeros(shape = npc1.omega_i.shape)
    npc2.omega_i = np.zeros(shape = npc2.omega_i.shape)
#setParamsToZero()

points = np.random.normal(0.0, 1.0, (N, p))
activations1 = npc1.getActivations(points)
activations2 = npc2.getActivations(points)

fig = plt.figure(figsize=(20, 10), dpi=80)
axes = plt.subplot(111)
plt.subplots_adjust(right = 0.5)
plt.xlabel("npc1 activation")
plt.ylabel("npc2 activation")

plt.scatter(activations1, activations2, color = 'red', alpha = alpha_val)
plt.scatter(points[0], points[1], color = 'blue', alpha = alpha_val)


axislim = max(- 0.7 * min(axes.get_xlim()[0], axes.get_ylim()[0]),0.7 * max(axes.get_xlim()[1], axes.get_ylim()[1]))
axes.set_xlim([-axislim, axislim])
axes.set_ylim([-axislim, axislim])



def plot():
    activations1 = npc1.getActivations(points)
    activations2 = npc2.getActivations(points)
    plt.sca(axes)
    axes.clear()
    plt.scatter(activations1, activations2, color = 'red', alpha = sActivationAlpha.val)
    plt.scatter(points[0], points[1], color = 'blue', alpha = sInputAlpha.val)
    #axislim = [0.7 * min(axes.get_xlim()[0], axes.get_ylim()[0]),0.7 * max(axes.get_xlim()[1], axes.get_ylim()[1])]
    axes.set_xlim([-sZoom.val, sZoom.val])
    axes.set_ylim([-sZoom.val, sZoom.val])
    plt.xlabel("npc1 activation")
    plt.ylabel("npc2 activation")

def update(val):
    print('updating!')
    npc1.W = np.array([sW1[n].val for n in range(N)])
    npc2.W = np.array([sW2[n].val for n in range(N)])
    for n1 in range(1, N):
        for n2 in range(0, n1):
            npc1.omega_i[n1, n2] = sOm1[n1-1][n2].val
            npc2.omega_i[n1, n2] = sOm2[n1-1][n2].val
            npc1.omega_i[n2, n1] = sOm1[n1-1][n2].val
            npc2.omega_i[n2, n1] = sOm2[n1-1][n2].val
    plot()
    


# procedurally create sliders for all variables:
# first the W's
axW1 = []
axW2 = []
sW1 = []
sW2 = []

for n in range(N):
    axW1.append(plt.axes([0.6 + 0.4 / N * n, 0.90, 0.20 / N, 0.03]))
    axW2.append(plt.axes([0.6 + 0.4 / N * n, 0.85, 0.20 / N, 0.03]))

    labels = ["{}".format(n) if n > 0 else "NPC{} W_:{}".format(i+1, n) for i in range(2)]

    sW1.append(Slider(axW1[n], labels[0], -1, 1, valinit=npc1.W[n]))
    sW2.append(Slider(axW2[n], labels[1], -1, 1, valinit=npc2.W[n]))
    sW1[n].on_changed(update)
    sW2[n].on_changed(update)

#Next up: the omega's:
axOm1 = []
axOm2 = []
sOm1 = []
sOm2 = []
for n1 in range(1, N):
    axOm1.append([])
    axOm2.append([])
    sOm1.append([])
    sOm2.append([])

    for n2 in range(0, n1):
        axOm1[n1-1].append(plt.axes([0.5 + 0.4/N * (n1), 0.75 - 0.3/N*n2, 0.20 / N, 0.03]))
        axOm2[n1-1].append(plt.axes([0.5 + 0.4/N * (n1), 0.40 - 0.3/N*n2, 0.20 / N, 0.03]))

        labels = ["{},{}".format(n1, n2) if (n1 > 1 or n2 > 0) else "NPC{} Om_:{},{}".format(i+1, n1, n2) for i in range(2)]

        sOm1[n1-1].append(Slider(axOm1[n1-1][n2], labels[0], -1, 1, valinit=npc1.omega_i[n1, n2]))
        sOm2[n1-1].append(Slider(axOm2[n1-1][n2], labels[1], -1, 1, valinit=npc2.omega_i[n1, n2]))
        sOm1[n1-1][n2].on_changed(update)
        sOm2[n1-1][n2].on_changed(update)




axInputAlpha = plt.axes([0.6, 0.1, 0.06, 0.03])
axActiavtionAlpha = plt.axes([0.75, 0.1, 0.06, 0.03])
axZoom = plt.axes([0.90, 0.1, 0.06, 0.03])
sInputAlpha = Slider(axInputAlpha, "Input Opacity", 0, .6, valinit = alpha_val)
sActivationAlpha = Slider(axActiavtionAlpha, "Acivation Opacity", 0, .6, valinit = alpha_val)
sZoom = Slider(axZoom, "Zoom", 0, axislim * 2, valinit = axislim)
sInputAlpha.on_changed(update)
sActivationAlpha.on_changed(update)
sZoom.on_changed(update)

axKillW1 = plt.axes([0.6, 0.95, 0.04, 0.03])
axKillW2 = plt.axes([0.7, 0.95, 0.04, 0.03])
axKillOm1 = plt.axes([0.8, 0.95, 0.04, 0.03])
axKillOm2 = plt.axes([0.9, 0.95, 0.04, 0.03])
bKillW1 = Button(axKillW1, "W1 to 0")
bKillW2 = Button(axKillW2, "W2 to 0")
bKillOm1 = Button(axKillOm1, "Om1 to 0")
bKillOm2 = Button(axKillOm2, "Om2 to 0")

def killW1(event):
    for n in range(N):
        sW1[n].eventson = False
        npc1.W[n] = 0
        sW1[n].set_val(npc1.W[n])
        sW1[n].eventson = True
    update(None)
def killW2(event):
    for n in range(N):
        sW2[n].eventson = False
        npc2.W[n] = 0
        sW2[n].set_val(npc2.W[n])
        sW2[n].eventson = True
    update(None)
def killOm1(event):
    for n1 in npc1.omega_i:
        for n2 in n1:
            n2 = 0
    for n1 in sOm1:
        for n2 in n1:
            n2.eventson = False
            n2.set_val(0)
            n2.eventson = True
    
    update(None)
def killOm2(event):
    for n1 in npc2.omega_i:
        for n2 in n1:
            n2 = 0
    for n1 in sOm2:
        for n2 in n1:
            n2.eventson = False
            n2.set_val(0)
            n2.eventson = True
    update(None)

bKillW1.on_clicked(killW1)
bKillW2.on_clicked(killW2)
bKillOm1.on_clicked(killOm1)
bKillOm2.on_clicked(killOm2)


plt.show()
