import numpy as np, math, matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import animation

# Create figure
fig = plt.figure()    
ax = fig.gca()

# Axes labels and title are established
ax = fig.gca()
ax.set_xlabel('x1')
ax.set_ylabel('x2')

ax.set_ylim(-2,2)
ax.set_xlim(-2,2)
plt.gca().set_aspect('equal', adjustable='box')


Ws = [1, 1]
Wt = [1.4, 0.4]
omega = 0.5

def init():
    pass

def animate(t):

    ax.clear() 
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)
    plt.gca().set_aspect('equal', adjustable='box')


    ax.add_patch(plt.Arrow(0, 0, Ws[0], Ws[1], width = 0.2, color = 'blue'))
    ax.add_patch(plt.Arrow(0, 0, Wt[0], Wt[1], width = 0.2, color = 'red'))



    x = np.random.normal(1)
    y = np.random.normal(1)
    print(x, y)


    ax.add_patch(plt.Circle((x, y), radius = .03, color = 'black'))

    return

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               interval=100,
                               blit=False)

plt.show()
