###############
#
# Transform R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
#
###############

import numpy as np
import matplotlib.pyplot as plt

# white noise
wns = []
rws = []
for i in range(20):
    wn = np.random.normal(
        loc=0,      # maen
        scale=1,    # standard deviation
        size=100    # the size of aray
    )
    wns.append(wn)

    rw = np.cumsum(wn)
    rws.append(rw)

# building x
x = np.arange(0, 100)

# visualizing white noise
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
for wn in wns:
    ax1.plot(x, wn)
ax1.set_title('white noise')

# visualizing random walk
ax2 = fig.add_subplot(2, 1, 2)
for rw in rws:
    ax2.plot(x, rw)
ax2.set_title('random walk')

fig.tight_layout()
plt.show()
