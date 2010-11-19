import numpy as np
import matplotlib.pyplot as plt
from pointingtools import Siam
import re
s=Siam()
plt.figure()
pair = {'a':'b','M':'S'}
for tag,m in s.siam.iteritems():

    print(tag)
    label = tag

    vec=np.dot(m,[1,0,0])
    y=np.dot(m,[0,1,0])*.002

    if y[0] < 0:
        y *= -1

    if re.match('.*[abMS].*',tag) is None:
        plt.plot(vec[0],vec[1],'bs')
    else:
        col = 'k'
        if re.match('.*[bS].*',tag):
            col = 'r'
            label = None
        else:
            label += '+' + pair[tag[-1]]
        plt.plot(np.array([0,y[0]])+vec[0],np.array([0,y[1]]+vec[1]),col)

    if label:
        plt.text(vec[0]-.002, vec[1]-.005, label, fontsize=6)
plt.grid()
plt.ylim([-.09,.09])
plt.show()
