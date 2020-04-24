import pandas as pd
import matplotlib.pyplot as plt
import time

from NMF import NMF
from utils import plot_gallery

data = pd.read_csv('data.txt', sep='\t', header=None).values.T
print(data.shape)
t = time.time()

W, H = NMF(data, 8, 100, 1e-4)
print(H.shape, W.shape)

plot_gallery('%s - Train time %.1fs' % ('Non-negative components - NMF', time.time() - t),
             W.T,
             4,
             2,
             (64, 64))
plt.show()