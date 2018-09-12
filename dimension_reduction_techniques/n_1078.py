#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:12:42 2018

@author: dawnstear
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:42:59 2018

@author: dawnstear
"""

# t-SNE and PCA plot metric and times
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib.pyplot import scatter, figure, subplot, savefig
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

'''https://portals.broadinstitute.org/single_cell/study
/atlas-of-human-blood-dendritic-cells-and-monocytes
Do pie chart of % of each cell class.'''

data  =  pd.read_csv('/scr1/users/stearb/scdata/n_1078/data.csv')  
#data  =  pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_1078/data.csv')  

print(np.shape(data))

np.random.seed(42)
data = shuffle(data)
celltypes = data['TYPE'] # save cell type vector in case we need it later
labels = data['Labels'] # save labels
data_ = data.drop(['Labels','TYPE'],axis=1) 
cellcount, genecount = np.shape(data_)
X = data_
y = labels

np.random.seed(42)
data = shuffle(data)
celltypes = data['TYPE'] # save cell type vector in case we need it later
labels = data['Labels'] # save labels
data_ = data.drop(['Labels','TYPE'],axis=1) 
cellcount, genecount = np.shape(data_)
X = data_
y = labels
#print(np.size(X)) # 28.66 mil


###########  tSNE 1078   ########################
start = time.time()
X_tsne = TSNE(learning_rate=100)
X_tsne = X_tsne.fit_transform(X)
time_elapsed = time.time() - start

fig, ax = plt.subplots()
figure(figsize=(100, 100))
ax.set(xlabel='t-SNE 1', ylabel='t_SNE 2',title='t-SNE: 1078 cells with 10 subtypes')
ax.legend(y)
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)

# path to CHOP mac desktop folder
fig.savefig('/Users/stearb/desktop/metrics/n_1078/tSNE_'+time_elapsed+'.png')

###########   PCA 1078   ########################
start = time.time()
pca = PCA(n_components=2, svd_solver='full')
pca_array = pca.fit_transform(X) 
time_elapsed = time.time() - start
#print(time_elapsed)

fig, ax = plt.subplots()
figure(figsize=(10, 10))
ax.set(xlabel='PCA 1', ylabel='PCA 2',title='PCA: 1078 cells with 10 subtypes')
ax.legend(y)
ax.scatter(pca_array[:, 0], pca_array[:, 1], c=y)

# path to CHOP mac desktop folder
fig.savefig(fig.savefig('/Users/stearb/desktop/metrics/n_1078/PCA_'+time_elapsed+'.png'))



