#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:12:42 2018

@author: dawnstear
"""


'''https://portals.broadinstitute.org/single_cell/study
/atlas-of-human-blood-dendritic-cells-and-monocytes
Do pie chart of % of each cell class.'''

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, figure, subplot, savefig
plt.switch_backend('agg')
from sklearn.utils import shuffle
import time
import os
import umap

def plot2D(X,y,method,title,time_elapsed):

    fig, ax = plt.subplots()
    figure(figsize=(20, 20))
    ax.set(xlabel=method+' 1', ylabel=method+' 2',title = title)
    ax.legend(y)
    ax.scatter(X[:, 0], X[:, 1], c=y)
    #fig.savefig('/Users/dawnstear/desktop/Mid_Atlantic_Poster/tSNE_'+str(np.float16(time_elapsed))+'.png')
    fig.savefig('/scr1/users/stearb/results/plots_1078/'+method+str(np.float16(time_elapsed))+'.png')
    #fig.savefig('/users/stearb/desktop/myfig.png')
    
#os.mkdir('/scr1/users/stearb/results/plots_1078/')    
    
data  =  pd.read_csv('/scr1/users/stearb/scdata/n_1078/data.csv')  
#data = pd.read_csv('/Users/stearb/desktop/vae-scRNA-master/data/n_1078/data.csv')
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

n_neighbors = 30


###########  tSNE 1078   ########################
print('Calculating t-SNE..')
start = time.time()
tsne = TSNE(learning_rate=100)
tsne_array = tsne.fit_transform(X)
time_elapsed = time.time() - start
plot2D(tsne_array,y,'tSNE','t-SNE: 1078 cells with 10 cell subtypes',time_elapsed)

########################## PCA ########################
print('Calculating PCA..')
t = time.time()
pca = PCA(n_components=2, svd_solver='randomized')
pca_array = pca.fit_transform(X) 
time_elapsed = time.time() - t
plot2D(pca_array,y,'PCA','PCA: 1078 cells with 10 cell subtypes', time_elapsed)


########### LLE 1078 ##########################
print('Calculating LLE..')

start = time.time()
lle = manifold.LocallyLinearEmbedding(n_neighbors,n_components=2,method='standard')
lle_array = lle.fit_transform(X)
time_elapsed = time.time() - start
plot2D(lle_array,y,'LLE','Local Linear Embedding: 1078 cells with 10 cell subtypes',time_elapsed)


###################### LDA ###########################
print('Calculating LDA...')
t= time.time()
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
lda_array = lda.fit_transform(X, y)
time_elapsed = int(time.time() - t)
plot2D(lda_array,y,'LDA','LDA: 1078 cells with 10 subtypes',time_elapsed)

########################Isomap ################################
print('Calculating Isomap....')
t = time.time()
iso = manifold.Isomap(n_neighbors, n_components=2)
iso_array = iso.fit_transform(X)
time_elapsed = time.time() - t
plot2D(iso_array,y,'isomap','Isomap: 1078 cells with 10 subtypes',time_elapsed)

####################### Spectral Embedding #########################
print('Calculating spectral embedding....')
t = time.time()
spectral = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
spectral_array = spectral.fit_transform(X)
time_elapsed = time.time() - t
plot2D(spectral_array,y,'spectral','Spectral Embedding: 1078 cells with 10 subtypes',time_elapsed)

###################### NNMF ##########################
print('Calculating NNMF.......')
t =  time.time()
nnmf = decomposition.NMF(n_components=2, init='random', random_state=0)
nnmf_array = nnmf.fit_transform(X)
time_elapsed = time.time() - t
plot2D(nnmf_array,y,'nnmf','Non negative matrix factorization:\n1078 cells with 10 subtypes',time_elapsed)

################ UMAP ##############################
import umap
print('Calculating UMAP......')
t =  time.time()
umap_array = umap.UMAP().fit_transform(digits.data)
time_elapsed = time.time() - t
plot2D(umap_array,y,'UMAP','Uniform Manifold Approximation and Projection:\n1078 cells with 10 subtypes',time_elapsed)


################ ZIFA ############################
from ZIFA import ZIFA
from ZIFA import block_ZIFA
k=2
print('Calculating ZIFA......')
t =  time.time()
zifa_array, model_params = ZIFA.fitModel(X, k)
#zifa_array, model_params = block_ZIFA.fitModel(Y, k) default blocksize is genes/500
time_elapsed = time.time() - t
plot2D(zifa_array,y,'ZIFA','Zero Inflated Dimensionality Reduction',time_elapsed)




