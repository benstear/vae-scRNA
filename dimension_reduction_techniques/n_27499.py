#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:44:44 2018

@author: dawnstear
"""
# https://portals.broadinstitute.org/single_cell/study/retinal-bipolar-neuron-drop-seq
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import (manifold, datasets, decomposition, #ensemble,
                     discriminant_analysis, random_projection)
import pandas as pd
import numpy as np
from matplotlib.pyplot import scatter, figure, subplot, savefig
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.utils import shuffle
import time
#import umap

'''
def plot2D(X,y,method,title,time_elapsed):
    path='/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data'
    fig, ax = plt.subplots()
    figure(figsize=(10, 10))
    ax.set(xlabel=method+' 1', ylabel=method+' 2',title = title)
    ax.legend(y)
    ax.scatter(X[:, 0], X[:, 1], c=y)
    #fig.savefig('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_13313/tSNE_'+
    #            str(np.float16(time_elapsed))+'.png')
    fig.savefig(path+'/n_27499_bipolar/'+method+'_'+str(np.float16(time_elapsed))+'.png')
'''


t = time.time()
#data27k  =  pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_27499_bipolar/exp_matrix.txt',sep='\t')  
data27k  =  pd.read_csv('/scr1/users/stearb/scdata/n_27499/exp_matrix.txt',sep='\t')  
#labels27k = pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_27499_bipolar/clust_retinal_bipolar.txt',sep='\t')
labels27k  =  pd.read_csv('/scr1/users/stearb/scdata/n_27499/clust_retinal_bipolar.txt',sep='\t')

#data27k = pd.read_csv('/Users/stearb/desktop/vae-scRNA-master/data/n_27499/exp_matrix.txt')
#labels27k = pd.read_csv('/Users/stearb/desktop/vae-scRNA-master/data/n_27499/clust_retinal_bipolar.txt')

time_elapsed = time.time() - t
print(time_elapsed)
#print(np.shape(data27k))
#print(np.shape(labels27k))

data27k = data27k.T
CLUSTERS = labels27k['CLUSTER']

y = CLUSTERS.iloc[1:]
data27k_clipped = data27k.iloc[1:]
X = data27k_clipped.values
print('Done loading...')
'''
ValueError: c of shape (27499,) not acceptable as a color sequence for
 x with size 27499, y with size 27499
 
 new_list = [y[i] for i in range(0, len(y))] ''' # NEED MEDIAN # OF GENES!!!!!

#assert np.array_equal(barcodes,cellnames)

n_neighbors=30
n_components = 2
'''
###########  tSNE   ########################
start = time.time()
tsne = TSNE(learning_rate=100)
tsne_array = tsne.fit_transform(X)
time_elapsed = time.time() - start
plot2D(tsne_array,y,'tSNE','t-SNE: 27,500 cells with 15 cell subtypes',time_elapsed)
'''
########################## PCA ########################
t = time.time()
pca = PCA(n_components=2, svd_solver='auto')
pca_array = pca.fit_transform(X) 
time_elapsed = time.time() - t
print('pca time = %s ' % time_elapsed)
#plot2D(pca_array,y,'PCA','PCA: 27,500 cells with 15 subtypes', time_elapsed)

'''
###################### LDA ###########################
t= time.time()
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
lda_array = lda.fit_transform(X, y)
time_elapsed = int(time.time() - t)
#plot2D(lda_array,y,'LDA','LDA: 27,500 cells with 15 subtypes',time_elapsed)
'''

########################Isomap ################################
t = time.time()
iso = manifold.Isomap(n_neighbors, n_components=2)
iso_array = iso.fit_transform(X)
time_elapsed = time.time() - t
print('iso time = %s ' % time_elapsed)

#plot2D(iso_array,y,'isomap','Isomap: 27,500 cells with 15 subtypes',time_elapsed)

####################### Spectral Embedding #########################
t = time.time()
spectral = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
spectral_array = spectral.fit_transform(X)
time_elapsed = time.time() - t
#plot2d(spectral_array,y,method='spectral',
#       title='Spectral Embedding: 27,500 cells with 15 subtypes',time_elapsed)
print('spectral time = %s ' % time_elapsed)

###################### NNMF ##########################
t =  time.time()
nnmf = decomposition.NMF(n_components=2, init='random', random_state=0)
nnmf_array = nnmf.fit_transform(X)
time_elapsed = time.time() - t
print('nnmf time = %s ' % time_elapsed)
#plot2D(nnmf_array,y,method='nnmf',
#       title='Non negative matrix factorization:\n27,500 cells with 15 subtypes',time_elapsed)
'''
################ UMAP ##############################
t =  time.time()
umap_array = umap.UMAP().fit_transform(X)
time_elapsed = time.time() - t
print('umap time = %s ' % time_elapsed)
#plot2D(umap_array,y,'UMAP',
#       'Uniform Manifold Approximation and Projection:\n27,500 cells with 15 subtypes',time_elapsed)
'''

################ ZIFA ############################



