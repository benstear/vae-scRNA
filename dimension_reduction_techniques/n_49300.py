#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:57:23 2018

@author: dawnstear
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.pyplot import scatter, figure, subplot, savefig
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
#import umap
import numpy as np
print('Starting to download 49,300 cell dataset')
#data49k = pd.read_csv('/Users/stearb/desktop/vae-scRNA-master/data/n_49300/GSE63472_P14Retina_merged_digital_expression.txt', sep='\t')
data49k  =  pd.read_csv('/scr1/users/stearb/scdata/n_493000/GSE63472_P14Retina_merged_digital_expression.txt',sep='\t')  
#data49k  =  pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_49300/GSE63472_P14Retina_merged_digital_expression.txt',sep='\t')  
#X = data49k_clipped.values
print('Done loading')

n_neighbors=30
n_components = 2
'''
###########  tSNE   ########################
start = time.time()
tsne = TSNE(learning_rate=100)
tsne_array = tsne.fit_transform(X)
time_elapsed = time.time() - start
plot2D(tsne_array,y,'tSNE','t-SNE: 27,500 cells with 15 cell subtypes',time_elapsed)

########################## PCA ########################
t = time.time()
pca = PCA(n_components=2, svd_solver='auto')
pca_array = pca.fit_transform(X) 
time_elapsed = time.time() - t
print('pca time = %s ' % time_elapsed)


#plot2D(pca_array,y,'PCA','PCA: 27,500 cells with 15 subtypes', time_elapsed)


###################### LDA ###########################
t= time.time()
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
lda_array = lda.fit_transform(X, y)
time_elapsed = int(time.time() - t)
#plot2D(lda_array,y,'LDA','LDA: 27,500 cells with 15 subtypes',time_elapsed)


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

################ UMAP ##############################
t =  time.time()
umap_array = umap.UMAP().fit_transform(X)
time_elapsed = time.time() - t
print('umap time = %s ' % time_elapsed)
#plot2D(umap_array,y,'UMAP',
#       'Uniform Manifold Approximation and Projection:\n27,500 cells with 15 subtypes',time_elapsed)

'''
################ ZIFA ############################














