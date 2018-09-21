#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:27:33 2018

@author: stearb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:00:58 2018

@author: dawnstear
"""

# t-SNE and PCA plot metric and times
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import (manifold, datasets, decomposition, #ensemble,
                     discriminant_analysis, random_projection)
import pandas as pd
import numpy as np
from matplotlib.pyplot import scatter, figure, subplot, savefig
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
import os
import umap
import ZIFA



print('Begining the 13,313 cell dataset...\n...')
### Data Set: 13,313 single nucleus mouse brain 
# https://portals.broadinstitute.org/single_cell/study/dronc-seq-single-nucleus-rna-seq-on-mouse-archived-brain#study-download

#os.mkdir('/scr1/users/stearb/results/plots_13313/')    # only needed if doesnt exist


def plot2D(X,y,method,title,time_elapsed):

    fig, ax = plt.subplots()
    figure(figsize=(20, 20))
    ax.set(xlabel=method+' 1', ylabel=method+' 2',title = title)
    ax.legend(y)
    ax.scatter(X[:, 0], X[:, 1], c=y)
    #fig.savefig('/Users/dawnstear/desktop/Mid_Atlantic_Poster/tSNE_'+str(np.float16(time_elapsed))+'.png')
    fig.savefig('/scr1/users/stearb/results/plots_13313/'+method+'__'+str(np.float16(time_elapsed))+'.png')
    #fig.savefig('/users/stearb/desktop/myfig.png')
    


t = time.time()
try: data13k, labels13k
except:
    #data13k  =  pd.read_csv('/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_13313_singlenuc/Mouse_Processed_GTEx_Data.DGE.log-UMI-Counts.txt',sep='\t')  
    data13k  =  pd.read_csv('/scr1/users/stearb/scdata/n_13313/Mouse_Processed_GTEx_Data.DGE.log-UMI-Counts.txt',sep='\t')  

    data13k = data13k.T
    labels13k  =  pd.read_csv('/scr1/users/stearb/scdata/n_13313/metadata_singleNuc_13k.txt',sep='\t')

#dataloadtime = time.time()-t)
#data13k.values[0:3][0:3] # includes gene names
data13k_clipped = data13k.iloc[1:] # clip them off the top
labels13k_clipped = labels13k.iloc[1:]  # drop first row, info not needed

CELL_NAME = labels13k['NAME']   # check that cell BARCODE's match
CLUSTER_NAME =labels13k['Cluster']
CLUSTER_ID = labels13k['ClusterID']

X = data13k_clipped.values
y = CLUSTER_ID.values
BARCODE = data13k_clipped.iloc[:,1]

# compare CELL_NAME and BARCODE
barcodes = np.asarray(BARCODE[:].index)
cellnames = CELL_NAME.values[1:]
np.array_equal(barcodes,cellnames)

n_neighbors=30
n_components = 2

#################### tSNE ################################
print('Calculating t-SNE..')
start = time.time()
tsne = TSNE(n_components,learning_rate=100)
tsne_array = tsne.fit_transform(X)      # cant do tSNE with data this large
time_elapsed = time.time() - start
#print(time_elapsed//60,time_elapsed%60)
plot2D(tsne_array,y,'tSNE','t-SNE: 13,313 cells with 49 cell subtypes',time_elapsed)


######################## PCA 13,313 #######################
print('Calculating PCA..')
start = time.time()
pca = PCA(n_components, svd_solver='randomized') # use 'auto' not 'full' svd_solver, this will use the 
pca_array = pca.fit_transform(X)             # more efficient 'randomized' solver 
time_elapsed = time.time() - start
plot2D(pca_array,y,'PCA','PCA: 13,313 cells with 49 cell subtypes',time_elapsed)



###############################################
# Local Linear Embedding
print('Calculating LLE..')
start = time.time()
lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components)
lle_array = lle.fit_transform(X)
time_elapsed = time.time() - start
plot2D(lle_array,y,'LLE','LLE: 13,313 cells with 49 cell subtypes',time_elapsed)
#lle_params = lle_array.get_params()

#################### LDA ##############################
print('Calculating LDA...')
start = time.time()
lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='svd',n_components=2)
lda_array = lda.fit_transform(X, y)
time_elapsed = int(time.time() - start)
plot2D(lda_array,y,'LDA','LDA: 13,313 cells with 49 cell subtypes',time_elapsed)


##################### Isomap ##############################
print('Calculating Isomap....')
t = time.time()
iso = manifold.Isomap(n_neighbors, n_components=2,eigen_solver='auto')
iso_array = iso.fit_transform(X)
time_elapsed = time.time() - t
plot2D(iso_array,y,'Isomap','Isomap: 13,313 cells with 49 cell subtypes',time_elapsed)


################### Spectral ###################################
print('Calculating spectral embedding....')
t = time.time()
spectral = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver='amg')
spectral_array = spectral.fit_transform(X)
time_elapsed = time.time() - t
plot2D(spectral_array,y,'Spectral','Spectral Embedding: 13,313 cells with 49 cell subtypes',time_elapsed)


######################  NNMF ###############################
print('Calculating NNMF.......')
t =  time.time()
nnmf = decomposition.NMF(n_components=2, init='random', random_state=0)
nnmf_array = nnmf.fit_transform(X)
time_elapsed = time.time() - t
plot2D(nnmf_array,y,'NNMF','Non-Negative Matrix Factorization: \n13,313 cells with 49 cell subtypes',time_elapsed)

###################### UMAP  ##############################
print('Calculating UMAP......')
t =  time.time()
umap_array = umap.UMAP().fit_transform(X)
time_elapsed = time.time() - t
plot2D(umap_array,y,'UMAP','Uniform Manifold Approximation and Projection:\n13,313 cells with 49 subtypes',time_elapsed)

'''
################ ZIFA ############################
from ZIFA import ZIFA
from ZIFA import block_ZIFA
k=2
print('Calculating ZIFA......')
t =  time.time()
#zifa_array, model_params = ZIFA.fitModel(X, k)
zifa_array, model_params = block_ZIFA.fitModel(X, k) #default blocksize is genes/500
time_elapsed = time.time() - t
plot2D(zifa_array,y,'ZIFA','Zero Inflated Dimensionality Reduction',time_elapsed)
'''

