#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:29:04 2018

@author: dawnstear
"""

# Create a cluster function
import numpy as np
import sklearn
import matplotlib.pyplot as plt



pca = [0, 2.4, 16.7]
tsne = [0, 48.5, 5502]
iso = [0, 26.4, 9285]
spect = [0, 27.5, 6149]
nnmf = [0, 11, 57]   

n = [0,1078,13313,27499]

plt.plot(pca,n)
plt.plot(tsne,n)
plt.plot(iso,n)
plt.plot(spect,n)
plt.plot(nnmf,n)

plt.legend(['PCA','t-SNE','Isomap','Spectral Embedding','Non-negative Matrix Factorization'])


plt.show()

plt.savefig('/Users/stearb/desktop/times'+'.png')

'''
n 1078 run times
-----------------
pca= 2.4
lda= 6.0  no
umap= 9.6 
nnmf= 10.9
iso= 26.44
spectral= 27.5
tsne= 48.5
lle = 80  no



n 13,313 run times
------------------
-cmd 1st run: qsub -l h_vmem=28G -l m_mem_free=28G ./run_13313.sh
--job ID: 3119618
 $ Memory Error while on LLE
 
-cmd 2nd run: qsub -l h_vmem=52G -l m_mem_free=52G ./run_13313.sh 
-- job ID: 3121707
 $ Memory Error while on LLE

-cmd 3rd run: qsub -l h_vmem=52G -l m_mem_free=52G ./run_13313.sh 
-- job ID: 3121954
--just skip LLE
$ for LDA, labels were different dimension than data

-cmd 4th run: qsub -l h_vmem=52G -l m_mem_free=52G ./run_13313.sh 
--j id= 3122254


pca = 16.7
tsne = 5502


n 27,499 run times
------------------
-cmd 1st run: qsub -l h_vmem=60G -l m_mem_free=60G ./run_27499.sh

-cmd 2nd run: qsub -l h_vmem=50G -l m_mem_free=50G ./run_27499.sh
--j 3123316



'''