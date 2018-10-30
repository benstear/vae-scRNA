#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:11:43 2018
@author: stearb
"""

import numpy as np
import pandas as pd
import h5py
import collections
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sp_sparse
import tables
import dask.array as da
import multiprocessing as mp

'''
# genecount: 27998
# total cell count: 1,306,127
#filename = '/Users/dawnstear/desktop/Mid_Atlantic_Poster/sc_data/n_1_3mil/1M_neurons_neuron20k.h5'
filename = '/Users/stearb/desktop/vae-scRNA-master/data/n_1_million/1M_neurons_neuron20k.h5'
f = h5py.File(filename, 'r+')
key = list(f.keys())[0] # mm10
data_dict = f['mm10']
print(list(data_dict.keys()))
data = np.asarray(data_dict['data'])
gene_names = np.asarray(data_dict['gene_names'])
genes = np.asarray(data_dict['genes'])
barcodes = np.asarray(data_dict['barcodes'])
index = np.asarray(data_dict['indices'])
indptr = np.asarray(data_dict['indptr'])
shape = np.asarray(data_dict['shape'])

# genes x barcodes
# 27997 x 20000 
num_cpu_cores = mp.cpu_count()  # 4 cores
############################################3

filename_all = '/Users/stearb/desktop/1M_neurons_filtered_gene_bc_matrices_h5.h5' 
f = h5py.File(filename_all, 'r+')
key = list(f.keys())[0] # mm10
data_dict = f['mm10']
print(list(data_dict.keys()))

data = np.asarray(data_dict['data'])
gene_names = np.asarray(data_dict['gene_names'])
genes = np.asarray(data_dict['genes'])
barcodes = np.asarray(data_dict['barcodes'])
index = np.asarray(data_dict['indices'])
indptr = np.asarray(data_dict['indptr'])
shape = np.asarray(data_dict['shape'])
'''


np.random.seed(0)
GeneBCMatrix = collections.namedtuple('GeneBCMatrix', ['gene_ids', 'gene_names', 'barcodes', 'matrix'])

def get_matrix_from_h5(filename, genome):
    with tables.open_file(filename, 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
            matrix = sp_sparse.csc_matrix((dsets['data'], dsets['indices'], dsets['indptr']), shape=dsets['shape'])
            return GeneBCMatrix(dsets['genes'], dsets['gene_names'], dsets['barcodes'], matrix)
        except tables.NoSuchNodeError:
            raise Exception("Genome %s does not exist in this file." % genome)
        except KeyError:
            raise Exception("File is missing one or more required datasets.")

def save_matrix_to_h5(gbm, filename, genome):
    flt = tables.Filters(complevel=1)
    with tables.open_file(filename, 'w', filters=flt) as f:
        try:
            group = f.create_group(f.root, genome)
            f.create_carray(group, 'genes', obj=gbm.gene_ids)
            f.create_carray(group, 'gene_names', obj=gbm.gene_names)
            f.create_carray(group, 'barcodes', obj=gbm.barcodes)
            f.create_carray(group, 'data', obj=gbm.matrix.data)
            f.create_carray(group, 'indices', obj=gbm.matrix.indices)
            f.create_carray(group, 'indptr', obj=gbm.matrix.indptr)
            f.create_carray(group, 'shape', obj=gbm.matrix.shape)
        except:
            raise Exception("Failed to write H5 file.")
        
def subsample_matrix(gbm, barcode_indices):
    return GeneBCMatrix(gbm.gene_ids, gbm.gene_names, gbm.barcodes[barcode_indices], gbm.matrix[:, barcode_indices])

def get_expression(gbm, gene_name):
    gene_indices = np.where(gbm.gene_names == gene_name)[0]
    if len(gene_indices) == 0:
        raise Exception("%s was not found in list of gene names." % gene_name)
    return gbm.matrix[gene_indices[0], :].toarray().squeeze()


# load matrix (NOTE: takes several minutes, requires 32GB of RAM)
filtered_matrix_h5 = "/Users/stearb/desktop/1M_neurons_filtered_gene_bc_matrices_h5.h5"
genome = "mm10"
gene_bc_matrix = get_matrix_from_h5(filtered_matrix_h5, genome)










