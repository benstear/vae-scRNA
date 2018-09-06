#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:08:44 2018

@author: dawnstear
"""
import numpy as np
import os
from inspect import signature
import datetime
import subprocess # replaces os.system()
import pandas as pd
import tensorflow as tf


def next_batch(datamatrix,batch_size=0,buffer_size=0):  #need num_batches ?
    """
    Data input pipeline function
    
    Parameters
    ----------
    
    datamatrix : numpy array
    -Should be a square, numeric numpy array with cell samples as rows and
    genes as columns. Labels should NOT be included in the datamatrix, this 
    function is designed for an unsupervised learning model.
    
    """
    #cellcount, genecount = np.shape(datamatrix)
    #if batch_size > datamatrix.axis(0)
    
    # Set some constants
    cellcount, genecount = np.shape(datamatrix)
    
    if not batch_size:
        BATCH_SIZE = tf.cast(0.2*len(datamatrix), tf.int64) # default is 1/5 total cells
    else:
        BATCH_SIZE = tf.cast(batch_size, tf.int64) # make sure batch_size is int64
            
    if not buffer_size:
        BUFFER_SIZE = 55
    else:
        BUFFER_SIZE = tf.cast(buffer_size,tf.int64)
        

    
    # create DATASET OBJECT from numpy array, repeat prevents OutOfRange Error so we can loop through data indefinitely
    dataset = tf.data.Dataset.from_tensor_slices((datamatrix)
    ).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE,drop_remainder=True)
        
    # create ITERATOR OBJECT
    # The iterator arising from this method can only be initialized and run once – it can’t be re-initialized. 
    # iterator = dataset.make_one_shot_iterator() # one shot iter is the simplest kind, runs through data once
    
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    
    return next_batch
        
        